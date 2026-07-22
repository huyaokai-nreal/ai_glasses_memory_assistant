package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.Closeable
import java.util.UUID
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class NativeModelPipeline(
    context: Context,
    pack: InstalledModelPack,
    private val ownerId: String,
    private val captureId: String,
    private val onWakeDetected: (String) -> Unit,
    private val onPartial: (String) -> Unit,
    private val onEnrollmentProgress: (String, String, Int, Int, String) -> Unit,
    private val onReplyQueued: (String) -> Unit,
    private val onFailure: (Throwable) -> Unit,
) : PcmFrameSink, Closeable {
    private data class Frame(val samples: ShortArray, val capturedAtNanos: Long)

    private val queue = ArrayBlockingQueue<Frame>(MAX_QUEUED_FRAMES)
    private val running = AtomicBoolean(true)
    private val sessionId = "android-${UUID.randomUUID()}"
    private val vad = SherpaVadAdapter(context, pack)
    private val keyword = SherpaKeywordAdapter(context, pack)
    private val onlineAsr = SherpaOnlineAsrAdapter(context, pack)
    private val ambientAsr = SherpaAmbientAsrAdapter(context, pack)
    private val speaker = SherpaSpeakerAdapter(context, pack)
    private val speakerModelName = pack.manifest.components.getValue("speaker")
        .options.optString("model_name", "sherpa-speaker")
    private var wakeDeadlineMillis = 0L
    private var wakeKeyword = ""
    private var totalSamples = 0L
    private var onlineText = ""
    private val enrollmentLock = Any()
    private var enrollmentCommand: EnrollmentCommand? = null
    private var enrollmentSessionId = ""
    private var enrollmentSampleCount = 0
    private val worker = Thread(::runLoop, "sherpa-native-audio").apply { start() }

    private sealed interface EnrollmentCommand {
        data class Start(val sessionId: String) : EnrollmentCommand
        data object Cancel : EnrollmentCommand
    }

    override fun accept(pcm16: ShortArray, sampleCount: Int, capturedAtNanos: Long) {
        if (!running.get()) return
        val frame = Frame(pcm16.copyOf(sampleCount), capturedAtNanos)
        if (!queue.offer(frame) && running.getAndSet(false)) {
            onFailure(IllegalStateException("本地模型处理速度低于实时收音，已安全停止"))
            worker.interrupt()
        }
        NativeAudioState.setInferenceQueueDepth(queue.size)
    }

    fun startEnrollment(sessionId: String) {
        synchronized(enrollmentLock) { enrollmentCommand = EnrollmentCommand.Start(sessionId) }
    }

    fun cancelEnrollment() {
        synchronized(enrollmentLock) { enrollmentCommand = EnrollmentCommand.Cancel }
    }

    override fun close() {
        if (running.getAndSet(false)) worker.interrupt()
        if (Thread.currentThread() !== worker) worker.join(STOP_JOIN_MILLIS)
    }

    private fun runLoop() {
        try {
            while (running.get() || queue.isNotEmpty()) {
                val frame = queue.poll(250, TimeUnit.MILLISECONDS) ?: continue
                process(frame)
                NativeAudioState.setInferenceQueueDepth(queue.size)
            }
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (error: Throwable) {
            if (running.getAndSet(false)) onFailure(error)
        } finally {
            runCatching { vad.close() }
            runCatching { keyword.close() }
            runCatching { onlineAsr.close() }
            runCatching { ambientAsr.close() }
            runCatching { speaker.close() }
            queue.clear()
            NativeAudioState.setInferenceQueueDepth(0)
        }
    }

    private fun process(frame: Frame) {
        applyEnrollmentCommand()
        expireWakeIfNeeded()
        val floats = FloatArray(frame.samples.size) { index -> frame.samples[index] / 32768f }
        totalSamples += floats.size
        val completedSegments = vad.accept(floats)
        if (completedSegments.isNotEmpty()) NativeAudioState.addVadSegments(completedSegments.size)
        if (enrollmentSessionId.isNotBlank()) {
            completedSegments.forEach(::consumeEnrollmentSegment)
            return
        }
        var wakeJustDetected = false
        if (!wakePending()) {
            keyword.accept(floats)?.let { detected ->
                wakeKeyword = detected
                wakeDeadlineMillis = SystemClock.elapsedRealtime() + WAKE_QUERY_TIMEOUT_MILLIS
                onlineText = ""
                onlineAsr.reset()
                wakeJustDetected = true
                onWakeDetected(detected)
            }
        }
        if (wakeJustDetected) {
            vad.flush()
            vad.reset()
            onlineAsr.reset()
            return
        }
        if (wakePending()) {
            val partial = onlineAsr.accept(floats).first
            if (partial.isNotBlank() && partial != onlineText) {
                onlineText = partial
                onPartial(partial)
            }
        }
        completedSegments.forEach(::consumeSegment)
    }

    private fun consumeSegment(segment: com.k2fsa.sherpa.onnx.SpeechSegment) {
        val segmentStart = segment.start.toLong()
        val segmentEnd = segmentStart + segment.samples.size
        val lane = if (wakePending()) "assistant" else "ambient"
        val recognition = if (lane == "assistant") null else ambientAsr.recognize(segment.samples)
        val text = if (lane == "assistant") onlineAsr.finish().ifBlank { onlineText.trim() } else recognition?.text.orEmpty()
        val embedding = speaker.compute(segment.samples)
        val speakerState = PythonRuntime.classifySpeaker(
            ownerId,
            embedding ?: FloatArray(0),
            speakerModelName,
        )
        val eventType = if (text.isBlank()) "speech_rejected" else "transcript_final"
        val event = JSONObject()
            .put("schema_version", "audio_event.v1")
            .put("event_id", UUID.randomUUID().toString())
            .put("audio_session_id", sessionId)
            .put("segment_id", "segment-${UUID.randomUUID()}")
            .put("type", eventType)
            .put("lane", lane)
            .put("source_type", if (lane == "assistant") "wake_query" else "ambient_audio")
            .put("start_ms", samplesToMillis(segmentStart))
            .put("end_ms", samplesToMillis(segmentEnd))
            .put("text", text)
            .put("final", true)
            .put("vad", JSONObject().put("state", "speech_end").put("backend", "sherpa_silero"))
            .put(
                "wake",
                JSONObject()
                    .put("detected", lane == "assistant")
                    .put("keyword", if (lane == "assistant") wakeKeyword else "")
                    .put("backend", "sherpa_kws"),
            )
            .put(
                "asr",
                JSONObject()
                    .put("backend", if (lane == "assistant") "sherpa_online" else "sherpa_sensevoice")
                    .put("language", recognition?.language.orEmpty()),
            )
            .put("speaker", speakerState)
            .put("overlap", overlapMetadata(segment.samples, embedding))
            .put("audio_retention", "discarded_after_processing")
        val privateEvent = JSONObject()
        if (embedding != null) {
            val values = org.json.JSONArray()
            embedding.forEach(values::put)
            privateEvent.put("speaker_embedding", values)
            privateEvent.put("speaker_embedding_model", speakerModelName)
        }
        val queueResult = PythonRuntime.ingestAudioEvent(ownerId, captureId, event, privateEvent)
        NativeAudioState.markFinal(ambient = lane == "ambient", rejected = text.isBlank())
        if (lane == "assistant" && queueResult.optString("status") in setOf("pending", "running")) {
            onReplyQueued(event.getString("event_id"))
        }
        if (lane == "assistant") clearWake()
        if (lane == "assistant") onPartial("")
    }

    private fun consumeEnrollmentSegment(segment: com.k2fsa.sherpa.onnx.SpeechSegment) {
        val sessionId = enrollmentSessionId
        if (sessionId.isBlank()) return
        val embedding = speaker.compute(segment.samples)
        if (embedding == null) {
            onEnrollmentProgress("error", sessionId, enrollmentSampleCount, ENROLLMENT_SAMPLE_TOTAL, "语音太短，请重新朗读这一段")
            return
        }
        val sampleIndex = enrollmentSampleCount + 1
        val event = JSONObject()
            .put("schema_version", "audio_event.v1")
            .put("event_id", UUID.randomUUID().toString())
            .put("audio_session_id", sessionId)
            .put("segment_id", "enrollment-${UUID.randomUUID()}")
            .put("type", "speaker_update")
            .put("lane", "enrollment")
            .put("source_type", "speaker_enrollment")
            .put("start_ms", 0)
            .put("end_ms", samplesToMillis(segment.samples.size.toLong()))
            .put("text", "")
            .put("final", true)
            .put("vad", JSONObject().put("state", "speech_end").put("backend", "sherpa_silero"))
            .put("wake", JSONObject())
            .put("asr", JSONObject())
            .put("speaker", JSONObject().put("state", "enrollment").put("model", speakerModelName))
            .put("overlap", JSONObject().put("state", "unknown").put("reason", "enrollment_not_evaluated"))
            .put("audio_retention", "discarded_after_processing")
        val privateEvent = JSONObject()
            .put("speaker_embedding", org.json.JSONArray().apply { embedding.forEach(::put) })
            .put("speaker_embedding_model", speakerModelName)
            .put("enrollment_session_id", sessionId)
            .put("sample_index", sampleIndex)
            .put("sample_total", ENROLLMENT_SAMPLE_TOTAL)
        val queued = PythonRuntime.ingestAudioEvent(ownerId, "", event, privateEvent)
        enrollmentSampleCount = sampleIndex
        if (sampleIndex >= ENROLLMENT_SAMPLE_TOTAL) {
            enrollmentSessionId = ""
            onEnrollmentProgress("processing", sessionId, sampleIndex, ENROLLMENT_SAMPLE_TOTAL, "")
            val completed = PythonRuntime.waitAudioEvent(
                ownerId,
                queued.getString("event_id"),
                ENROLLMENT_SAVE_TIMEOUT_SECONDS,
            )
            val result = completed.optJSONObject("dispatch")?.optJSONObject("result")
            if (completed.optString("status") == "completed" && result?.optString("status") == "ok") {
                onEnrollmentProgress("completed", sessionId, sampleIndex, ENROLLMENT_SAMPLE_TOTAL, "")
            } else {
                val detail = completed.optString("error_type").ifBlank { "声纹保存超时，请重试" }
                onEnrollmentProgress("error", sessionId, sampleIndex, ENROLLMENT_SAMPLE_TOTAL, detail)
            }
        } else {
            onEnrollmentProgress("recording", sessionId, sampleIndex, ENROLLMENT_SAMPLE_TOTAL, "")
        }
    }

    private fun applyEnrollmentCommand() {
        val command = synchronized(enrollmentLock) {
            enrollmentCommand.also { enrollmentCommand = null }
        } ?: return
        vad.flush()
        vad.reset()
        clearWake()
        onPartial("")
        when (command) {
            is EnrollmentCommand.Start -> {
                enrollmentSessionId = command.sessionId
                enrollmentSampleCount = 0
                onEnrollmentProgress("recording", command.sessionId, 0, ENROLLMENT_SAMPLE_TOTAL, "")
            }
            EnrollmentCommand.Cancel -> {
                val cancelled = enrollmentSessionId
                enrollmentSessionId = ""
                enrollmentSampleCount = 0
                onEnrollmentProgress("cancelled", cancelled, 0, ENROLLMENT_SAMPLE_TOTAL, "")
            }
        }
    }

    private fun overlapMetadata(samples: FloatArray, embedding: FloatArray?): JSONObject {
        if (embedding == null) return overlapJson(SpeakerOverlapRules.INSUFFICIENT_EVIDENCE)
        val embeddings = mutableListOf<FloatArray>()
        val windowSamples = AudioRecorder.SAMPLE_RATE * 2
        if (samples.size >= AudioRecorder.SAMPLE_RATE * 4) {
            for (start in samples.indices step windowSamples) {
                val end = (start + windowSamples).coerceAtMost(samples.size)
                if (end - start < AudioRecorder.SAMPLE_RATE) continue
                speaker.compute(samples.copyOfRange(start, end))?.let(embeddings::add)
            }
        }
        val similarities = embeddings.zipWithNext(::cosineSimilarity).filterNotNull()
        return overlapJson(SpeakerOverlapRules.classify(samples.size, similarities))
    }

    private fun overlapJson(result: SpeakerOverlapResult): JSONObject = JSONObject()
        .put("state", result.state)
        .put("reason", result.reason)

    private fun cosineSimilarity(left: FloatArray, right: FloatArray): Float? {
        if (left.size != right.size || left.isEmpty()) return null
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        left.indices.forEach { index ->
            dot += left[index] * right[index]
            leftNorm += left[index] * left[index]
            rightNorm += right[index] * right[index]
        }
        if (leftNorm <= 0.0 || rightNorm <= 0.0) return null
        return (dot / kotlin.math.sqrt(leftNorm * rightNorm)).toFloat()
    }

    private fun expireWakeIfNeeded() {
        if (wakePending() && SystemClock.elapsedRealtime() >= wakeDeadlineMillis) clearWake()
    }

    private fun clearWake() {
        wakeDeadlineMillis = 0
        wakeKeyword = ""
        onlineText = ""
        onlineAsr.reset()
        onPartial("")
    }

    private fun wakePending(): Boolean = wakeDeadlineMillis > 0

    private fun samplesToMillis(samples: Long): Int =
        (samples * 1000L / AudioRecorder.SAMPLE_RATE).coerceIn(0, Int.MAX_VALUE.toLong()).toInt()

    companion object {
        private const val MAX_QUEUED_FRAMES = 16
        private const val WAKE_QUERY_TIMEOUT_MILLIS = 10_000L
        private const val STOP_JOIN_MILLIS = 5_000L
        private const val ENROLLMENT_SAVE_TIMEOUT_SECONDS = 10.0
        private const val ENROLLMENT_SAMPLE_TOTAL = 3
    }
}

internal data class SpeakerOverlapResult(val state: String, val reason: String)

internal object SpeakerOverlapRules {
    private const val SAMPLE_RATE = 16_000
    private const val CONFLICT_THRESHOLD = 0.65f
    val INSUFFICIENT_EVIDENCE = SpeakerOverlapResult("unknown", "insufficient_speaker_evidence")

    fun classify(sampleCount: Int, adjacentSimilarities: List<Float>): SpeakerOverlapResult {
        if (sampleCount < SAMPLE_RATE * 2) return INSUFFICIENT_EVIDENCE
        if (sampleCount < SAMPLE_RATE * 4) {
            return SpeakerOverlapResult("not_observed", "single_consistent_speaker_window")
        }
        val minimum = adjacentSimilarities.minOrNull()
            ?: return SpeakerOverlapResult("unknown", "insufficient_speaker_windows")
        return if (minimum < CONFLICT_THRESHOLD) {
            SpeakerOverlapResult("suspected", "speaker_window_conflict")
        } else {
            SpeakerOverlapResult("not_observed", "speaker_windows_consistent")
        }
    }
}
