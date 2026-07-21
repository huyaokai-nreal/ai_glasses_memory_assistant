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
    private val worker = Thread(::runLoop, "sherpa-native-audio").apply { start() }

    override fun accept(pcm16: ShortArray, sampleCount: Int, capturedAtNanos: Long) {
        if (!running.get()) return
        val frame = Frame(pcm16.copyOf(sampleCount), capturedAtNanos)
        if (!queue.offer(frame) && running.getAndSet(false)) {
            onFailure(IllegalStateException("本地模型处理速度低于实时收音，已安全停止"))
            worker.interrupt()
        }
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
        }
    }

    private fun process(frame: Frame) {
        expireWakeIfNeeded()
        val floats = FloatArray(frame.samples.size) { index -> frame.samples[index] / 32768f }
        totalSamples += floats.size
        val completedSegments = vad.accept(floats)
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
            onlineText = onlineAsr.accept(floats).first.ifBlank { onlineText }
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
            .put("overlap", JSONObject().put("state", "unknown").put("reason", "android_overlap_not_available"))
            .put("audio_retention", "discarded_after_processing")
        val privateEvent = JSONObject()
        if (embedding != null) {
            val values = org.json.JSONArray()
            embedding.forEach(values::put)
            privateEvent.put("speaker_embedding", values)
            privateEvent.put("speaker_embedding_model", speakerModelName)
        }
        val queueResult = PythonRuntime.ingestAudioEvent(ownerId, captureId, event, privateEvent)
        if (lane == "assistant" && queueResult.optString("status") in setOf("pending", "running")) {
            onReplyQueued(event.getString("event_id"))
        }
        if (lane == "assistant") clearWake()
    }

    private fun expireWakeIfNeeded() {
        if (wakePending() && SystemClock.elapsedRealtime() >= wakeDeadlineMillis) clearWake()
    }

    private fun clearWake() {
        wakeDeadlineMillis = 0
        wakeKeyword = ""
        onlineText = ""
        onlineAsr.reset()
    }

    private fun wakePending(): Boolean = wakeDeadlineMillis > 0

    private fun samplesToMillis(samples: Long): Int =
        (samples * 1000L / AudioRecorder.SAMPLE_RATE).coerceIn(0, Int.MAX_VALUE.toLong()).toInt()

    companion object {
        private const val MAX_QUEUED_FRAMES = 16
        private const val WAKE_QUERY_TIMEOUT_MILLIS = 10_000L
        private const val STOP_JOIN_MILLIS = 5_000L
    }
}
