package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

private const val INPUT_PROBE_SILENCE_DBFS = -120.0

data class AudioInputProbeSnapshot(
    val route: AudioInputRoute? = null,
    val receivedFrames: Boolean = false,
    val peakDbfs: Double = INPUT_PROBE_SILENCE_DBFS,
    val transcript: String = "",
    val asrUnavailable: Boolean = false,
    val error: String = "",
)

data class AudioInputProbeResult(
    val passed: Boolean,
    val status: String,
    val guidance: String,
)

object AudioInputProbeEvaluator {
    const val SPEECH_CONFIRMATION_PEAK_DBFS = -55.0

    fun evaluate(snapshot: AudioInputProbeSnapshot): AudioInputProbeResult = when {
        snapshot.error.isNotBlank() -> AudioInputProbeResult(
            passed = false,
            status = "测试失败",
            guidance = snapshot.error,
        )
        snapshot.route == null -> AudioInputProbeResult(
            passed = false,
            status = "未确认收音设备",
            guidance = "Android 未返回实际输入路由，请重新连接蓝牙收音设备后再试",
        )
        !snapshot.receivedFrames -> AudioInputProbeResult(
            passed = false,
            status = "未收到音频数据",
            guidance = "已路由到 ${snapshot.route.label}，但未读取到音频帧，请重新连接设备后再试",
        )
        snapshot.transcript.isNotBlank() -> AudioInputProbeResult(
            passed = true,
            status = "收音正常",
            guidance = "正在使用 ${snapshot.route.label}，已识别到说话内容",
        )
        snapshot.asrUnavailable -> AudioInputProbeResult(
            passed = true,
            status = "收音正常",
            guidance = "正在使用 ${snapshot.route.label}，ASR 未就绪，未显示文字；请播放本次录音确认声音",
        )
        snapshot.peakDbfs < SPEECH_CONFIRMATION_PEAK_DBFS -> AudioInputProbeResult(
            passed = false,
            status = "未检测到可识别语音",
            guidance = "请对 ${snapshot.route.label} 说话后重试，或播放本次录音确认声音",
        )
        else -> AudioInputProbeResult(
            passed = true,
            status = "收音正常",
            guidance = "正在使用 ${snapshot.route.label}，已检测到说话声音",
        )
    }
}

data class AudioInputProbeUpdate(
    val testing: Boolean,
    val snapshot: AudioInputProbeSnapshot,
    val result: AudioInputProbeResult? = null,
    val recording: AudioInputProbeRecording? = null,
)

/** A single probe's in-memory audio. It must be cleared after playback or lifecycle cancellation. */
class AudioInputProbeRecording internal constructor(samples: ShortArray) {
    private val lock = Any()
    private var samples = samples
    private val cancelled = AtomicBoolean(false)
    @Volatile private var activeTrack: AudioTrack? = null

    fun play(onFinished: (String?) -> Unit): Boolean {
        val playbackSamples = synchronized(lock) { samples.takeIf { it.isNotEmpty() && !cancelled.get() } }
            ?: return false
        Thread({
            var failure: String? = null
            try {
                val minimumBytes = AudioTrack.getMinBufferSize(
                    AudioRecorder.SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                )
                check(minimumBytes > 0) { "设备不支持测试录音回放" }
                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build(),
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(AudioRecorder.SAMPLE_RATE)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build(),
                    )
                    .setBufferSizeInBytes(max(minimumBytes, playbackSamples.size * Short.SIZE_BYTES))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()
                activeTrack = track
                track.play()
                var offset = 0
                while (!cancelled.get() && offset < playbackSamples.size) {
                    val written = track.write(playbackSamples, offset, playbackSamples.size - offset, AudioTrack.WRITE_BLOCKING)
                    check(written > 0) { "播放测试录音失败：$written" }
                    offset += written
                }
                val deadlineMillis = System.currentTimeMillis() + PLAYBACK_DRAIN_TIMEOUT_MILLIS
                while (!cancelled.get() && track.playbackHeadPosition < offset && System.currentTimeMillis() < deadlineMillis) {
                    Thread.sleep(20)
                }
            } catch (error: Throwable) {
                if (!cancelled.get()) failure = error.message?.takeIf(String::isNotBlank) ?: "播放测试录音失败"
            } finally {
                runCatching { activeTrack?.stop() }
                runCatching { activeTrack?.release() }
                activeTrack = null
                clear()
                onFinished(failure)
            }
        }, "android-audio-input-playback").start()
        return true
    }

    fun clear() {
        cancelled.set(true)
        runCatching { activeTrack?.stop() }
        synchronized(lock) {
            samples.fill(0)
            samples = ShortArray(0)
        }
    }

    companion object {
        private const val PLAYBACK_DRAIN_TIMEOUT_MILLIS = 6_000L
    }
}

/** A bounded foreground-only probe. PCM is retained only for the current one-time Settings playback. */
class AudioInputProbe(
    private val context: Context,
    private val onUpdate: (AudioInputProbeUpdate) -> Unit,
) {
    private val cancelled = AtomicBoolean(false)
    @Volatile private var activeRecord: AudioRecord? = null

    fun start() {
        Thread(::runProbe, "android-audio-input-probe").start()
    }

    fun cancel() {
        cancelled.set(true)
        runCatching { activeRecord?.stop() }
    }

    private fun runProbe() {
        var snapshot = AudioInputProbeSnapshot()
        var recordingSamples = ShortArray(0)
        var asr: SherpaOnlineAsrAdapter? = null
        try {
            asr = createAsrOrNull { snapshot = snapshot.copy(asrUnavailable = true) }
            val prepared = AudioRecorder.createVoiceRecognitionAudioRecord(context)
            val recorder = prepared.recorder
            activeRecord = recorder
            try {
                recorder.startRecording()
                check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "麦克风未能开始录音" }
                snapshot = snapshot.copy(route = AndroidAudioInputRouting.verify(context, prepared))
                onUpdate(AudioInputProbeUpdate(testing = true, snapshot = snapshot))
                val buffer = ShortArray(AudioRecorder.FRAME_SAMPLES)
                recordingSamples = ShortArray(MAX_RECORDING_SAMPLES)
                var recordedSamples = 0
                val deadlineMillis = System.currentTimeMillis() + TEST_DURATION_MILLIS
                while (!cancelled.get() && System.currentTimeMillis() < deadlineMillis) {
                    val read = recorder.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                    if (cancelled.get()) break
                    check(read >= 0) { "读取音频失败：$read" }
                    if (read == 0) continue
                    val level = AudioLevelMeter.measure(buffer, read)
                    snapshot = snapshot.copy(
                        receivedFrames = true,
                        peakDbfs = max(snapshot.peakDbfs, level.peakDbfs),
                    )
                    val copied = minOf(read, recordingSamples.size - recordedSamples)
                    if (copied > 0) {
                        buffer.copyInto(recordingSamples, recordedSamples, 0, copied)
                        recordedSamples += copied
                    }
                    asr?.let { adapter ->
                        runCatching {
                            adapter.accept(FloatArray(read) { index -> buffer[index] / 32768f }).first
                        }.onSuccess { partial ->
                            if (partial.isNotBlank()) snapshot = snapshot.copy(transcript = partial)
                        }.onFailure {
                            runCatching { adapter.close() }
                            asr = null
                            snapshot = snapshot.copy(asrUnavailable = true)
                        }
                    }
                    onUpdate(AudioInputProbeUpdate(testing = true, snapshot = snapshot))
                }
                if (!cancelled.get()) {
                    asr?.let { adapter ->
                        runCatching { adapter.finish() }.onSuccess { final ->
                            if (final.isNotBlank()) snapshot = snapshot.copy(transcript = final)
                        }.onFailure {
                            snapshot = snapshot.copy(asrUnavailable = true)
                        }
                    }
                    val completedRecording = recordingSamples.copyOf(recordedSamples)
                    recordingSamples.fill(0)
                    recordingSamples = completedRecording
                }
            } finally {
                activeRecord = null
                runCatching { if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop() }
                recorder.release()
            }
        } catch (error: Throwable) {
            if (!cancelled.get()) snapshot = snapshot.copy(error = inputErrorMessage(error))
        } finally {
            runCatching { asr?.close() }
        }
        if (!cancelled.get()) {
            val recording = recordingSamples.takeIf {
                snapshot.error.isBlank() && snapshot.receivedFrames && it.isNotEmpty()
            }?.let(::AudioInputProbeRecording)
            if (recording == null) recordingSamples.fill(0)
            onUpdate(
                AudioInputProbeUpdate(
                    testing = false,
                    snapshot = snapshot,
                    result = AudioInputProbeEvaluator.evaluate(snapshot),
                    recording = recording,
                ),
            )
        } else {
            recordingSamples.fill(0)
        }
    }

    private fun createAsrOrNull(onUnavailable: () -> Unit): SherpaOnlineAsrAdapter? = runCatching {
        val pack = ModelPackInstaller(context).current() ?: error("未安装完整的本地 ASR 模型")
        SherpaOnlineAsrAdapter(context, pack)
    }.getOrElse {
        onUnavailable()
        null
    }

    private fun inputErrorMessage(error: Throwable): String = when (error) {
        is SecurityException -> "麦克风权限未授予"
        else -> error.message?.takeIf(String::isNotBlank) ?: "收音测试无法启动"
    }

    companion object {
        private const val TEST_DURATION_MILLIS = 5_000L
        private const val MAX_RECORDING_SAMPLES = AudioRecorder.SAMPLE_RATE * 5
    }
}
