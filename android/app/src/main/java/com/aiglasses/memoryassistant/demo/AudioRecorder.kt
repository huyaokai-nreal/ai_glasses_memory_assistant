package com.aiglasses.memoryassistant.demo

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sqrt

data class AudioLevel(val rmsDbfs: Double, val peakDbfs: Double)

object AudioLevelMeter {
    private const val SILENCE_DBFS = -120.0

    fun measure(pcm16: ShortArray, sampleCount: Int): AudioLevel {
        val count = sampleCount.coerceIn(0, pcm16.size)
        if (count == 0) return AudioLevel(SILENCE_DBFS, SILENCE_DBFS)
        var squareSum = 0.0
        var peak = 0
        repeat(count) { index ->
            val sample = pcm16[index].toInt()
            squareSum += sample.toDouble() * sample
            peak = max(peak, abs(sample))
        }
        val rms = sqrt(squareSum / count) / 32768.0
        val normalizedPeak = peak / 32768.0
        return AudioLevel(toDbfs(rms), toDbfs(normalizedPeak))
    }

    private fun toDbfs(value: Double): Double =
        if (value <= 0.0) SILENCE_DBFS else (20.0 * log10(value)).coerceAtLeast(SILENCE_DBFS)
}

/** The PCM buffer is reused after this call and must not be retained by the sink. */
fun interface PcmFrameSink {
    fun accept(pcm16: ShortArray, sampleCount: Int, capturedAtNanos: Long)
}

class AudioRecorder(
    private val context: Context,
    private val sink: PcmFrameSink,
    private val onInputRoute: (AudioInputRoute) -> Unit,
    private val onFailure: (Throwable) -> Unit,
) {
    private val running = AtomicBoolean(false)
    private val paused = AtomicBoolean(false)
    private val restartRequested = AtomicBoolean(false)
    private val gate = Object()
    @Volatile private var activeRecord: AudioRecord? = null
    @Volatile private var worker: Thread? = null

    fun start() {
        if (!running.compareAndSet(false, true)) return
        paused.set(false)
        worker = Thread(::captureLoop, "android-pcm-capture").apply {
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    fun pause() {
        if (!running.get()) return
        paused.set(true)
        stopActiveRecord()
    }

    fun resume() {
        if (!running.get()) return
        paused.set(false)
        synchronized(gate) { gate.notifyAll() }
    }

    fun restartInput() {
        if (!running.get() || paused.get()) return
        restartRequested.set(true)
        stopActiveRecord()
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        paused.set(false)
        stopActiveRecord()
        synchronized(gate) { gate.notifyAll() }
        worker?.join(STOP_JOIN_MILLIS)
        worker = null
    }

    private fun captureLoop() {
        try {
            while (running.get()) {
                waitUntilResumed()
                if (!running.get()) break
                restartRequested.set(false)
                val prepared = createAudioRecord()
                val recorder = prepared.recorder
                activeRecord = recorder
                try {
                    recorder.startRecording()
                    onInputRoute(AndroidAudioInputRouting.verify(context, prepared))
                    val buffer = ShortArray(FRAME_SAMPLES)
                    while (running.get() && !paused.get() && !restartRequested.get()) {
                        val read = recorder.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                        when {
                            read > 0 -> sink.accept(buffer, read, System.nanoTime())
                            read == 0 -> continue
                            !running.get() || paused.get() || restartRequested.get() -> break
                            else -> error("AudioRecord.read failed: $read")
                        }
                    }
                } finally {
                    activeRecord = null
                    runCatching { if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop() }
                    recorder.release()
                }
            }
        } catch (error: Throwable) {
            if (running.getAndSet(false)) onFailure(error)
        } finally {
            activeRecord = null
        }
    }

    private fun waitUntilResumed() {
        synchronized(gate) {
            while (running.get() && paused.get()) gate.wait()
        }
    }

    private fun createAudioRecord(): PreparedAudioRecord = createVoiceRecognitionAudioRecord(context)

    private fun stopActiveRecord() {
        runCatching { activeRecord?.stop() }
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        internal const val FRAME_SAMPLES = 4_000
        private const val BUFFER_FRAME_COUNT = 4
        private const val STOP_JOIN_MILLIS = 2_000L

        /** Keeps one recording contract and one input-routing policy for every native capture path. */
        internal fun createVoiceRecognitionAudioRecord(context: Context): PreparedAudioRecord {
            check(context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "麦克风权限已被撤销"
            }
            val minimumBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            check(minimumBytes > 0) { "设备不支持 16 kHz 单声道 PCM16 录音" }
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                max(minimumBytes, FRAME_SAMPLES * Short.SIZE_BYTES * BUFFER_FRAME_COUNT),
            )
            check(recorder.state == AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                "麦克风初始化失败，可能正被其他应用占用"
            }
            return try {
                AndroidAudioInputRouting.prepare(context, recorder)
            } catch (error: Throwable) {
                recorder.release()
                throw error
            }
        }
    }
}
