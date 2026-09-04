package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

data class OfflineDecodedAudio(
    val samples: FloatArray,
    val sourceFormat: String,
    val sourceSampleRate: Int,
    val sourceChannelCount: Int,
    val sourceDurationMillis: Long,
)

/** Reads a PCM16 WAV into the same 16 kHz mono float contract used by native capture. */
object OfflineWavPcm16Reader {
    const val TARGET_SAMPLE_RATE = AudioRecorder.SAMPLE_RATE
    const val MAX_DURATION_MILLIS = 15 * 60 * 1_000L
    private const val MAX_FORMAT_CHUNK_BYTES = 4_096L

    fun read(source: InputStream): OfflineDecodedAudio {
        val input = BufferedInputStream(source)
        require(readFourCc(input) == "RIFF") { "不是 RIFF WAV 文件" }
        readUnsignedInt(input)
        require(readFourCc(input) == "WAVE") { "不是 WAVE 音频文件" }

        var format: WavFormat? = null
        while (true) {
            val chunkId = readFourCc(input)
            val chunkSize = readUnsignedInt(input)
            when (chunkId) {
                "fmt " -> {
                    require(format == null) { "WAV 包含重复 fmt 块" }
                    format = readFormat(input, chunkSize)
                }
                "data" -> {
                    val parsedFormat = requireNotNull(format) { "WAV 的 data 块必须位于 fmt 块之后" }
                    return readData(input, chunkSize, parsedFormat)
                }
                else -> skipFully(input, paddedChunkSize(chunkSize))
            }
        }
    }

    private fun readFormat(input: InputStream, chunkSize: Long): WavFormat {
        require(chunkSize in 16..MAX_FORMAT_CHUNK_BYTES) { "WAV fmt 块无效" }
        val audioFormat = readUnsignedShort(input)
        val channels = readUnsignedShort(input)
        val sampleRate = readUnsignedInt(input)
        readUnsignedInt(input)
        val blockAlign = readUnsignedShort(input)
        val bitsPerSample = readUnsignedShort(input)
        skipFully(input, chunkSize - 16)
        if (chunkSize % 2L != 0L) skipFully(input, 1)
        require(audioFormat == PCM_FORMAT) { "仅支持 PCM16 WAV，请先转码" }
        require(channels in 1..2) { "仅支持单声道或立体声 WAV" }
        require(sampleRate in 1..MAX_SOURCE_SAMPLE_RATE.toLong()) { "WAV 采样率无效或过高" }
        require(bitsPerSample == PCM16_BITS) { "仅支持 PCM16 WAV，请先转码" }
        require(blockAlign == channels * PCM16_BYTES) { "WAV PCM16 块大小无效" }
        return WavFormat(sampleRate.toInt(), channels, blockAlign)
    }

    private fun readData(input: InputStream, dataSize: Long, format: WavFormat): OfflineDecodedAudio {
        require(dataSize % format.blockAlign == 0L) { "WAV data 块长度无效" }
        val sourceFrames = dataSize / format.blockAlign
        require(sourceFrames > 0) { "WAV 不包含音频样本" }
        require(sourceFrames <= maxSourceFrames(format.sampleRate)) { "WAV 超过 15 分钟限制" }
        val reader = Pcm16FrameReader(input, dataSize)
        val durationMicros = sourceFrames * 1_000_000L / format.sampleRate
        val durationMillis = durationMicros / 1_000L
        val normalized = OfflinePcm16Normalizer(format.sampleRate, format.channelCount, durationMicros)
        repeat(sourceFrames.toInt()) { normalized.accept(reader.readMono(format.channelCount)) }
        reader.drain()
        reader.requireFullyConsumed()
        return OfflineDecodedAudio(
            samples = normalized.finish(),
            sourceFormat = "WAV",
            sourceSampleRate = format.sampleRate,
            sourceChannelCount = format.channelCount,
            sourceDurationMillis = durationMillis,
        )
    }

    private fun maxSourceFrames(sampleRate: Int): Long = MAX_DURATION_MILLIS * sampleRate / 1_000L

    private fun paddedChunkSize(size: Long): Long = size + size % 2L

    private fun readFourCc(input: InputStream): String {
        val bytes = ByteArray(4)
        readFully(input, bytes)
        return bytes.toString(Charsets.US_ASCII)
    }

    private fun readUnsignedShort(input: InputStream): Int {
        val low = readByte(input)
        return low or (readByte(input) shl 8)
    }

    private fun readUnsignedInt(input: InputStream): Long =
        readUnsignedShort(input).toLong() or (readUnsignedShort(input).toLong() shl 16)

    private fun readByte(input: InputStream): Int = input.read().also { require(it >= 0) { "WAV 文件已截断" } }

    private fun readFully(input: InputStream, bytes: ByteArray) {
        var offset = 0
        while (offset < bytes.size) {
            val count = input.read(bytes, offset, bytes.size - offset)
            require(count > 0) { "WAV 文件已截断" }
            offset += count
        }
    }

    private fun skipFully(input: InputStream, bytes: Long) {
        var remaining = bytes
        while (remaining > 0) {
            val skipped = input.skip(remaining)
            if (skipped > 0) {
                remaining -= skipped
            } else {
                readByte(input)
                remaining -= 1
            }
        }
    }

    private data class WavFormat(val sampleRate: Int, val channelCount: Int, val blockAlign: Int)

    private class Pcm16FrameReader(input: InputStream, byteCount: Long) {
        private val input = input
        private var remaining = byteCount
        private val buffer = ByteArray(8_192)
        private var offset = 0
        private var count = 0

        fun readMono(channelCount: Int): Float {
            var sum = 0
            repeat(channelCount) { sum += readSignedShort() }
            return (sum / channelCount.toFloat() / 32_768f).coerceIn(-1f, 1f)
        }

        fun requireFullyConsumed() {
            require(remaining == 0L) { "WAV data 块已截断" }
        }

        fun drain() {
            while (remaining > 0) readDataByte()
        }

        private fun readSignedShort(): Int {
            val low = readDataByte()
            val high = readDataByte()
            return (low or (high shl 8)).toShort().toInt()
        }

        private fun readDataByte(): Int {
            if (offset >= count) refill()
            remaining -= 1
            return buffer[offset++].toInt() and 0xff
        }

        private fun refill() {
            require(remaining > 0) { "WAV data 块已截断" }
            count = input.read(buffer, 0, min(buffer.size.toLong(), remaining).toInt())
            require(count > 0) { "WAV data 块已截断" }
            offset = 0
        }
    }

    private const val PCM_FORMAT = 1
    private const val PCM16_BITS = 16
    private const val PCM16_BYTES = 2
    private const val MAX_SOURCE_SAMPLE_RATE = 192_000
}

/** Dispatches selected local audio without retaining its compressed source beyond decoding. */
object OfflineAudioDecoder {
    fun read(context: Context, uri: Uri, isCancelled: () -> Boolean): OfflineDecodedAudio =
        if (isRiffWave(context, uri)) {
            context.contentResolver.openInputStream(uri)?.use(OfflineWavPcm16Reader::read)
                ?: error("无法读取所选音频文件")
        } else {
            OfflineM4aAacReader.read(context, uri, isCancelled)
        }

    private fun isRiffWave(context: Context, uri: Uri): Boolean {
        val header = ByteArray(12)
        val count = context.contentResolver.openInputStream(uri)?.use { input ->
            BufferedInputStream(input).use { buffered ->
                var offset = 0
                while (offset < header.size) {
                    val read = buffered.read(header, offset, header.size - offset)
                    if (read <= 0) break
                    offset += read
                }
                offset
            }
        } ?: error("无法读取所选音频文件")
        return count == header.size &&
            header.copyOfRange(0, 4).toString(Charsets.US_ASCII) == "RIFF" &&
            header.copyOfRange(8, 12).toString(Charsets.US_ASCII) == "WAVE"
    }
}

/** Uses Android's platform AAC decoder for the M4A files produced by Lark. */
private object OfflineM4aAacReader {
    fun read(context: Context, uri: Uri, isCancelled: () -> Boolean): OfflineDecodedAudio {
        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: error("M4A 中没有可用的音频轨")
            val trackFormat = extractor.getTrackFormat(trackIndex)
            val mime = trackFormat.getString(MediaFormat.KEY_MIME) ?: error("M4A 音频轨缺少 MIME 类型")
            require(mime == MIME_AAC) { "当前仅支持 AAC M4A，文件编码为 $mime" }
            require(trackFormat.containsKey(MediaFormat.KEY_DURATION)) { "M4A 缺少时长信息，无法安全解码" }
            val durationMicros = trackFormat.getLong(MediaFormat.KEY_DURATION)
            require(durationMicros in 1..MAX_DURATION_MICROS) { "M4A 超过 15 分钟限制" }
            val declaredRate = trackFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val declaredChannels = trackFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            require(declaredChannels in 1..2) { "仅支持单声道或立体声 M4A" }
            require(declaredRate in 1..MAX_SOURCE_SAMPLE_RATE) { "M4A 采样率无效或过高" }

            extractor.selectTrack(trackIndex)
            decoder = MediaCodec.createDecoderByType(mime).apply {
                configure(trackFormat, null, null, 0)
                start()
            }
            val decoded = decodeToNativePcm(extractor, decoder, durationMicros, isCancelled)
            return OfflineDecodedAudio(
                samples = decoded.samples,
                sourceFormat = "M4A/AAC",
                sourceSampleRate = decoded.sourceSampleRate,
                sourceChannelCount = decoded.sourceChannelCount,
                sourceDurationMillis = durationMicros / 1_000L,
            )
        } finally {
            runCatching { decoder?.stop() }
            runCatching { decoder?.release() }
            extractor.release()
        }
    }

    private fun decodeToNativePcm(
        extractor: MediaExtractor,
        decoder: MediaCodec,
        durationMicros: Long,
        isCancelled: () -> Boolean,
    ): DecodedPcm {
        val bufferInfo = MediaCodec.BufferInfo()
        var normalizer: OfflinePcm16Normalizer? = null
        var outputRate = 0
        var outputChannels = 0
        var inputFinished = false
        var outputFinished = false
        while (!outputFinished) {
            ensureActive(isCancelled)
            if (!inputFinished) {
                val inputIndex = decoder.dequeueInputBuffer(CODEC_TIMEOUT_MICROS)
                if (inputIndex >= 0) {
                    val inputBuffer = requireNotNull(decoder.getInputBuffer(inputIndex)) { "AAC 解码器未提供输入缓冲区" }
                    inputBuffer.clear()
                    val size = extractor.readSampleData(inputBuffer, 0)
                    if (size < 0) {
                        decoder.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputFinished = true
                    } else {
                        decoder.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }
            while (true) {
                val outputIndex = decoder.dequeueOutputBuffer(bufferInfo, CODEC_TIMEOUT_MICROS)
                when {
                    outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val outputFormat = decoder.outputFormat
                        outputRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        outputChannels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        val encoding = if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        } else {
                            AudioFormat.ENCODING_PCM_16BIT
                        }
                        require(encoding == AudioFormat.ENCODING_PCM_16BIT) { "AAC 解码器未输出 PCM16" }
                        require(outputChannels in 1..2) { "AAC 解码后声道数不受支持：$outputChannels" }
                        require(outputRate in 1..MAX_SOURCE_SAMPLE_RATE) { "AAC 解码后采样率无效或过高" }
                        normalizer = OfflinePcm16Normalizer(outputRate, outputChannels, durationMicros)
                    }
                    outputIndex >= 0 -> {
                        val target = requireNotNull(normalizer) { "AAC 解码器未报告 PCM 输出格式" }
                        if (bufferInfo.size > 0) {
                            val buffer = requireNotNull(decoder.getOutputBuffer(outputIndex)) { "AAC 解码器未提供输出缓冲区" }
                                .duplicate()
                                .order(ByteOrder.LITTLE_ENDIAN)
                            buffer.position(bufferInfo.offset)
                            buffer.limit(bufferInfo.offset + bufferInfo.size)
                            consumePcm16(buffer, outputChannels, target, isCancelled)
                        }
                        decoder.releaseOutputBuffer(outputIndex, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputFinished = true
                    }
                }
            }
        }
        val completed = requireNotNull(normalizer) { "AAC 解码器没有输出 PCM 数据" }
        return DecodedPcm(completed.finish(), outputRate, outputChannels)
    }

    private fun consumePcm16(
        buffer: ByteBuffer,
        channels: Int,
        normalizer: OfflinePcm16Normalizer,
        isCancelled: () -> Boolean,
    ) {
        val bytesPerFrame = channels * Short.SIZE_BYTES
        require(buffer.remaining() % bytesPerFrame == 0) { "AAC 解码器输出了不完整的 PCM16 帧" }
        while (buffer.remaining() >= bytesPerFrame) {
            ensureActive(isCancelled)
            var sum = 0
            repeat(channels) { sum += buffer.short.toInt() }
            normalizer.accept(sum / channels.toFloat() / 32_768f)
        }
    }

    private fun ensureActive(isCancelled: () -> Boolean) {
        if (isCancelled()) throw CancellationException("离线音频测试已取消")
    }

    private data class DecodedPcm(val samples: FloatArray, val sourceSampleRate: Int, val sourceChannelCount: Int)

    private const val MIME_AAC = "audio/mp4a-latm"
    private const val MAX_SOURCE_SAMPLE_RATE = 192_000
    private const val MAX_DURATION_MICROS = OfflineWavPcm16Reader.MAX_DURATION_MILLIS * 1_000L
    private const val CODEC_TIMEOUT_MICROS = 10_000L
}

/** Incrementally downmixes and resamples decoded PCM without keeping source PCM in memory. */
internal class OfflinePcm16Normalizer(
    private val sourceSampleRate: Int,
    sourceChannelCount: Int,
    sourceDurationMicros: Long,
) {
    private val output = FloatArray(targetCapacity(sourceDurationMicros))
    private var sourceFrameCount = 0L
    private var targetFrameCount = 0
    private var previous = 0f
    private var hasPrevious = false

    init {
        require(sourceChannelCount in 1..2) { "仅支持单声道或立体声音频" }
        require(sourceSampleRate in 1..MAX_SOURCE_SAMPLE_RATE) { "音频采样率无效或过高" }
        require(sourceDurationMicros in 1..MAX_DURATION_MICROS) { "音频超过 15 分钟限制" }
    }

    fun accept(monoSample: Float) {
        require(sourceFrameCount < maxSourceFrames()) { "音频超过 15 分钟限制" }
        val current = monoSample.coerceIn(-1f, 1f)
        if (!hasPrevious) {
            previous = current
            hasPrevious = true
            sourceFrameCount = 1
            return
        }
        emitUntil(sourceFrameCount, previous, current)
        previous = current
        sourceFrameCount += 1
    }

    fun finish(): FloatArray {
        require(hasPrevious) { "音频不包含可解码的 PCM16 样本" }
        val requiredTargetFrames = ceil(sourceFrameCount.toDouble() * AudioRecorder.SAMPLE_RATE / sourceSampleRate).toLong()
        require(requiredTargetFrames <= output.size) { "解码后的音频超过容器申明的 15 分钟限制" }
        emitUntil(sourceFrameCount, previous, previous)
        return if (targetFrameCount == output.size) output else output.copyOf(targetFrameCount)
    }

    private fun emitUntil(nextSourceFrame: Long, left: Float, right: Float) {
        while (targetFrameCount < output.size) {
            val sourcePosition = targetFrameCount.toDouble() * sourceSampleRate / AudioRecorder.SAMPLE_RATE
            if (sourcePosition >= nextSourceFrame) return
            val base = floor(sourcePosition).toLong()
            require(base == nextSourceFrame - 1) { "音频重采样帧顺序异常" }
            val fraction = (sourcePosition - base).toFloat().coerceIn(0f, 1f)
            output[targetFrameCount] = left + (right - left) * fraction
            targetFrameCount += 1
        }
    }

    private fun targetCapacity(sourceDurationMicros: Long): Int =
        ceil(sourceDurationMicros.toDouble() * AudioRecorder.SAMPLE_RATE / 1_000_000.0).toLong()
            .coerceIn(1, MAX_TARGET_SAMPLES)
            .toInt()

    private fun maxSourceFrames(): Long = OfflineWavPcm16Reader.MAX_DURATION_MILLIS * sourceSampleRate / 1_000L

    private companion object {
        const val MAX_SOURCE_SAMPLE_RATE = 192_000
        const val MAX_DURATION_MICROS = OfflineWavPcm16Reader.MAX_DURATION_MILLIS * 1_000L
        const val MAX_TARGET_SAMPLES = OfflineWavPcm16Reader.MAX_DURATION_MILLIS * AudioRecorder.SAMPLE_RATE / 1_000L
    }
}

data class OfflineAudioGainResult(
    val samples: FloatArray,
    val enabled: Boolean,
    val decibels: Int,
    val inputPeak: Float,
    val outputPeak: Float,
    val clippedSampleCount: Int,
)

object OfflineAudioGain {
    const val MIN_DECIBELS = -24
    const val MAX_DECIBELS = 24

    fun apply(samples: FloatArray, enabled: Boolean, decibels: Int): OfflineAudioGainResult {
        require(decibels in MIN_DECIBELS..MAX_DECIBELS) { "增益必须在 $MIN_DECIBELS 到 $MAX_DECIBELS dB 之间" }
        val inputPeak = peak(samples)
        if (!enabled) {
            return OfflineAudioGainResult(samples, false, 0, inputPeak, inputPeak, 0)
        }
        val multiplier = 10.0.pow(decibels / 20.0).toFloat()
        val adjusted = FloatArray(samples.size)
        var outputPeak = 0f
        var clipped = 0
        samples.indices.forEach { index ->
            val scaled = samples[index] * multiplier
            val value = scaled.coerceIn(-1f, 1f)
            if (value != scaled) clipped += 1
            adjusted[index] = value
            outputPeak = max(outputPeak, abs(value))
        }
        return OfflineAudioGainResult(adjusted, true, decibels, inputPeak, outputPeak, clipped)
    }

    private fun peak(samples: FloatArray): Float = samples.fold(0f) { value, sample -> max(value, abs(sample)) }
}

data class OfflineAsrSegment(
    val startMillis: Int,
    val endMillis: Int,
    val text: String,
    val language: String,
)

data class OfflineAudioTestResult(
    val modelVersion: String,
    val vadBackend: String,
    val ambientAsrBackend: String,
    val sourceFormat: String,
    val sourceSampleRate: Int,
    val sourceChannelCount: Int,
    val sourceDurationMillis: Long,
    val normalizedSampleCount: Int,
    val gain: OfflineAudioGainResult,
    val elapsedMillis: Long,
    val segments: List<OfflineAsrSegment>,
    val vadRuntimeProfile: VadRuntimeProfile? = null,
) {
    val transcript: String = segments.map(OfflineAsrSegment::text).filter(String::isNotBlank).joinToString("\n")

    /** Debug-only JSON contains final text/timing, never decoded samples or source bytes. */
    fun toEvalAliDebugJson(caseId: String): String {
        require(caseId.matches(Regex("[A-Za-z0-9._-]{1,128}"))) { "Eval_Ali case id is invalid" }
        val modelProfile = JSONObject()
            .put("model_pack_version", modelVersion)
            .put("vad_backend", vadBackend)
            .put("ambient_asr_backend", ambientAsrBackend)
        vadRuntimeProfile?.let { modelProfile.put("vad_parameters", it.parametersJson()) }
        return JSONObject()
            .put("schema", "android_eval_ali_events.v1")
            .put("case_id", caseId)
            .put("model_profile", modelProfile)
            .put("elapsed_ms", elapsedMillis)
            .put("events", JSONArray().apply {
                segments.forEachIndexed { index, segment ->
                    put(JSONObject()
                        .put("event_id", "android-eval-$index")
                        .put("type", if (segment.text.isBlank()) "speech_rejected" else "transcript_final")
                        .put("lane", "ambient")
                        .put("source_type", "ambient_audio")
                        .put("start_ms", segment.startMillis)
                        .put("end_ms", segment.endMillis)
                        .put("text", segment.text)
                        .put("final", true)
                        .put("speaker", JSONObject().put("voice_group", "android_ambient"))
                        .put("overlap", JSONObject().put("state", "not_observed"))
                        .put("audio_retention", "discarded_after_processing")
                    )
                }
            })
            .toString()
    }
}

internal data class OfflinePcmSegment(val startSample: Long, val samples: FloatArray)

internal interface OfflineVadProcessor {
    fun accept(samples: FloatArray): List<OfflinePcmSegment>
    fun flush(): List<OfflinePcmSegment>
}

internal interface OfflineAmbientAsrProcessor {
    fun recognize(samples: FloatArray): AmbientRecognition
}

/** Runs the VAD/ASR ordering independently from Android model construction so safety paths remain unit-testable. */
internal class OfflineVadAsrPipeline(
    private val vad: OfflineVadProcessor,
    private val asr: OfflineAmbientAsrProcessor,
    private val isCancelled: () -> Boolean,
) {
    fun run(samples: FloatArray): List<OfflineAsrSegment> {
        val output = mutableListOf<OfflineAsrSegment>()
        var offset = 0
        while (offset < samples.size) {
            ensureActive()
            val end = min(offset + AudioRecorder.FRAME_SAMPLES, samples.size)
            val frame = samples.copyOfRange(offset, end)
            try {
                vad.accept(frame).forEach { segment -> transcribeSegment(segment, output) }
            } finally {
                frame.fill(0f)
            }
            offset = end
        }
        ensureActive()
        vad.flush().forEach { segment -> transcribeSegment(segment, output) }
        ensureActive()
        return output
    }

    private fun transcribeSegment(segment: OfflinePcmSegment, output: MutableList<OfflineAsrSegment>) {
        try {
            ensureActive()
            val recognition = asr.recognize(segment.samples)
            output += OfflineAsrSegment(
                startMillis = samplesToMillis(segment.startSample),
                endMillis = samplesToMillis(segment.startSample + segment.samples.size),
                text = recognition.text,
                language = recognition.language,
            )
        } finally {
            segment.samples.fill(0f)
        }
    }

    private fun ensureActive() {
        if (isCancelled()) throw CancellationException("离线音频测试已取消")
    }

    private fun samplesToMillis(samples: Long): Int =
        (samples * 1_000L / AudioRecorder.SAMPLE_RATE).coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
}

/** Settings-only offline model test. It never creates events or calls the shared Python runtime. */
class OfflineAudioTestRunner(private val context: Context) {
    private val cancelled = AtomicBoolean(false)

    fun cancel() {
        cancelled.set(true)
    }

    fun run(uri: Uri, gainEnabled: Boolean, gainDecibels: Int): OfflineAudioTestResult {
        val startedAt = SystemClock.elapsedRealtime()
        val pack = ModelPackInstaller(context).current() ?: error("未安装完整且校验通过的本地模型包")
        val audio = OfflineAudioDecoder.read(context, uri, cancelled::get)
        val gain = OfflineAudioGain.apply(audio.samples, gainEnabled, gainDecibels)
        val vadProfile = VadRuntimeProfile.from(pack)
        if (gain.samples !== audio.samples) audio.samples.fill(0f)
        try {
            ensureActive()
            SherpaVadAdapter(context, pack, vadProfile).use { vad ->
                SherpaAmbientAsrAdapter(context, pack).use { asr ->
                    val segments = OfflineVadAsrPipeline(
                        vad = object : OfflineVadProcessor {
                            override fun accept(samples: FloatArray): List<OfflinePcmSegment> =
                                vad.accept(samples).map { OfflinePcmSegment(it.start.toLong(), it.samples) }

                            override fun flush(): List<OfflinePcmSegment> =
                                vad.flush().map { OfflinePcmSegment(it.start.toLong(), it.samples) }
                        },
                        asr = object : OfflineAmbientAsrProcessor {
                            override fun recognize(samples: FloatArray): AmbientRecognition = asr.recognize(samples)
                        },
                        isCancelled = cancelled::get,
                    ).run(gain.samples)
                    ensureActive()
                    return OfflineAudioTestResult(
                        modelVersion = pack.version,
                        vadBackend = pack.manifest.components.getValue("vad").engine,
                        ambientAsrBackend = pack.manifest.components.getValue("ambient_asr").engine,
                        sourceFormat = audio.sourceFormat,
                        sourceSampleRate = audio.sourceSampleRate,
                        sourceChannelCount = audio.sourceChannelCount,
                        sourceDurationMillis = audio.sourceDurationMillis,
                        normalizedSampleCount = gain.samples.size,
                        gain = gain.copy(samples = FloatArray(0)),
                        elapsedMillis = SystemClock.elapsedRealtime() - startedAt,
                        segments = segments,
                        vadRuntimeProfile = vadProfile,
                    )
                }
            }
        } finally {
            gain.samples.fill(0f)
            if (gain.samples !== audio.samples) audio.samples.fill(0f)
        }
    }

    private fun ensureActive() {
        if (cancelled.get()) throw CancellationException("离线音频测试已取消")
    }
}
