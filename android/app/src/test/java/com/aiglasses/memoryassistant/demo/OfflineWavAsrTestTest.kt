package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.concurrent.CancellationException

class OfflineWavAsrTestTest {
    @Test
    fun evalAliDebugExportContainsFinalEventsButNoAudioPayload() {
        val result = OfflineAudioTestResult(
            modelVersion = "candidate-pack",
            vadBackend = "ten_vad",
            ambientAsrBackend = "sense_voice",
            sourceFormat = "WAV",
            sourceSampleRate = 16_000,
            sourceChannelCount = 1,
            sourceDurationMillis = 75_000,
            normalizedSampleCount = 1_200_000,
            gain = OfflineAudioGainResult(FloatArray(0), false, 0, 0f, 0f, 0),
            elapsedMillis = 123,
            segments = listOf(OfflineAsrSegment(0, 1_000, "测试", "zh")),
            vadRuntimeProfile = VadRuntimeProfile.from(
                "ten_vad",
                "candidate-pack",
                org.json.JSONObject().put("threshold", 0.35).put("min_speech_seconds", 0.1),
            ),
        )

        val export = result.toEvalAliDebugJson("R0001-smoke")

        assertTrue(export.contains("android_eval_ali_events.v1"))
        assertTrue(export.contains("transcript_final"))
        assertTrue(export.contains("\"threshold\":0.35"))
        assertTrue(export.contains("\"min_speech_seconds\":0.1"))
        assertFalse(export.contains("samples"))
        assertFalse(export.contains("pcm"))
    }

    @Test
    fun pcm16MonoIsResampledToNativeContract() {
        val audio = OfflineWavPcm16Reader.read(
            ByteArrayInputStream(wav(sampleRate = 8_000, channels = 1, samples = shortArrayOf(0, 16_384, -16_384))),
        )

        assertEquals(8_000, audio.sourceSampleRate)
        assertEquals("WAV", audio.sourceFormat)
        assertEquals(1, audio.sourceChannelCount)
        assertEquals(6, audio.samples.size)
        assertEquals(0f, audio.samples[0], 0.0001f)
        assertEquals(0.25f, audio.samples[1], 0.0001f)
        assertEquals(0.5f, audio.samples[2], 0.0001f)
        assertEquals(-0.5f, audio.samples[4], 0.0001f)
    }

    @Test
    fun stereoIsDownmixedBeforeResampling() {
        val audio = OfflineWavPcm16Reader.read(
            ByteArrayInputStream(wav(sampleRate = 16_000, channels = 2, samples = shortArrayOf(16_384, -16_384, 32_767, 32_767))),
        )

        assertEquals(2, audio.samples.size)
        assertEquals(0f, audio.samples[0], 0.0001f)
        assertEquals(32_767f / 32_768f, audio.samples[1], 0.0001f)
    }

    @Test
    fun decodedPcmUsesTheSameStreamingNormalizerAsWav() {
        val normalizer = OfflinePcm16Normalizer(sourceSampleRate = 8_000, sourceChannelCount = 1, sourceDurationMicros = 375)
        listOf(0f, 0.5f, -0.5f).forEach(normalizer::accept)

        val samples = normalizer.finish()
        assertEquals(6, samples.size)
        assertEquals(0.25f, samples[1], 0.0001f)
        assertEquals(-0.5f, samples[4], 0.0001f)
    }

    @Test
    fun decoderNormalizerRejectsAudioLongerThanFifteenMinutes() {
        assertFails("超过 15 分钟") {
            OfflinePcm16Normalizer(
                sourceSampleRate = 16_000,
                sourceChannelCount = 1,
                sourceDurationMicros = OfflineWavPcm16Reader.MAX_DURATION_MILLIS * 1_000L + 1,
            )
        }
    }

    @Test
    fun invalidOrTruncatedWavIsRejected() {
        assertFails("不是 RIFF") {
            OfflineWavPcm16Reader.read(ByteArrayInputStream(ByteArray(12)))
        }
        assertFails("已截断") {
            OfflineWavPcm16Reader.read(ByteArrayInputStream(wav(sampleRate = 16_000, channels = 1, samples = shortArrayOf(1)).dropLast(1).toByteArray()))
        }
    }

    @Test
    fun nonPcm16AndOverlongWavAreRejected() {
        assertFails("仅支持 PCM16") {
            OfflineWavPcm16Reader.read(ByteArrayInputStream(wav(sampleRate = 16_000, channels = 1, samples = shortArrayOf(1), audioFormat = 3)))
        }
        assertFails("超过 15 分钟") {
            OfflineWavPcm16Reader.read(
                ByteArrayInputStream(wav(sampleRate = 1, channels = 1, samples = ShortArray(901))),
            )
        }
    }

    @Test
    fun disabledGainKeepsOriginalSamplesAndEnabledGainClipsSafely() {
        val samples = floatArrayOf(-0.5f, 0.25f, 0.8f)

        val original = OfflineAudioGain.apply(samples, enabled = false, decibels = 12)
        assertSame(samples, original.samples)
        assertEquals(0, original.decibels)
        assertEquals(0, original.clippedSampleCount)

        val adjusted = OfflineAudioGain.apply(samples, enabled = true, decibels = 6)
        assertEquals(-0.9976f, adjusted.samples[0], 0.002f)
        assertEquals(1f, adjusted.samples[2], 0f)
        assertEquals(1, adjusted.clippedSampleCount)
        assertEquals(1f, adjusted.outputPeak, 0f)
    }

    @Test
    fun pipelineReturnsNoTextWhenVadProducesNoSegments() {
        val result = OfflineVadAsrPipeline(
            vad = fakeVad(),
            asr = fakeAsr(),
            isCancelled = { false },
        ).run(FloatArray(AudioRecorder.FRAME_SAMPLES))

        assertTrue(result.isEmpty())
    }

    @Test
    fun pipelinePreservesMultipleSegmentOrderAndTiming() {
        val first = OfflinePcmSegment(0, floatArrayOf(0.1f))
        val second = OfflinePcmSegment(16_000, floatArrayOf(0.2f))
        val result = OfflineVadAsrPipeline(
            vad = fakeVad(onAccept = listOf(first), onFlush = listOf(second)),
            asr = object : OfflineAmbientAsrProcessor {
                override fun recognize(samples: FloatArray): AmbientRecognition = when (samples.first()) {
                    0.1f -> AmbientRecognition("第一段", "zh", "", "")
                    else -> AmbientRecognition("第二段", "zh", "", "")
                }
            },
            isCancelled = { false },
        ).run(FloatArray(AudioRecorder.FRAME_SAMPLES))

        assertEquals(listOf("第一段", "第二段"), result.map(OfflineAsrSegment::text))
        assertEquals(0, result[0].startMillis)
        assertEquals(1_000, result[1].startMillis)
        assertTrue(first.samples.all { it == 0f })
        assertTrue(second.samples.all { it == 0f })
    }

    @Test
    fun pipelinePropagatesAsrFailureAndClearsSegmentSamples() {
        val segment = OfflinePcmSegment(0, floatArrayOf(0.2f, 0.3f))
        val failure = runCatching {
            OfflineVadAsrPipeline(
                vad = fakeVad(onAccept = listOf(segment)),
                asr = object : OfflineAmbientAsrProcessor {
                    override fun recognize(samples: FloatArray): AmbientRecognition = error("ASR 失败")
                },
                isCancelled = { false },
            ).run(FloatArray(AudioRecorder.FRAME_SAMPLES))
        }.exceptionOrNull()

        assertTrue(failure?.message?.contains("ASR 失败") == true)
        assertTrue(segment.samples.all { it == 0f })
    }

    @Test
    fun pipelineStopsBeforeCallingVadWhenCancelled() {
        var vadCalled = false
        val failure = runCatching {
            OfflineVadAsrPipeline(
                vad = object : OfflineVadProcessor {
                    override fun accept(samples: FloatArray): List<OfflinePcmSegment> {
                        vadCalled = true
                        return emptyList()
                    }

                    override fun flush(): List<OfflinePcmSegment> = emptyList()
                },
                asr = fakeAsr(),
                isCancelled = { true },
            ).run(FloatArray(AudioRecorder.FRAME_SAMPLES))
        }.exceptionOrNull()

        assertTrue(failure is CancellationException)
        assertTrue(!vadCalled)
    }

    private fun fakeVad(
        onAccept: List<OfflinePcmSegment> = emptyList(),
        onFlush: List<OfflinePcmSegment> = emptyList(),
    ) = object : OfflineVadProcessor {
        private var accepted = false

        override fun accept(samples: FloatArray): List<OfflinePcmSegment> =
            if (accepted) emptyList() else onAccept.also { accepted = true }

        override fun flush(): List<OfflinePcmSegment> = onFlush
    }

    private fun fakeAsr() = object : OfflineAmbientAsrProcessor {
        override fun recognize(samples: FloatArray): AmbientRecognition = AmbientRecognition("文本", "zh", "", "")
    }

    private fun wav(sampleRate: Int, channels: Int, samples: ShortArray, audioFormat: Int = 1): ByteArray {
        require(samples.size % channels == 0)
        val dataBytes = samples.size * Short.SIZE_BYTES
        return ByteArrayOutputStream().apply {
            writeAscii("RIFF")
            writeInt(36 + dataBytes)
            writeAscii("WAVE")
            writeAscii("fmt ")
            writeInt(16)
            writeShort(audioFormat)
            writeShort(channels)
            writeInt(sampleRate)
            writeInt(sampleRate * channels * Short.SIZE_BYTES)
            writeShort(channels * Short.SIZE_BYTES)
            writeShort(16)
            writeAscii("data")
            writeInt(dataBytes)
            samples.forEach { writeShort(it.toInt()) }
        }.toByteArray()
    }

    private fun ByteArrayOutputStream.writeAscii(value: String) = write(value.toByteArray(Charsets.US_ASCII))

    private fun ByteArrayOutputStream.writeShort(value: Int) {
        write(value and 0xff)
        write(value ushr 8 and 0xff)
    }

    private fun ByteArrayOutputStream.writeInt(value: Int) {
        write(value and 0xff)
        write(value ushr 8 and 0xff)
        write(value ushr 16 and 0xff)
        write(value ushr 24 and 0xff)
    }

    private fun assertFails(expectedMessage: String, block: () -> Unit) {
        val error = runCatching(block).exceptionOrNull()
        assertTrue("expected failure containing $expectedMessage", error?.message?.contains(expectedMessage) == true)
    }
}
