package com.aiglasses.memoryassistant.demo

import android.content.Context
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.KeywordSpotter
import com.k2fsa.sherpa.onnx.KeywordSpotterConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineParaformerModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.TenVadModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractor
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import com.k2fsa.sherpa.onnx.SpeechSegment
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import java.io.Closeable

class SherpaVadAdapter(
    @Suppress("UNUSED_PARAMETER") context: Context,
    pack: InstalledModelPack,
    profile: VadRuntimeProfile = VadRuntimeProfile.from(pack),
) : Closeable {
    private val vad: Vad

    init {
        val component = pack.component("vad")
        require(component.engine == profile.backend) { "VAD component and runtime profile disagree" }
        val config = VadModelConfig().apply {
            if (profile.backend == "silero_vad") {
                sileroVadModelConfig = SileroVadModelConfig().apply {
                    model = pack.rolePath(component, "model")
                    threshold = profile.threshold
                    minSilenceDuration = profile.minSilenceSeconds
                    minSpeechDuration = profile.minSpeechSeconds
                    maxSpeechDuration = profile.maxSpeechSeconds
                    windowSize = profile.windowSize
                }
            } else {
                tenVadModelConfig = TenVadModelConfig().apply {
                    model = pack.rolePath(component, "model")
                    threshold = profile.threshold
                    minSilenceDuration = profile.minSilenceSeconds
                    minSpeechDuration = profile.minSpeechSeconds
                    maxSpeechDuration = profile.maxSpeechSeconds
                    windowSize = profile.windowSize
                }
            }
            sampleRate = AudioRecorder.SAMPLE_RATE
            numThreads = profile.numThreads
            provider = "cpu"
            debug = false
        }
        // Downloaded models use app-private absolute paths; sherpa requires a null AssetManager for them.
        vad = Vad(null, config)
    }

    fun accept(samples: FloatArray): List<SpeechSegment> {
        vad.acceptWaveform(samples)
        return drainSegments()
    }

    fun flush(): List<SpeechSegment> {
        vad.flush()
        return drainSegments()
    }

    fun reset() = vad.reset()

    private fun drainSegments(): List<SpeechSegment> = buildList {
            while (!vad.empty()) {
                add(vad.front())
                vad.pop()
            }
        }

    override fun close() = vad.release()
}

class SherpaKeywordAdapter(
    @Suppress("UNUSED_PARAMETER") context: Context,
    pack: InstalledModelPack,
) : Closeable {
    private val spotter: KeywordSpotter
    private var stream: OnlineStream

    init {
        val component = pack.component("kws")
        val config = KeywordSpotterConfig().apply {
            featConfig = featureConfig()
            modelConfig = pack.onlineModel(component)
            keywordsFile = pack.rolePath(component, "keywords")
            keywordsScore = component.options.float("keywords_score", 1.5f)
            keywordsThreshold = component.options.float("keywords_threshold", 0.25f)
            numTrailingBlanks = component.options.optInt("num_trailing_blanks", 1)
            maxActivePaths = component.options.optInt("max_active_paths", 4)
        }
        spotter = KeywordSpotter(null, config)
        stream = spotter.createStream()
    }

    fun accept(samples: FloatArray): KeywordDetection? {
        var consumedSamples = 0
        while (consumedSamples < samples.size) {
            val end = (consumedSamples + DETECTION_CHUNK_SAMPLES).coerceAtMost(samples.size)
            stream.acceptWaveform(samples.copyOfRange(consumedSamples, end), AudioRecorder.SAMPLE_RATE)
            while (spotter.isReady(stream)) spotter.decode(stream)
            consumedSamples = end
            val keyword = spotter.getResult(stream).keyword.trim()
            if (keyword.isNotEmpty()) {
                spotter.reset(stream)
                return KeywordDetection(keyword, consumedSamples)
            }
        }
        return null
    }

    override fun close() {
        stream.release()
        spotter.release()
    }

    companion object {
        private const val DETECTION_CHUNK_SAMPLES = 320
    }
}

data class KeywordDetection(val keyword: String, val consumedSamples: Int)

class SherpaOnlineAsrAdapter(
    @Suppress("UNUSED_PARAMETER") context: Context,
    pack: InstalledModelPack,
) : Closeable {
    private val recognizer: OnlineRecognizer
    private var stream: OnlineStream

    init {
        val component = pack.component("online_asr")
        val config = OnlineRecognizerConfig().apply {
            featConfig = featureConfig()
            modelConfig = pack.onlineModel(component)
            enableEndpoint = true
            decodingMethod = "greedy_search"
            maxActivePaths = component.options.optInt("max_active_paths", 4)
        }
        recognizer = OnlineRecognizer(null, config)
        stream = recognizer.createStream()
    }

    fun accept(samples: FloatArray): Pair<String, Boolean> {
        stream.acceptWaveform(samples, AudioRecorder.SAMPLE_RATE)
        while (recognizer.isReady(stream)) recognizer.decode(stream)
        return recognizer.getResult(stream).text.trim() to recognizer.isEndpoint(stream)
    }

    fun finish(): String {
        stream.inputFinished()
        while (recognizer.isReady(stream)) recognizer.decode(stream)
        return recognizer.getResult(stream).text.trim()
    }

    fun reset() = recognizer.reset(stream)

    override fun close() {
        stream.release()
        recognizer.release()
    }
}

class SherpaAmbientAsrAdapter(
    @Suppress("UNUSED_PARAMETER") context: Context,
    pack: InstalledModelPack,
) : Closeable {
    private val recognizer: OfflineRecognizer

    init {
        val component = pack.component("ambient_asr")
        require(component.engine == "sense_voice") { "ambient_asr engine must be sense_voice" }
        val senseVoice = OfflineSenseVoiceModelConfig().apply {
            model = pack.rolePath(component, "model")
            language = component.options.optString("language", "auto")
            useInverseTextNormalization = component.options.optBoolean("use_itn", true)
        }
        val model = OfflineModelConfig().apply {
            this.senseVoice = senseVoice
            tokens = pack.rolePath(component, "tokens")
            numThreads = component.options.optInt("num_threads", 2).coerceAtLeast(1)
            provider = "cpu"
            debug = false
        }
        recognizer = OfflineRecognizer(
            null,
            OfflineRecognizerConfig().apply {
                featConfig = featureConfig()
                modelConfig = model
                decodingMethod = "greedy_search"
            },
        )
    }

    fun recognize(samples: FloatArray): AmbientRecognition {
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(samples, AudioRecorder.SAMPLE_RATE)
            recognizer.decode(stream)
            val result = recognizer.getResult(stream)
            AmbientRecognition(
                text = result.text.trim(),
                language = result.lang,
                emotion = result.emotion,
                acousticEvent = result.event,
            )
        } finally {
            stream.release()
        }
    }

    override fun close() = recognizer.release()
}

data class AmbientRecognition(
    val text: String,
    val language: String,
    val emotion: String,
    val acousticEvent: String,
)

class SherpaSpeakerAdapter(
    @Suppress("UNUSED_PARAMETER") context: Context,
    pack: InstalledModelPack,
) : Closeable {
    private val extractor: SpeakerEmbeddingExtractor

    init {
        val component = pack.component("speaker")
        require(component.engine == "speaker_embedding") { "speaker engine must be speaker_embedding" }
        extractor = SpeakerEmbeddingExtractor(
            null,
            SpeakerEmbeddingExtractorConfig(
                pack.rolePath(component, "model"),
                component.options.optInt("num_threads", 1).coerceAtLeast(1),
                false,
                "cpu",
            ),
        )
    }

    fun compute(samples: FloatArray): FloatArray? {
        val stream = extractor.createStream()
        return try {
            stream.acceptWaveform(samples, AudioRecorder.SAMPLE_RATE)
            if (extractor.isReady(stream)) extractor.compute(stream) else null
        } finally {
            stream.release()
        }
    }

    override fun close() = extractor.release()
}

private fun featureConfig() = FeatureConfig().apply {
    sampleRate = AudioRecorder.SAMPLE_RATE
    featureDim = 80
    dither = 0f
}

private fun InstalledModelPack.component(name: String): ModelPackComponent =
    manifest.components[name] ?: error("installed model pack missing component: $name")

private fun InstalledModelPack.rolePath(component: ModelPackComponent, role: String): String {
    val relative = component.roles[role] ?: error("model component missing role: $role")
    return directory.resolve(relative).absolutePath
}

private fun InstalledModelPack.onlineModel(component: ModelPackComponent): OnlineModelConfig =
    OnlineModelConfig().apply {
        tokens = rolePath(component, "tokens")
        numThreads = component.options.optInt("num_threads", 2).coerceAtLeast(1)
        provider = "cpu"
        debug = false
        when (component.engine) {
            "online_paraformer" -> paraformer = OnlineParaformerModelConfig().apply {
                encoder = rolePath(component, "encoder")
                decoder = rolePath(component, "decoder")
            }
            "online_transducer" -> transducer = OnlineTransducerModelConfig().apply {
                encoder = rolePath(component, "encoder")
                decoder = rolePath(component, "decoder")
                joiner = rolePath(component, "joiner")
            }
            else -> error("unsupported online model engine: ${component.engine}")
        }
    }

private fun org.json.JSONObject.float(name: String, fallback: Float): Float =
    optDouble(name, fallback.toDouble()).toFloat()
