package com.aiglasses.memoryassistant.demo

import org.json.JSONObject

/** Resolved VAD settings shared by native inference and exported event telemetry. */
data class VadRuntimeProfile(
    val backend: String,
    val modelPackVersion: String,
    val threshold: Float,
    val minSilenceSeconds: Float,
    val minSpeechSeconds: Float,
    val maxSpeechSeconds: Float,
    val windowSize: Int,
    val numThreads: Int,
) {
    fun parametersJson(): JSONObject = JSONObject()
        .put("threshold", threshold)
        .put("min_silence_seconds", minSilenceSeconds)
        .put("min_speech_seconds", minSpeechSeconds)
        .put("max_speech_seconds", maxSpeechSeconds)
        .put("window_size", windowSize)
        .put("num_threads", numThreads)

    fun eventJson(state: String): JSONObject = parametersJson()
        .put("state", state)
        .put("backend", backend)
        .put("model_pack_version", modelPackVersion)

    companion object {
        fun from(pack: InstalledModelPack): VadRuntimeProfile {
            val component = pack.manifest.components.getValue("vad")
            return from(component.engine, pack.version, component.options)
        }

        fun from(backend: String, modelPackVersion: String, options: JSONObject): VadRuntimeProfile {
            require(backend in setOf("silero_vad", "ten_vad")) { "unsupported vad engine: $backend" }
            return VadRuntimeProfile(
                backend = backend,
                modelPackVersion = modelPackVersion,
                threshold = options.optDouble("threshold", 0.5).toFloat(),
                minSilenceSeconds = options.optDouble("min_silence_seconds", 0.5).toFloat(),
                minSpeechSeconds = options.optDouble("min_speech_seconds", 0.25).toFloat(),
                maxSpeechSeconds = options.optDouble("max_speech_seconds", 30.0).toFloat(),
                windowSize = options.optInt("window_size", if (backend == "ten_vad") 256 else 512),
                numThreads = options.optInt("num_threads", 1).coerceAtLeast(1),
            )
        }
    }
}
