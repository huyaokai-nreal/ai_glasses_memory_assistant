package com.aiglasses.memoryassistant.demo

import com.chaquo.python.Python
import org.json.JSONObject

data class RuntimeEndpoint(
    val baseUrl: String,
    val localToken: String,
    val ownerId: String,
)

object PythonRuntime {
    private fun runtimeModule() = Python.getInstance()
        .getModule("ai_glasses_memory_assistant.android_runtime")

    fun start(config: RuntimeConfig): RuntimeEndpoint {
        val payload = JSONObject()
            .put("app_home", config.appHome)
            .put("static_dir", config.staticDir)
            .put("provider", config.provider)
            .put("model", config.model)
            .put("base_url", config.baseUrl)
            .put("api_key", config.apiKey)
            .put("owner_id", config.ownerId)
        val result = runtimeModule().callAttr("start", payload.toString())
        val parsed = JSONObject(result.toString())
        return RuntimeEndpoint(
            baseUrl = parsed.getString("base_url"),
            localToken = parsed.getString("local_token"),
            ownerId = parsed.getString("owner_id"),
        )
    }

    fun startCapture(ownerId: String): String {
        val result = runtimeModule().callAttr("start_capture", ownerId)
        return JSONObject(result.toString()).getString("capture_id")
    }

    fun stopCapture(ownerId: String, captureId: String): JSONObject {
        val result = runtimeModule().callAttr("stop_capture", ownerId, captureId)
        return JSONObject(result.toString())
    }

    fun captureStatus(ownerId: String, captureId: String): JSONObject {
        val result = runtimeModule().callAttr("capture_status", ownerId, captureId)
        return JSONObject(result.toString())
    }

    fun setNetworkState(online: Boolean): JSONObject {
        val result = runtimeModule().callAttr("set_network_state", online)
        return JSONObject(result.toString())
    }

    fun queueStatus(ownerId: String): JSONObject {
        val result = runtimeModule().callAttr("queue_status", ownerId)
        return JSONObject(result.toString())
    }

    fun waitAudioEvent(ownerId: String, eventId: String, timeoutSeconds: Double = 10.0): JSONObject {
        val result = runtimeModule().callAttr("wait_audio_event", ownerId, eventId, timeoutSeconds)
        return JSONObject(result.toString())
    }

    fun speakerProfile(ownerId: String): JSONObject {
        val result = runtimeModule().callAttr("speaker_profile", ownerId)
        return JSONObject(result.toString())
    }

    fun cancelSpeakerEnrollment(ownerId: String, enrollmentSessionId: String): JSONObject {
        val result = runtimeModule().callAttr("cancel_speaker_enrollment", ownerId, enrollmentSessionId)
        return JSONObject(result.toString())
    }

    fun setDeviceState(state: JSONObject): JSONObject {
        val result = runtimeModule().callAttr("set_device_state", state.toString())
        return JSONObject(result.toString())
    }

    fun classifySpeaker(ownerId: String, embedding: FloatArray, modelName: String): JSONObject {
        val values = org.json.JSONArray()
        embedding.forEach(values::put)
        val result = runtimeModule().callAttr("classify_speaker", ownerId, values.toString(), modelName)
        return JSONObject(result.toString())
    }

    fun ingestAudioEvent(ownerId: String, captureId: String, event: JSONObject, privateEvent: JSONObject): JSONObject {
        val result = runtimeModule().callAttr(
            "ingest_audio_event",
            ownerId,
            captureId,
            event.toString(),
            privateEvent.toString(),
        )
        return JSONObject(result.toString())
    }

    fun createDiagnosticBundle(appHome: String, outputPath: String, deviceState: JSONObject): JSONObject {
        val result = runtimeModule().callAttr(
            "create_diagnostic_bundle",
            appHome,
            outputPath,
            deviceState.toString(),
        )
        return JSONObject(result.toString())
    }
}
