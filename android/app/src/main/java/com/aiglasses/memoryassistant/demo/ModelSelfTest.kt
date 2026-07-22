package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.os.Debug
import android.os.SystemClock
import org.json.JSONObject

data class ModelComponentSelfTest(
    val state: String,
    val elapsedMs: Long,
    val error: String = "",
) {
    fun toJson() = JSONObject()
        .put("state", state)
        .put("elapsed_ms", elapsedMs)
        .put("error", error)

    companion object {
        fun fromJson(value: JSONObject) = ModelComponentSelfTest(
            state = value.optString("state", "not_run"),
            elapsedMs = value.optLong("elapsed_ms", 0L),
            error = value.optString("error"),
        )
    }
}

data class ModelSelfTestSnapshot(
    val state: String = "not_run",
    val version: String = "",
    val checkedAtMillis: Long = 0L,
    val pssKb: Int = 0,
    val components: Map<String, ModelComponentSelfTest> = emptyMap(),
) {
    fun toJson() = JSONObject()
        .put("state", state)
        .put("version", version)
        .put("checked_at_ms", checkedAtMillis)
        .put("pss_kb", pssKb)
        .put(
            "components",
            JSONObject().apply { components.forEach { (name, result) -> put(name, result.toJson()) } },
        )

    companion object {
        fun fromJson(raw: String): ModelSelfTestSnapshot {
            val root = JSONObject(raw)
            val componentsJson = root.optJSONObject("components") ?: JSONObject()
            val components = buildMap {
                componentsJson.keys().forEach { name ->
                    componentsJson.optJSONObject(name)?.let { put(name, ModelComponentSelfTest.fromJson(it)) }
                }
            }
            return ModelSelfTestSnapshot(
                state = root.optString("state", "not_run"),
                version = root.optString("version"),
                checkedAtMillis = root.optLong("checked_at_ms", 0L),
                pssKb = root.optInt("pss_kb", 0),
                components = components,
            )
        }
    }
}

object ModelSelfTestState {
    private const val PREFERENCES = "model_self_test"
    private const val RESULT = "result_json"
    private val lock = Any()
    private var snapshot = ModelSelfTestSnapshot()

    fun snapshot(): ModelSelfTestSnapshot = synchronized(lock) { snapshot }

    fun restore(context: Context, installedVersion: String?): ModelSelfTestSnapshot = synchronized(lock) {
        val raw = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).getString(RESULT, "").orEmpty()
        snapshot = runCatching { ModelSelfTestSnapshot.fromJson(raw) }
            .getOrNull()
            ?.takeIf { it.version.isNotBlank() && it.version == installedVersion }
            ?: ModelSelfTestSnapshot(version = installedVersion.orEmpty())
        if (snapshot.state == "ok") ModelPackState.markReady(snapshot.version)
        snapshot
    }

    fun markRunning(version: String) = update(ModelSelfTestSnapshot(state = "running", version = version))

    fun save(context: Context, value: ModelSelfTestSnapshot) {
        update(value)
        val saved = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(RESULT, value.toJson().toString())
            .commit()
        check(saved) { "模型自检结果保存失败" }
        if (value.state == "ok") ModelPackState.markReady(value.version)
        else ModelPackState.markError("本地模型自检未通过")
    }

    fun clear(context: Context, version: String = "") {
        check(context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit().remove(RESULT).commit()) {
            "旧模型自检结果清除失败"
        }
        update(ModelSelfTestSnapshot(version = version))
    }

    private fun update(value: ModelSelfTestSnapshot) {
        synchronized(lock) { snapshot = value }
    }
}

class ModelSelfTestRunner(private val context: Context) {
    fun run(pack: InstalledModelPack): ModelSelfTestSnapshot {
        ModelSelfTestState.markRunning(pack.version)
        val checks = linkedMapOf<String, ModelComponentSelfTest>()
        checks["vad"] = check {
            SherpaVadAdapter(context, pack).use { adapter ->
                adapter.accept(FloatArray(512))
                adapter.flush()
            }
        }
        checks["kws"] = check {
            SherpaKeywordAdapter(context, pack).use { adapter -> adapter.accept(FloatArray(1_600)) }
        }
        checks["online_asr"] = check {
            SherpaOnlineAsrAdapter(context, pack).use { adapter ->
                adapter.accept(FloatArray(1_600))
                adapter.finish()
            }
        }
        checks["ambient_asr"] = check {
            SherpaAmbientAsrAdapter(context, pack).use { adapter -> adapter.recognize(FloatArray(16_000)) }
        }
        checks["speaker"] = check {
            SherpaSpeakerAdapter(context, pack).use { adapter -> adapter.compute(FloatArray(32_000)) }
        }
        val result = ModelSelfTestSnapshot(
            state = if (checks.values.all { it.state == "ok" }) "ok" else "failed",
            version = pack.version,
            checkedAtMillis = System.currentTimeMillis(),
            pssKb = Debug.getPss().coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
            components = checks,
        )
        ModelSelfTestState.save(context, result)
        return result
    }

    private inline fun check(block: () -> Unit): ModelComponentSelfTest {
        val started = SystemClock.elapsedRealtime()
        return runCatching(block).fold(
            onSuccess = { ModelComponentSelfTest("ok", SystemClock.elapsedRealtime() - started) },
            onFailure = { error ->
                ModelComponentSelfTest(
                    state = "failed",
                    elapsedMs = SystemClock.elapsedRealtime() - started,
                    error = sanitize(error),
                )
            },
        )
    }

    private fun sanitize(error: Throwable): String {
        val detail = error.message.orEmpty()
            .replace(Regex("/data/(?:user|data)/[^\\s]+"), "<app-private-path>")
            .take(300)
        return listOf(error.javaClass.simpleName, detail).filter(String::isNotBlank).joinToString(": ")
    }
}
