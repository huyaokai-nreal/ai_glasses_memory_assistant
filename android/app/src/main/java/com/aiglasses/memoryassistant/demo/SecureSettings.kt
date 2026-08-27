package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

data class RuntimeConfig(
    val appHome: String,
    val staticDir: String,
    val provider: String,
    val model: String,
    val baseUrl: String,
    val apiKey: String,
    val ownerId: String,
)

class SecureSettings(context: Context) {
    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun isConfigured(): Boolean = apiKey().isNotBlank() && model().isNotBlank() && baseUrl().isNotBlank()

    fun provider(): String = prefs.getString(KEY_PROVIDER, DEFAULT_PROVIDER).orEmpty()

    fun model(): String = prefs.getString(KEY_MODEL, DEFAULT_MODEL).orEmpty()

    fun baseUrl(): String = prefs.getString(KEY_BASE_URL, DEFAULT_BASE_URL).orEmpty()

    fun ownerId(): String {
        val existing = prefs.getString(KEY_OWNER_ID, null)
        if (!existing.isNullOrBlank()) return existing
        val generated = "android_${UUID.randomUUID().toString().replace("-", "").take(16)}"
        prefs.edit().putString(KEY_OWNER_ID, generated).apply()
        return generated
    }

    fun apiKey(): String {
        val encodedCiphertext = prefs.getString(KEY_API_CIPHERTEXT, null) ?: return ""
        val encodedIv = prefs.getString(KEY_API_IV, null) ?: return ""
        return runCatching {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                secretKey(),
                GCMParameterSpec(128, Base64.decode(encodedIv, Base64.NO_WRAP)),
            )
            String(cipher.doFinal(Base64.decode(encodedCiphertext, Base64.NO_WRAP)), Charsets.UTF_8)
        }.getOrDefault("")
    }

    fun save(provider: String, model: String, baseUrl: String, apiKey: String) {
        val normalizedProvider = provider.trim()
        val normalizedModel = model.trim()
        val normalizedBaseUrl = baseUrl.trim().trimEnd('/')
        val normalizedApiKey = apiKey.trim()
        require(normalizedProvider.isNotEmpty()) { "Provider 不能为空" }
        require(normalizedModel.isNotEmpty()) { "Model 不能为空" }
        require(normalizedBaseUrl.startsWith("https://") || isLocalOrPrivateHost(normalizedBaseUrl)) {
            "Base URL 必须使用 HTTPS（内网/本地地址允许 HTTP）"
        }
        require(normalizedApiKey.isNotEmpty()) { "API key 不能为空" }

        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        val ciphertext = cipher.doFinal(normalizedApiKey.toByteArray(Charsets.UTF_8))
        prefs.edit()
            .putString(KEY_PROVIDER, normalizedProvider)
            .putString(KEY_MODEL, normalizedModel)
            .putString(KEY_BASE_URL, normalizedBaseUrl)
            .putString(KEY_API_CIPHERTEXT, Base64.encodeToString(ciphertext, Base64.NO_WRAP))
            .putString(KEY_API_IV, Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .apply()
        ownerId()
    }

    fun runtimeConfig(staticDir: String): RuntimeConfig = RuntimeConfig(
        appHome = appContext.filesDir.resolve("runtime").absolutePath,
        staticDir = staticDir,
        provider = provider(),
        model = model(),
        baseUrl = baseUrl(),
        apiKey = apiKey(),
        ownerId = ownerId(),
    )

    // Local/private hosts may use plain HTTP so a LAN test LLM (e.g. qwen on 10.x) works
    // without TLS; public hosts still require HTTPS.
    private fun isLocalOrPrivateHost(url: String): Boolean {
        val host = runCatching { java.net.URL(url).host }.getOrNull() ?: return false
        if (host == "localhost" || host == "127.0.0.1" || host == "[::1]") return true
        val parts = host.split('.').mapNotNull { it.toIntOrNull() }
        if (parts.size != 4) return false
        val (a, b) = parts[0] to parts[1]
        return when {
            a == 10 -> true
            a == 172 && b in 16..31 -> true
            a == 192 && b == 168 -> true
            a == 169 && b == 254 -> true
            a == 127 -> true
            else -> false
        }
    }

    private fun secretKey(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (keyStore.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").run {
            init(
                KeyGenParameterSpec.Builder(
                    KEY_ALIAS,
                    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
                )
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .build(),
            )
            generateKey()
        }
    }

    companion object {
        const val DEFAULT_PROVIDER = "llama_cpp"
        const val DEFAULT_MODEL = "qwen3.8-27b-32k"
        const val DEFAULT_BASE_URL = "http://10.252.17.5:11438/v1"
        private const val PREFS_NAME = "secure_runtime_settings"
        private const val KEY_PROVIDER = "provider"
        private const val KEY_MODEL = "model"
        private const val KEY_BASE_URL = "base_url"
        private const val KEY_OWNER_ID = "owner_id"
        private const val KEY_API_CIPHERTEXT = "api_key_ciphertext"
        private const val KEY_API_IV = "api_key_iv"
        private const val KEY_ALIAS = "ai_glasses_deepseek_key"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}
