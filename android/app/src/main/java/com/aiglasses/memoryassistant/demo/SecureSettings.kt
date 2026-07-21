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

    fun modelManifestUrl(): String = prefs.getString(KEY_MODEL_MANIFEST_URL, "").orEmpty()

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

    fun save(provider: String, model: String, baseUrl: String, apiKey: String, modelManifestUrl: String = "") {
        val normalizedProvider = provider.trim()
        val normalizedModel = model.trim()
        val normalizedBaseUrl = baseUrl.trim().trimEnd('/')
        val normalizedApiKey = apiKey.trim()
        val normalizedManifestUrl = modelManifestUrl.trim()
        require(normalizedProvider.isNotEmpty()) { "Provider 不能为空" }
        require(normalizedModel.isNotEmpty()) { "Model 不能为空" }
        require(normalizedBaseUrl.startsWith("https://")) { "Base URL 必须使用 HTTPS" }
        require(normalizedApiKey.isNotEmpty()) { "API key 不能为空" }
        require(normalizedManifestUrl.isEmpty() || normalizedManifestUrl.startsWith("https://")) {
            "模型清单 URL 必须使用 HTTPS"
        }

        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        val ciphertext = cipher.doFinal(normalizedApiKey.toByteArray(Charsets.UTF_8))
        prefs.edit()
            .putString(KEY_PROVIDER, normalizedProvider)
            .putString(KEY_MODEL, normalizedModel)
            .putString(KEY_BASE_URL, normalizedBaseUrl)
            .putString(KEY_API_CIPHERTEXT, Base64.encodeToString(ciphertext, Base64.NO_WRAP))
            .putString(KEY_API_IV, Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .putString(KEY_MODEL_MANIFEST_URL, normalizedManifestUrl)
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
        const val DEFAULT_PROVIDER = "deepseek"
        const val DEFAULT_MODEL = "deepseek-v4-flash"
        const val DEFAULT_BASE_URL = "https://api.deepseek.com"
        private const val PREFS_NAME = "secure_runtime_settings"
        private const val KEY_PROVIDER = "provider"
        private const val KEY_MODEL = "model"
        private const val KEY_BASE_URL = "base_url"
        private const val KEY_OWNER_ID = "owner_id"
        private const val KEY_MODEL_MANIFEST_URL = "model_manifest_url"
        private const val KEY_API_CIPHERTEXT = "api_key_ciphertext"
        private const val KEY_API_IV = "api_key_iv"
        private const val KEY_ALIAS = "ai_glasses_deepseek_key"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}
