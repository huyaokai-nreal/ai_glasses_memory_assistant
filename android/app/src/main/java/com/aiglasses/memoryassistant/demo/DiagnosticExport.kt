package com.aiglasses.memoryassistant.demo

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.CipherOutputStream
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec

object DiagnosticBundleEncryptor {
    private val magic = "AIGDIAG1".toByteArray(Charsets.US_ASCII)
    private const val saltSize = 16
    private const val ivSize = 12
    const val iterations = 210_000

    fun encrypt(input: InputStream, output: OutputStream, passphrase: CharArray, random: SecureRandom = SecureRandom()) {
        require(passphrase.size >= 8) { "导出密码至少需要 8 个字符" }
        val salt = ByteArray(saltSize).also(random::nextBytes)
        val iv = ByteArray(ivSize).also(random::nextBytes)
        val specification = PBEKeySpec(passphrase, salt, iterations, 256)
        val keyBytes = try {
            SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(specification).encoded
        } finally {
            specification.clearPassword()
        }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(keyBytes, "AES"), GCMParameterSpec(128, iv))
        keyBytes.fill(0)
        DataOutputStream(BufferedOutputStream(output)).use { destination ->
            destination.write(magic)
            destination.writeInt(iterations)
            destination.writeByte(salt.size)
            destination.write(salt)
            destination.writeByte(iv.size)
            destination.write(iv)
            CipherOutputStream(destination, cipher).use { encrypted -> input.copyTo(encrypted) }
        }
    }
}

class DiagnosticExporter(private val context: Context) {
    fun export(destination: Uri, passphrase: CharArray): JSONObject {
        var temporary: File? = null
        return try {
            require(!NativeAudioState.snapshot().running) { "请先停止全天收音再导出诊断包" }
            val appHome = context.filesDir.resolve("runtime")
            val temporaryFile = File.createTempFile("ai-glasses-diagnostic-", ".zip", context.cacheDir)
            temporary = temporaryFile
            val metadata = deviceMetadata()
            val bundle = PythonRuntime.createDiagnosticBundle(appHome.absolutePath, temporaryFile.absolutePath, metadata)
            context.contentResolver.openOutputStream(destination, "w").use { output ->
                requireNotNull(output) { "无法打开导出文件" }
                temporaryFile.inputStream().buffered().use { input ->
                    DiagnosticBundleEncryptor.encrypt(input, output, passphrase)
                }
            }
            bundle.put("encrypted", true).put("format", "AIGDIAG1")
        } finally {
            passphrase.fill('\u0000')
            temporary?.delete()
        }
    }

    private fun deviceMetadata(): JSONObject {
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        val battery = requireNotNull(context.getSystemService(BatteryManager::class.java))
        val activityManager = requireNotNull(context.getSystemService(ActivityManager::class.java))
        val memory = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageInfo.longVersionCode
        } else {
            @Suppress("DEPRECATION")
            packageInfo.versionCode.toLong()
        }
        val supportedAbis = org.json.JSONArray().apply { Build.SUPPORTED_ABIS.forEach(::put) }
        val queue = runCatching { PythonRuntime.queueStatus(SecureSettings(context).ownerId()).optJSONObject("summary") }
            .getOrNull()
        return JSONObject()
            .put("app_version_name", packageInfo.versionName.orEmpty())
            .put("app_version_code", versionCode)
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("supported_abis", supportedAbis)
            .put("battery_percent", battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY))
            .put("battery_charge_counter_uah", battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER))
            .put("memory_available_bytes", memory.availMem)
            .put("memory_low", memory.lowMemory)
            .put("native_audio", NativeAudioState.snapshot().toJson())
            .put("queue_summary", queue ?: JSONObject())
    }
}
