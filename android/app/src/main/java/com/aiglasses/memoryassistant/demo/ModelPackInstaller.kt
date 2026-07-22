package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.os.StatFs
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID

data class ModelInstallProgress(
    val fileIndex: Int,
    val fileCount: Int,
    val downloadedBytes: Long,
    val totalBytes: Long,
)

data class InstalledModelPack(
    val version: String,
    val directory: File,
    val manifest: ModelPackManifest,
)

class ModelPackInstaller(context: Context) {
    private val root = context.applicationContext.filesDir.resolve("models")

    fun currentVersion(): String? = runCatching {
        JSONObject(root.resolve(CURRENT_FILE).readText()).getString("version")
            .takeIf { it.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")) }
    }.getOrNull()

    fun current(): InstalledModelPack? {
        val pointer = root.resolve(CURRENT_FILE)
        if (!pointer.isFile) return null
        return runCatching {
            val pointerJson = JSONObject(pointer.readText())
            val version = pointerJson.getString("version")
            val directoryName = pointerJson.getString("directory")
            check(directoryName.matches(Regex("[A-Za-z0-9._-]{1,128}")))
            val directory = root.resolve("packs").resolve(directoryName)
            val manifest = ModelPackManifest.parse(directory.resolve(MANIFEST_FILE).readText())
            check(manifest.version == version)
            checkInstalledFiles(directory, manifest)
            InstalledModelPack(version, directory, manifest)
        }.getOrNull()
    }

    fun install(rawManifest: String, onProgress: (ModelInstallProgress) -> Unit = {}): InstalledModelPack {
        check(!NativeAudioState.snapshot().running) { "录音中不能更新模型，请先停止全天待机" }
        val manifest = ModelPackManifest.parse(rawManifest)
        check(manifest.installMode == ModelPackManifest.INSTALL_REMOTE_FILES) {
            "adb_local 模型包只能通过 ADB 本地安装"
        }
        current()?.takeIf { it.version == manifest.version }?.let { return it }

        val totalBytes = manifest.files.fold(0L) { total, file -> Math.addExact(total, file.sizeBytes) }
        val requiredBytes = Math.multiplyExact(totalBytes, 2L)
        root.mkdirs()
        check(StatFs(root.absolutePath).availableBytes >= requiredBytes) {
            "存储空间不足，需要至少 ${requiredBytes / (1024 * 1024)} MB 可用空间"
        }

        val staging = root.resolve(".staging-${manifest.version}-${UUID.randomUUID()}")
        val packsDirectory = root.resolve("packs")
        val destination = packsDirectory.resolve("${manifest.version}-${UUID.randomUUID()}")
        staging.mkdirs()
        var downloadedBytes = 0L
        try {
            manifest.files.forEachIndexed { index, file ->
                val target = staging.resolve(file.path)
                target.parentFile?.mkdirs()
                downloadVerified(file, target) { currentFileBytes ->
                    onProgress(
                        ModelInstallProgress(
                            fileIndex = index + 1,
                            fileCount = manifest.files.size,
                            downloadedBytes = downloadedBytes + currentFileBytes,
                            totalBytes = totalBytes,
                        ),
                    )
                }
                downloadedBytes += file.sizeBytes
            }
            staging.resolve(MANIFEST_FILE).writeText(manifest.rawJson)
            checkInstalledFiles(staging, manifest)
            packsDirectory.mkdirs()
            Files.move(staging.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE)
            writeCurrentPointer(manifest.version, destination.name)
            packsDirectory.listFiles()
                ?.filter { it.isDirectory && it != destination }
                ?.forEach(File::deleteRecursively)
            return InstalledModelPack(manifest.version, destination, manifest)
        } catch (error: Throwable) {
            staging.deleteRecursively()
            throw error
        }
    }

    fun fetchManifest(rawUrl: String): String {
        require(rawUrl.startsWith("https://")) { "模型清单 URL 必须使用 HTTPS" }
        val connection = openHttps(rawUrl)
        return try {
            connection.inputStream.buffered().use { input ->
                val output = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    check(output.size() + count <= MAX_MANIFEST_BYTES) { "模型清单超过大小限制" }
                    output.write(buffer, 0, count)
                }
                output.toString(Charsets.UTF_8.name())
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun downloadVerified(file: ModelPackFile, target: File, onBytes: (Long) -> Unit) {
        val temporary = target.resolveSibling(".${target.name}.part")
        temporary.delete()
        val digest = MessageDigest.getInstance("SHA-256")
        var count = 0L
        val connection = openHttps(file.url)
        try {
            connection.inputStream.buffered().use { input ->
                temporary.outputStream().buffered().use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        count += read
                        check(count <= file.sizeBytes) { "model download exceeds declared size: ${file.path}" }
                        digest.update(buffer, 0, read)
                        output.write(buffer, 0, read)
                        onBytes(count)
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
        check(count == file.sizeBytes) { "model download size mismatch: ${file.path}" }
        val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
        check(actualSha256 == file.sha256) { "model checksum mismatch: ${file.path}" }
        Files.move(
            temporary.toPath(),
            target.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun openHttps(rawUrl: String): HttpURLConnection {
        var url = URI(rawUrl).toURL()
        repeat(MAX_REDIRECTS + 1) { redirectCount ->
            check(url.protocol == "https") { "model URL redirect must use HTTPS" }
            val connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = CONNECT_TIMEOUT_MILLIS
                readTimeout = READ_TIMEOUT_MILLIS
                instanceFollowRedirects = false
                setRequestProperty("Accept-Encoding", "identity")
            }
            val code = connection.responseCode
            if (code in 200..299) return connection
            if (code !in 300..399 || redirectCount == MAX_REDIRECTS) {
                connection.disconnect()
                error("model download HTTP $code")
            }
            val location = connection.getHeaderField("Location") ?: error("model redirect is missing Location")
            url = URI(url.toString()).resolve(location).toURL()
            connection.disconnect()
        }
        error("too many model download redirects")
    }

    private fun checkInstalledFiles(directory: File, manifest: ModelPackManifest) {
        manifest.files.forEach { file ->
            val installed = directory.resolve(file.path)
            check(installed.isFile && installed.length() == file.sizeBytes) {
                "installed model file is missing or incomplete: ${file.path}"
            }
            check(sha256(installed) == file.sha256) {
                "installed model checksum mismatch: ${file.path}"
            }
        }
    }

    private fun writeCurrentPointer(version: String, directory: String) {
        val pointer = root.resolve(CURRENT_FILE)
        val temporary = root.resolve(".$CURRENT_FILE.part")
        temporary.writeText(
            JSONObject()
                .put("version", version)
                .put("directory", directory)
                .toString(),
        )
        Files.move(
            temporary.toPath(),
            pointer.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val CURRENT_FILE = "current.json"
        private const val MANIFEST_FILE = "manifest.json"
        private const val MAX_REDIRECTS = 5
        private const val CONNECT_TIMEOUT_MILLIS = 15_000
        private const val READ_TIMEOUT_MILLIS = 60_000
        private const val MAX_MANIFEST_BYTES = 1024 * 1024
    }
}
