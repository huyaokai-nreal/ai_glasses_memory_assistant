package com.aiglasses.memoryassistant.demo

import org.json.JSONObject

data class ModelPackFile(
    val path: String,
    val url: String,
    val sha256: String,
    val sizeBytes: Long,
)

data class ModelPackComponent(
    val engine: String,
    val roles: Map<String, String>,
    val options: JSONObject,
)

data class ModelPackManifest(
    val version: String,
    val sherpaOnnxVersion: String,
    val installMode: String,
    val files: List<ModelPackFile>,
    val components: Map<String, ModelPackComponent>,
    val rawJson: String,
) {
    companion object {
        const val SCHEMA = "model_pack.v1"
        const val SHERPA_VERSION = "1.13.4"
        const val INSTALL_REMOTE_FILES = "remote_files"
        const val INSTALL_ADB_LOCAL = "adb_local"
        private val requiredComponents = setOf("vad", "kws", "online_asr", "ambient_asr", "speaker")
        private val safeVersion = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
        private val safePathPart = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
        private val sha256 = Regex("[a-f0-9]{64}")

        fun parse(rawJson: String): ModelPackManifest {
            val root = JSONObject(rawJson)
            require(root.optString("schema") == SCHEMA) { "model manifest schema must be $SCHEMA" }
            val version = root.requireString("version")
            require(safeVersion.matches(version)) { "model manifest version is invalid" }
            val sherpaVersion = root.requireString("sherpa_onnx_version")
            require(sherpaVersion == SHERPA_VERSION) {
                "model pack requires sherpa-onnx $sherpaVersion, app provides $SHERPA_VERSION"
            }
            val installMode = root.optString("install_mode", INSTALL_REMOTE_FILES).trim()
            require(installMode in setOf(INSTALL_REMOTE_FILES, INSTALL_ADB_LOCAL)) {
                "unsupported model manifest install_mode: $installMode"
            }

            val filesJson = root.optJSONArray("files") ?: error("model manifest files must be an array")
            require(filesJson.length() > 0) { "model manifest files cannot be empty" }
            val files = buildList {
                for (index in 0 until filesJson.length()) {
                    val item = filesJson.optJSONObject(index) ?: error("model file $index must be an object")
                    val path = normalizeRelativePath(item.requireString("path"))
                    val url = item.optString("url").trim()
                    if (installMode == INSTALL_REMOTE_FILES) {
                        require(url.startsWith("https://")) { "model URL must use HTTPS: $path" }
                    } else {
                        require(url.isEmpty()) { "adb_local model file must not declare a URL: $path" }
                    }
                    val digest = item.requireString("sha256").lowercase()
                    require(sha256.matches(digest)) { "model SHA-256 is invalid: $path" }
                    val size = item.optLong("size_bytes", -1)
                    require(size > 0) { "model size_bytes must be positive: $path" }
                    add(ModelPackFile(path, url, digest, size))
                }
            }
            require(files.map(ModelPackFile::path).toSet().size == files.size) {
                "model manifest contains duplicate file paths"
            }
            files.fold(0L) { total, file -> Math.addExact(total, file.sizeBytes) }

            val componentsJson = root.optJSONObject("components") ?: error("model manifest components must be an object")
            val components = buildMap {
                componentsJson.keys().forEach { name ->
                    val item = componentsJson.optJSONObject(name) ?: error("model component $name must be an object")
                    val rolesJson = item.optJSONObject("roles") ?: error("model component $name roles must be an object")
                    val roles = buildMap {
                        rolesJson.keys().forEach { role ->
                            put(role, normalizeRelativePath(rolesJson.requireString(role)))
                        }
                    }
                    require(roles.isNotEmpty()) { "model component $name roles cannot be empty" }
                    put(
                        name,
                        ModelPackComponent(
                            engine = item.requireString("engine"),
                            roles = roles,
                            options = item.optJSONObject("options") ?: JSONObject(),
                        ),
                    )
                }
            }
            require(components.keys.containsAll(requiredComponents)) {
                "model manifest missing components: ${(requiredComponents - components.keys).sorted().joinToString()}"
            }
            val knownPaths = files.map(ModelPackFile::path).toSet()
            components.forEach { (name, component) ->
                component.roles.forEach { (role, path) ->
                    require(path in knownPaths) { "model component $name role $role references an unknown file" }
                }
            }
            return ModelPackManifest(
                version = version,
                sherpaOnnxVersion = sherpaVersion,
                installMode = installMode,
                files = files,
                components = components,
                rawJson = root.toString(),
            )
        }

        private fun normalizeRelativePath(raw: String): String {
            val path = raw.trim().replace('\\', '/')
            val parts = path.split('/')
            require(path.isNotEmpty() && !path.startsWith('/') && parts.all { safePathPart.matches(it) }) {
                "model path must be a safe relative path"
            }
            require(parts.none { it == "." || it == ".." }) { "model path cannot traverse directories" }
            return parts.joinToString("/")
        }

        private fun JSONObject.requireString(name: String): String = optString(name).trim().also {
            require(it.isNotEmpty()) { "model manifest $name cannot be empty" }
        }
    }
}
