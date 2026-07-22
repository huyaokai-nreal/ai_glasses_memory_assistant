package com.aiglasses.memoryassistant.demo

import org.json.JSONObject

data class ModelPackSnapshot(
    val state: String = "not_installed",
    val version: String = "",
    val downloadedBytes: Long = 0,
    val totalBytes: Long = 0,
    val lastError: String = "",
) {
    fun toJson(): String = JSONObject()
        .put("state", state)
        .put("version", version)
        .put("downloaded_bytes", downloadedBytes)
        .put("total_bytes", totalBytes)
        .put("last_error", lastError)
        .toString()
}

object ModelPackState {
    private val lock = Any()
    private var snapshot = ModelPackSnapshot()

    fun snapshot(): ModelPackSnapshot = synchronized(lock) { snapshot }

    fun markExisting(version: String?) = update {
        when {
            version.isNullOrBlank() -> ModelPackSnapshot()
            it.version == version && it.state == "ready" -> it
            else -> ModelPackSnapshot(state = "installed", version = version)
        }
    }

    fun markDownloading() = update { ModelPackSnapshot(state = "downloading") }

    fun markProgress(progress: ModelInstallProgress) = update {
        it.copy(
            state = "downloading",
            downloadedBytes = progress.downloadedBytes,
            totalBytes = progress.totalBytes,
        )
    }

    fun markReady(version: String) = update { ModelPackSnapshot(state = "ready", version = version) }

    fun markInstalled(version: String) = update { ModelPackSnapshot(state = "installed", version = version) }

    fun markError(message: String) = update {
        it.copy(state = "error", lastError = message.take(500))
    }

    private inline fun update(block: (ModelPackSnapshot) -> ModelPackSnapshot) {
        synchronized(lock) { snapshot = block(snapshot) }
    }
}
