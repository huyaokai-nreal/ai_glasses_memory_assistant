package com.aiglasses.memoryassistant.demo

import android.content.Context
import java.io.File

object StaticAssets {
    private val files = listOf("index.html", "app.js", "styles.css", "audio-worklet.js")

    @Synchronized
    fun extract(context: Context): File {
        val webRoot = context.filesDir.resolve("web").apply { mkdirs() }
        val destination = webRoot.resolve("static")
        val staging = webRoot.resolve(".static-staging")
        val backup = webRoot.resolve(".static-backup")
        if (!destination.exists() && backup.isDirectory) {
            check(backup.renameTo(destination)) { "无法恢复上一份静态资源" }
        }
        staging.deleteRecursively()
        staging.mkdirs()
        try {
            files.forEach { name ->
                context.assets.open(name).use { input ->
                    staging.resolve(name).outputStream().use(input::copyTo)
                }
            }
            check(files.all { staging.resolve(it).isFile }) { "APK 静态资源不完整" }
            backup.deleteRecursively()
            if (destination.exists()) check(destination.renameTo(backup)) { "无法备份上一份静态资源" }
            if (!staging.renameTo(destination)) {
                if (!destination.exists() && backup.exists()) backup.renameTo(destination)
                error("无法激活 APK 静态资源")
            }
            backup.deleteRecursively()
            return destination
        } catch (error: Throwable) {
            staging.deleteRecursively()
            if (!destination.exists() && backup.exists()) backup.renameTo(destination)
            throw error
        }
    }
}
