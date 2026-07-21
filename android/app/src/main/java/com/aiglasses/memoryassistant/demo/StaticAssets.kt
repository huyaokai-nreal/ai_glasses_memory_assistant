package com.aiglasses.memoryassistant.demo

import android.content.Context
import java.io.File

object StaticAssets {
    private val files = listOf("index.html", "app.js", "styles.css", "audio-worklet.js")

    fun extract(context: Context): File {
        val destination = context.filesDir.resolve("web/static")
        val marker = destination.resolve(".version")
        val expectedVersion = BuildConfig.VERSION_CODE.toString()
        if (marker.takeIf(File::isFile)?.readText() == expectedVersion && files.all { destination.resolve(it).isFile }) {
            return destination
        }
        destination.mkdirs()
        files.forEach { name ->
            val target = destination.resolve(name)
            val temporary = destination.resolve(".$name.tmp")
            context.assets.open(name).use { input ->
                temporary.outputStream().use(input::copyTo)
            }
            if (target.exists() && !target.delete()) error("无法替换静态资源：$name")
            if (!temporary.renameTo(target)) error("无法安装静态资源：$name")
        }
        marker.writeText(expectedVersion)
        return destination
    }
}
