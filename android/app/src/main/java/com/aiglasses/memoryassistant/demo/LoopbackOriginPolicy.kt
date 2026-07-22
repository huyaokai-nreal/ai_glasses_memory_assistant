package com.aiglasses.memoryassistant.demo

import java.net.URI

object LoopbackOriginPolicy {
    fun allows(origin: String?, runtimeBaseUrl: String): Boolean {
        if (origin.isNullOrBlank() || runtimeBaseUrl.isBlank()) return false
        return runCatching {
            val candidate = URI(origin)
            val expected = URI(runtimeBaseUrl)
            candidate.scheme == "http" &&
                candidate.scheme == expected.scheme &&
                candidate.host == "127.0.0.1" &&
                candidate.host == expected.host &&
                candidate.port == expected.port &&
                candidate.userInfo == null &&
                candidate.query == null &&
                candidate.fragment == null &&
                (candidate.path.isNullOrEmpty() || candidate.path == "/")
        }.getOrDefault(false)
    }
}
