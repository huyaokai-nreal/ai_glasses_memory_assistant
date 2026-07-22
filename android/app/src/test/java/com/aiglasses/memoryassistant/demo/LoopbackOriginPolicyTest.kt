package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LoopbackOriginPolicyTest {
    private val runtime = "http://127.0.0.1:43127"

    @Test
    fun acceptsOnlyCurrentRuntimeOrigin() {
        assertTrue(LoopbackOriginPolicy.allows(runtime, runtime))
        assertTrue(LoopbackOriginPolicy.allows("$runtime/", runtime))
        assertFalse(LoopbackOriginPolicy.allows("http://127.0.0.1:43128", runtime))
        assertFalse(LoopbackOriginPolicy.allows("http://localhost:43127", runtime))
        assertFalse(LoopbackOriginPolicy.allows("https://127.0.0.1:43127", runtime))
    }

    @Test
    fun rejectsExternalAndAmbiguousOrigins() {
        assertFalse(LoopbackOriginPolicy.allows("http://example.com:43127", runtime))
        assertFalse(LoopbackOriginPolicy.allows("http://example.com@127.0.0.1:43127", runtime))
        assertFalse(LoopbackOriginPolicy.allows("$runtime/path", runtime))
        assertFalse(LoopbackOriginPolicy.allows("$runtime?next=outside", runtime))
        assertFalse(LoopbackOriginPolicy.allows(null, runtime))
    }
}
