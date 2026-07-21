package com.aiglasses.memoryassistant.demo

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RuntimeAssetsTest {
    @Test
    fun testSharedWebAssetsExtractIntoPrivateStorage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = StaticAssets.extract(context)
        assertTrue(directory.resolve("index.html").isFile)
        assertTrue(directory.resolve("app.js").isFile)
        assertTrue(directory.resolve("styles.css").isFile)
        assertTrue(directory.resolve("audio-worklet.js").isFile)
    }
}
