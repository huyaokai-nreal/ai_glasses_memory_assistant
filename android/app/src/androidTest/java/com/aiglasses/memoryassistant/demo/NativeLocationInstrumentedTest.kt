package com.aiglasses.memoryassistant.demo

import android.content.Intent
import androidx.test.platform.app.InstrumentationRegistry
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeLocationInstrumentedTest {
    @Test
    fun pythonPreflightKeepsDeviceLocationDemandScoped() {
        assertTrue(PythonRuntime.locationPreflight("今天的天气怎么样").getBoolean("needed"))
        assertFalse(PythonRuntime.locationPreflight("北京天气怎么样").getBoolean("needed"))
        assertFalse(PythonRuntime.locationPreflight("讲个笑话").getBoolean("needed"))
    }

    @Test
    fun oneShotProviderReturnsAnAvailableLocationOnAuthorizedDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val activity = instrumentation.startActivitySync(
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        val provider = NativeLocationProvider(context)
        val completed = CountDownLatch(1)
        var result: NativeLocationResult? = null

        try {
            provider.request {
                result = it
                completed.countDown()
            }

            assertTrue("one-shot location did not complete", completed.await(10, TimeUnit.SECONDS))
            assertEquals(
                "location status=${result?.status}, error=${result?.error}",
                "available",
                result?.status,
            )
            assertNotNull(result?.latitude)
            assertNotNull(result?.longitude)
        } finally {
            provider.close()
            instrumentation.runOnMainSync { activity.finish() }
        }
    }
}
