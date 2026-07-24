package com.aiglasses.memoryassistant.demo

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeLocationResultTest {
    @Test
    fun `available result serializes current turn coordinates`() {
        val payload = NativeLocationResult(
            status = "available",
            latitude = 39.987654,
            longitude = 116.123456,
            accuracy = 12.5,
            timestampSeconds = 1_778_131_200.0,
        ).toJson()

        assertEquals("available", payload.getString("status"))
        assertEquals("android_location_manager", payload.getString("source"))
        assertEquals(39.987654, payload.getDouble("latitude"), 0.0)
        assertEquals(116.123456, payload.getDouble("longitude"), 0.0)
        assertEquals(12.5, payload.getDouble("accuracy"), 0.0)
    }

    @Test
    fun `failure statuses never serialize coordinates`() {
        listOf("denied", "disabled", "timeout", "unavailable").forEach { status ->
            val payload = NativeLocationResult.failure(status, "test_$status").toJson()
            assertEquals(status, payload.getString("status"))
            assertEquals("test_$status", payload.getString("error"))
            assertFalse(payload.has("latitude"))
            assertFalse(payload.has("longitude"))
        }
    }

    @Test
    fun `manifest declares foreground location without background location`() {
        val manifest = listOf(
            File("src/main/AndroidManifest.xml"),
            File("app/src/main/AndroidManifest.xml"),
        ).first { it.isFile }.readText()

        assertTrue(manifest.contains("android.permission.FOREGROUND_SERVICE_LOCATION"))
        assertTrue(manifest.contains("android:foregroundServiceType=\"microphone|location\""))
        assertFalse(manifest.contains("android.permission.ACCESS_BACKGROUND_LOCATION"))
    }
}
