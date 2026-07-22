package com.aiglasses.memoryassistant.demo

import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.zip.ZipFile

@RunWith(AndroidJUnit4::class)
class RuntimeAssetsTest {
    @Test
    fun testSharedWebAssetsExtractIntoPrivateStorage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val firstDirectory = StaticAssets.extract(context)
        firstDirectory.resolve("app.js").writeText("stale-web-resource")

        val directory = StaticAssets.extract(context)

        listOf("index.html", "app.js", "styles.css", "audio-worklet.js").forEach { name ->
            assertTrue(directory.resolve(name).isFile)
            val packaged = context.assets.open(name).use { it.readBytes() }
            assertTrue("asset mismatch: $name", packaged.contentEquals(directory.resolve(name).readBytes()))
        }
    }

    @Test
    fun testInstalledModelPackPassesFiveComponentSelfTest() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val pack = ModelPackInstaller(context).current()
        assumeNotNull(pack)

        val result = ModelSelfTestRunner(context).run(requireNotNull(pack))

        assertEquals("ok", result.state)
        assertEquals(setOf("vad", "kws", "online_asr", "ambient_asr", "speaker"), result.components.keys)
    }

    @Test
    fun testDiagnosticBundleSanitizesAndEncryptsIsolatedFixture() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = context.cacheDir.resolve("diagnostic-fixture-${UUID.randomUUID()}")
        val data = root.resolve("data").apply { mkdirs() }
        val timeline = data.resolve("timeline.db")
        val plainZip = context.cacheDir.resolve("diagnostic-${UUID.randomUUID()}.zip")
        val extracted = context.cacheDir.resolve("diagnostic-${UUID.randomUUID()}.db")
        try {
            SQLiteDatabase.openOrCreateDatabase(timeline, null).use { database ->
                database.execSQL("CREATE TABLE speaker_profiles (user_id TEXT, embedding TEXT)")
                database.execSQL("CREATE TABLE speaker_enrollment_samples (user_id TEXT, embedding TEXT)")
                database.execSQL(
                    "CREATE TABLE device_audio_events (event_id TEXT, private_payload TEXT, event_payload TEXT, dispatch_payload TEXT)",
                )
                database.execSQL("INSERT INTO speaker_profiles VALUES ('u1', '[0.1,0.2]')")
                database.execSQL("INSERT INTO speaker_enrollment_samples VALUES ('u1', '[0.3,0.4]')")
                database.execSQL(
                    "INSERT INTO device_audio_events VALUES (?, ?, ?, ?)",
                    arrayOf(
                        "event-1",
                        JSONObject().put("speaker_embedding", org.json.JSONArray(listOf(0.5, 0.6))).toString(),
                        JSONObject().put("text", "hello").toString(),
                        JSONObject().put("state", "completed").toString(),
                    ),
                )
            }

            val result = PythonRuntime.createDiagnosticBundle(
                root.absolutePath,
                plainZip.absolutePath,
                JSONObject().put("api_key", "fixture-secret").put("model", "X4000"),
            )

            assertEquals(2, result.getJSONObject("redaction").getInt("voice_profile_rows_removed"))
            ZipFile(plainZip).use { archive ->
                val metadata = archive.getInputStream(archive.getEntry("diagnostics.json")).bufferedReader().readText()
                assertFalse(metadata.contains("fixture-secret"))
                archive.getInputStream(archive.getEntry("database/timeline.db")).use { input ->
                    extracted.outputStream().use(input::copyTo)
                }
            }
            SQLiteDatabase.openDatabase(extracted.absolutePath, null, SQLiteDatabase.OPEN_READONLY).use { database ->
                database.rawQuery("SELECT COUNT(*) FROM speaker_profiles", null).use { cursor ->
                    assertTrue(cursor.moveToFirst())
                    assertEquals(0, cursor.getInt(0))
                }
                database.rawQuery("SELECT private_payload FROM device_audio_events", null).use { cursor ->
                    assertTrue(cursor.moveToFirst())
                    assertEquals("{}", cursor.getString(0))
                }
            }
            val encrypted = ByteArrayOutputStream()
            plainZip.inputStream().use { input ->
                DiagnosticBundleEncryptor.encrypt(input, encrypted, "fixture-password".toCharArray())
            }
            assertEquals("AIGDIAG1", String(encrypted.toByteArray().copyOfRange(0, 8), Charsets.US_ASCII))
            assertFalse(encrypted.toByteArray().containsSlice("fixture-secret".toByteArray()))
        } finally {
            root.deleteRecursively()
            plainZip.delete()
            extracted.delete()
        }
    }

    private fun ByteArray.containsSlice(needle: ByteArray): Boolean {
        if (needle.isEmpty() || needle.size > size) return false
        return (0..size - needle.size).any { offset ->
            needle.indices.all { index -> this[offset + index] == needle[index] }
        }
    }
}
