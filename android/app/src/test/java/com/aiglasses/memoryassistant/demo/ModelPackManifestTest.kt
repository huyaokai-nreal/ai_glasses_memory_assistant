package com.aiglasses.memoryassistant.demo

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ModelPackManifestTest {
    @Test
    fun parsesCompletePinnedManifest() {
        val manifest = ModelPackManifest.parse(manifestJson())

        assertEquals("test-1", manifest.version)
        assertEquals(ModelPackManifest.SHERPA_VERSION, manifest.sherpaOnnxVersion)
        assertEquals(ModelPackManifest.INSTALL_REMOTE_FILES, manifest.installMode)
        assertEquals(setOf("vad", "kws", "online_asr", "ambient_asr", "speaker"), manifest.components.keys)
    }

    @Test
    fun parsesAdbLocalManifestWithoutUrls() {
        val root = JSONObject(manifestJson())
            .put("install_mode", ModelPackManifest.INSTALL_ADB_LOCAL)
        root.getJSONArray("files").getJSONObject(0).remove("url")

        val manifest = ModelPackManifest.parse(root.toString())

        assertEquals(ModelPackManifest.INSTALL_ADB_LOCAL, manifest.installMode)
        assertEquals("", manifest.files.single().url)
    }

    @Test
    fun rejectsRemoteManifestWithoutHttpsUrl() {
        val root = JSONObject(manifestJson())
        root.getJSONArray("files").getJSONObject(0).remove("url")

        assertThrows(IllegalArgumentException::class.java) {
            ModelPackManifest.parse(root.toString())
        }
    }

    @Test
    fun rejectsUrlOnAdbLocalManifest() {
        val root = JSONObject(manifestJson())
            .put("install_mode", ModelPackManifest.INSTALL_ADB_LOCAL)

        assertThrows(IllegalArgumentException::class.java) {
            ModelPackManifest.parse(root.toString())
        }
    }

    @Test
    fun rejectsDirectoryTraversal() {
        val root = JSONObject(manifestJson())
        root.getJSONArray("files").getJSONObject(0).put("path", "../model.onnx")

        assertThrows(IllegalArgumentException::class.java) {
            ModelPackManifest.parse(root.toString())
        }
    }

    @Test
    fun rejectsComponentRoleWithoutVerifiedFile() {
        val root = JSONObject(manifestJson())
        root.getJSONObject("components").getJSONObject("vad")
            .getJSONObject("roles").put("model", "vad/missing.onnx")

        assertThrows(IllegalArgumentException::class.java) {
            ModelPackManifest.parse(root.toString())
        }
    }

    private fun manifestJson(): String {
        val modelPath = "shared/model.onnx"
        val files = JSONArray().put(
            JSONObject()
                .put("path", modelPath)
                .put("url", "https://models.example.test/model.onnx")
                .put("sha256", "0".repeat(64))
                .put("size_bytes", 10),
        )
        val components = JSONObject()
        listOf("vad", "kws", "online_asr", "ambient_asr", "speaker").forEach { name ->
            components.put(
                name,
                JSONObject()
                    .put("engine", "test")
                    .put("roles", JSONObject().put("model", modelPath)),
            )
        }
        return JSONObject()
            .put("schema", ModelPackManifest.SCHEMA)
            .put("version", "test-1")
            .put("sherpa_onnx_version", ModelPackManifest.SHERPA_VERSION)
            .put("files", files)
            .put("components", components)
            .toString()
    }
}
