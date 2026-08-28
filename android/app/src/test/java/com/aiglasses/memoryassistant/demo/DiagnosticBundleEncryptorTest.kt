package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import javax.crypto.Cipher
import javax.crypto.CipherInputStream
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec

class DiagnosticBundleEncryptorTest {
    @Test
    fun encryptedBundleRoundTripsWithItsPassword() {
        val plaintext = "private diagnostic payload".toByteArray()
        val password = "eight-or-more".toCharArray()
        val output = ByteArrayOutputStream()

        DiagnosticBundleEncryptor.encrypt(ByteArrayInputStream(plaintext), output, password)

        val encoded = DataInputStream(ByteArrayInputStream(output.toByteArray()))
        assertEquals("AIGDIAG1", String(ByteArray(8).also(encoded::readFully), Charsets.US_ASCII))
        val iterations = encoded.readInt()
        assertEquals(DiagnosticBundleEncryptor.iterations, iterations)
        val salt = ByteArray(encoded.readUnsignedByte()).also(encoded::readFully)
        val iv = ByteArray(encoded.readUnsignedByte()).also(encoded::readFully)
        val specification = PBEKeySpec(password, salt, iterations, 256)
        val key = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(specification).encoded
        specification.clearPassword()
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(128, iv))
        key.fill(0)

        val restored = CipherInputStream(encoded, cipher).use { it.readBytes() }

        assertArrayEquals(plaintext, restored)
    }

    @Test
    fun adbSnapshotPolicyRejectsReleaseAndUnsafePaths() {
        assertThrows(IllegalArgumentException::class.java) {
            AdbDiagnosticSnapshotPolicy.requireDebugBuild(false)
        }
        AdbDiagnosticSnapshotPolicy.requireDebugBuild(true)

        val fileName = AdbDiagnosticSnapshotPolicy.fileName("a".repeat(32))
        val relativePath = AdbDiagnosticSnapshotPolicy.relativePath(fileName)

        assertEquals("cache/adb-diagnostics/$fileName", relativePath)
        assertTrue(AdbDiagnosticSnapshotPolicy.isValidRelativePath(relativePath))
        assertFalse(AdbDiagnosticSnapshotPolicy.isValidRelativePath("cache/adb-diagnostics/../secret.zip"))
        assertThrows(IllegalArgumentException::class.java) {
            AdbDiagnosticSnapshotPolicy.fileName("not-a-token")
        }
    }

    @Test
    fun adbUsageSnapshotPolicyRejectsReleaseAndUnsafePaths() {
        assertThrows(IllegalArgumentException::class.java) {
            AdbUsageSnapshotPolicy.requireDebugBuild(false)
        }
        AdbUsageSnapshotPolicy.requireDebugBuild(true)

        val fileName = AdbUsageSnapshotPolicy.fileName("b".repeat(32))
        val relativePath = AdbUsageSnapshotPolicy.relativePath(fileName)

        assertEquals("cache/adb-usage-snapshots/$fileName", relativePath)
        assertTrue(AdbUsageSnapshotPolicy.isValidRelativePath(relativePath))
        assertFalse(AdbUsageSnapshotPolicy.isValidRelativePath("cache/adb-usage-snapshots/../secret.zip"))
        assertThrows(IllegalArgumentException::class.java) {
            AdbUsageSnapshotPolicy.fileName("not-a-token")
        }
    }
}
