package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioInputProbeEvaluatorTest {
    private val bluetoothRoute = AudioInputRoute(
        id = 1,
        label = "测试蓝牙麦克风",
        typeLabel = "蓝牙通话设备",
        isBluetooth = true,
    )

    @Test
    fun bluetoothRouteWithRecognizedSpeechPasses() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(route = bluetoothRoute, receivedFrames = true, peakDbfs = -80.0, transcript = "测试蓝牙收音"),
        )

        assertTrue(result.passed)
        assertEquals("收音正常", result.status)
    }

    @Test
    fun systemInputPassesWhenNoBluetoothInputWasSelected() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(
                route = AudioInputRoute(2, "手机麦克风", "内置麦克风", isBluetooth = false),
                receivedFrames = true,
                peakDbfs = -30.0,
            ),
        )

        assertTrue(result.passed)
        assertEquals("收音正常", result.status)
    }

    @Test
    fun unknownRouteIsReported() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(receivedFrames = true, peakDbfs = -30.0),
        )

        assertFalse(result.passed)
        assertEquals("未确认收音设备", result.status)
    }

    @Test
    fun bluetoothRouteWithoutFramesFailsClearly() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(route = bluetoothRoute),
        )

        assertFalse(result.passed)
        assertEquals("未收到音频数据", result.status)
    }

    @Test
    fun quietBluetoothInputWithoutTranscriptRequestsSpeechRetry() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(
                route = bluetoothRoute,
                receivedFrames = true,
                peakDbfs = AudioInputProbeEvaluator.SPEECH_CONFIRMATION_PEAK_DBFS - 0.1,
            ),
        )

        assertFalse(result.passed)
        assertEquals("未检测到可识别语音", result.status)
    }

    @Test
    fun unavailableAsrDoesNotHideWorkingInput() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(
                route = bluetoothRoute,
                receivedFrames = true,
                peakDbfs = -90.0,
                asrUnavailable = true,
            ),
        )

        assertTrue(result.passed)
        assertEquals("收音正常", result.status)
        assertTrue(result.guidance.contains("ASR 未就绪"))
    }

    @Test
    fun routeFailureRemainsFailedEvenWhenAudioWasReceived() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(
                route = bluetoothRoute,
                receivedFrames = true,
                transcript = "不应使用错误路由的文字",
                error = "蓝牙收音设备未实际生效，已拒绝使用手机麦克风收音",
            ),
        )

        assertFalse(result.passed)
        assertEquals("测试失败", result.status)
    }

    @Test
    fun permissionOrInitializationErrorIsPreserved() {
        val result = AudioInputProbeEvaluator.evaluate(
            AudioInputProbeSnapshot(error = "麦克风权限未授予"),
        )

        assertFalse(result.passed)
        assertEquals("测试失败", result.status)
        assertEquals("麦克风权限未授予", result.guidance)
    }
}
