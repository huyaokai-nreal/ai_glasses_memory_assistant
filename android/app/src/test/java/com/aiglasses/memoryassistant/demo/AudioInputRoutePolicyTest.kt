package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioInputRoutePolicyTest {
    private val phone = AudioInputRoute(1, "手机麦克风", "内置麦克风", isBluetooth = false)
    private val bluetooth = AudioInputRoute(2, "领夹麦克风", "蓝牙通话设备", isBluetooth = true)
    private val secondBluetooth = AudioInputRoute(3, "蓝牙耳机", "蓝牙 LE 耳机", isBluetooth = true)

    @Test
    fun noBluetoothInputAllowsTheSystemRoute() {
        val selection = AudioInputRoutePolicy.select(listOf(phone))

        assertEquals(AudioInputSelection.SystemDefault, selection)
        assertEquals(phone, AudioInputRoutePolicy.verify(selection, phone, listOf(phone)))
    }

    @Test
    fun usbInputRemainsDistinctFromThePhoneMicrophone() {
        val usb = AudioInputRoute(
            4,
            "Mic Pro Receiver",
            "USB 音频设备",
            isBluetooth = false,
            source = AudioInputSource.USB,
        )

        assertEquals(AudioInputSource.USB, usb.source)
        assertEquals(AudioInputSelection.SystemDefault, AudioInputRoutePolicy.select(listOf(phone, usb)))
    }

    @Test
    fun oneBluetoothInputMustBeSelectedAndRouted() {
        val selection = AudioInputRoutePolicy.select(listOf(phone, bluetooth))

        assertEquals(AudioInputSelection.Bluetooth(bluetooth), selection)
        assertEquals(bluetooth, AudioInputRoutePolicy.verify(selection, bluetooth, listOf(phone, bluetooth)))
    }

    @Test
    fun rejectedPreferredBluetoothDeviceStopsBeforeRecording() {
        assertRejected {
            AudioInputRoutePolicy.requirePreferredDevice(AudioInputSelection.Bluetooth(bluetooth), accepted = false)
        }
    }

    @Test
    fun multipleBluetoothInputsAreRejectedWithoutGuessing() {
        val selection = AudioInputRoutePolicy.select(listOf(phone, bluetooth, secondBluetooth))

        assertTrue(selection is AudioInputSelection.AmbiguousBluetooth)
        assertRejected { AudioInputRoutePolicy.verify(selection, phone, listOf(phone, bluetooth, secondBluetooth)) }
    }

    @Test
    fun preferredBluetoothMismatchRejectsThePhoneRoute() {
        val selection = AudioInputSelection.Bluetooth(bluetooth)

        assertRejected { AudioInputRoutePolicy.verify(selection, phone, listOf(phone, bluetooth)) }
    }

    @Test
    fun secondBluetoothAppearingAfterSelectionRejectsTheAmbiguousRoute() {
        assertRejected {
            AudioInputRoutePolicy.verify(
                AudioInputSelection.Bluetooth(bluetooth),
                bluetooth,
                listOf(phone, bluetooth, secondBluetooth),
            )
        }
    }

    @Test
    fun bluetoothAppearingAfterSystemSelectionRejectsTheStaleRoute() {
        assertRejected {
            AudioInputRoutePolicy.verify(
                AudioInputSelection.SystemDefault,
                phone,
                listOf(phone, bluetooth),
            )
        }
    }

    @Test
    fun bluetoothDisconnectRecoveryReturnsToTheSystemRoute() {
        assertEquals(AudioInputSelection.Bluetooth(bluetooth), AudioInputRoutePolicy.select(listOf(phone, bluetooth)))
        assertEquals(AudioInputSelection.SystemDefault, AudioInputRoutePolicy.select(listOf(phone)))
    }

    private fun assertRejected(block: () -> Unit) {
        try {
            block()
            throw AssertionError("Expected route validation to fail")
        } catch (_: IllegalStateException) {
            // The policy uses check/error so callers can follow the existing recording-failure path.
        }
    }
}
