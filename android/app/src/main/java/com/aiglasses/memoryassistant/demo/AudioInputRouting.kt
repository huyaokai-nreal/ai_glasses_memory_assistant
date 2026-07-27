package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.AudioRecord
import android.os.Build

enum class AudioInputSource(val wireName: String) {
    BLUETOOTH("bluetooth"),
    USB("usb"),
    SYSTEM("system"),
}

data class AudioInputRoute(
    val id: Int,
    val label: String,
    val typeLabel: String,
    val isBluetooth: Boolean,
    val source: AudioInputSource = if (isBluetooth) AudioInputSource.BLUETOOTH else AudioInputSource.SYSTEM,
)

sealed interface AudioInputSelection {
    object SystemDefault : AudioInputSelection
    data class Bluetooth(val route: AudioInputRoute) : AudioInputSelection
    data class AmbiguousBluetooth(val routes: List<AudioInputRoute>) : AudioInputSelection
}

data class PreparedAudioRecord(
    val recorder: AudioRecord,
    val selection: AudioInputSelection,
)

object AudioInputRoutePolicy {
    fun select(routes: List<AudioInputRoute>): AudioInputSelection {
        val bluetoothRoutes = routes.filter(AudioInputRoute::isBluetooth)
        return when (bluetoothRoutes.size) {
            0 -> AudioInputSelection.SystemDefault
            1 -> AudioInputSelection.Bluetooth(bluetoothRoutes.single())
            else -> AudioInputSelection.AmbiguousBluetooth(bluetoothRoutes)
        }
    }

    fun verify(
        selection: AudioInputSelection,
        actualRoute: AudioInputRoute?,
        currentRoutes: List<AudioInputRoute>,
    ): AudioInputRoute {
        val actual = checkNotNull(actualRoute) { "Android 未返回实际收音设备" }
        return when (selection) {
            AudioInputSelection.SystemDefault -> {
                check(currentRoutes.none(AudioInputRoute::isBluetooth)) {
                    "检测到蓝牙收音设备已接入，请重新开始收音以切换蓝牙输入"
                }
                actual
            }
            is AudioInputSelection.Bluetooth -> {
                val bluetoothRoutes = currentRoutes.filter(AudioInputRoute::isBluetooth)
                check(bluetoothRoutes.size == 1 && bluetoothRoutes.single().id == selection.route.id) {
                    "蓝牙收音设备状态已变化，请重新开始收音"
                }
                check(actual.id == selection.route.id && actual.isBluetooth) {
                    "蓝牙收音设备 ${selection.route.label} 未实际生效，已拒绝使用 ${actual.label} 收音"
                }
                actual
            }
            is AudioInputSelection.AmbiguousBluetooth -> error(
                "检测到多个蓝牙收音设备：${selection.routes.joinToString("、") { it.label }}，请断开无关设备后重试",
            )
        }
    }

    fun requirePreferredDevice(selection: AudioInputSelection, accepted: Boolean) {
        if (selection is AudioInputSelection.Bluetooth) {
            check(accepted) { "无法将收音切换到蓝牙设备 ${selection.route.label}" }
        }
    }
}

object AndroidAudioInputRouting {
    fun prepare(context: Context, recorder: AudioRecord): PreparedAudioRecord {
        val available = availableInputs(context)
        val selection = AudioInputRoutePolicy.select(available.map(AvailableInput::route))
        when (selection) {
            is AudioInputSelection.AmbiguousBluetooth -> error(
                "检测到多个蓝牙收音设备：${selection.routes.joinToString("、") { it.label }}，请断开无关设备后重试",
            )
            is AudioInputSelection.Bluetooth -> {
                val device = checkNotNull(available.firstOrNull { it.route.id == selection.route.id }) {
                    "蓝牙收音设备 ${selection.route.label} 已不可用"
                }.device
                AudioInputRoutePolicy.requirePreferredDevice(selection, recorder.setPreferredDevice(device))
            }
            AudioInputSelection.SystemDefault -> Unit
        }
        return PreparedAudioRecord(recorder, selection)
    }

    fun verify(context: Context, prepared: PreparedAudioRecord): AudioInputRoute =
        AudioInputRoutePolicy.verify(
            selection = prepared.selection,
            actualRoute = prepared.recorder.routedDevice?.let(::routeFrom),
            currentRoutes = availableInputs(context).map(AvailableInput::route),
        )

    private fun availableInputs(context: Context): List<AvailableInput> =
        context.getSystemService(AudioManager::class.java)
            .getDevices(AudioManager.GET_DEVICES_INPUTS)
            .map { device -> AvailableInput(device, routeFrom(device)) }

    private fun routeFrom(device: AudioDeviceInfo): AudioInputRoute {
        val type = device.type
        val isBleHeadset = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && type == AudioDeviceInfo.TYPE_BLE_HEADSET
        val source = when {
            type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO || isBleHeadset -> AudioInputSource.BLUETOOTH
            type == AudioDeviceInfo.TYPE_USB_DEVICE -> AudioInputSource.USB
            else -> AudioInputSource.SYSTEM
        }
        val typeLabel = when {
            type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "蓝牙通话设备"
            isBleHeadset -> "蓝牙 LE 耳机"
            type == AudioDeviceInfo.TYPE_BUILTIN_MIC -> "内置麦克风"
            type == AudioDeviceInfo.TYPE_WIRED_HEADSET -> "有线耳机麦克风"
            type == AudioDeviceInfo.TYPE_USB_DEVICE -> "USB 音频设备"
            else -> "其他输入设备"
        }
        return AudioInputRoute(
            id = device.id,
            label = device.productName.toString().trim().ifBlank { typeLabel },
            typeLabel = typeLabel,
            isBluetooth = source == AudioInputSource.BLUETOOTH,
            source = source,
        )
    }

    private data class AvailableInput(
        val device: AudioDeviceInfo,
        val route: AudioInputRoute,
    )
}
