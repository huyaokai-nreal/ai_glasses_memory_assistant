import Foundation

enum AudioInputKind: String, Equatable {
    case bluetoothHFP
    case bluetoothA2DP
    case builtInMic
    case wiredMic
    case usbAudio
    case other

    var displayName: String {
        switch self {
        case .bluetoothHFP: return "蓝牙通话麦克风 (HFP)"
        case .bluetoothA2DP: return "蓝牙播放设备 (A2DP，无麦克风输入)"
        case .builtInMic: return "iPhone 内置麦克风"
        case .wiredMic: return "有线耳机麦克风"
        case .usbAudio: return "USB 音频设备"
        case .other: return "其他音频输入"
        }
    }
}

struct AudioInputRoute: Identifiable, Equatable {
    let id: String
    let name: String
    let kind: AudioInputKind
}

enum AudioRouteError: Error, Equatable, LocalizedError {
    case noBluetoothMicrophone
    case ambiguousBluetoothMicrophones([AudioInputRoute])
    case preferredInputUnavailable(String)
    case routeMismatch(expected: AudioInputRoute, actual: AudioInputRoute?)

    var errorDescription: String? {
        switch self {
        case .noBluetoothMicrophone:
            return "未检测到可用的蓝牙 HFP 麦克风。请确认 Insta Mic Pro 已作为 iPhone 音频输入连接。"
        case let .ambiguousBluetoothMicrophones(routes):
            let names = routes.map(\.name).joined(separator: "、")
            return "检测到多个蓝牙麦克风：\(names)。请断开无关设备后重试。"
        case let .preferredInputUnavailable(name):
            return "蓝牙麦克风 \(name) 已不可用，请重新检测。"
        case let .routeMismatch(expected, actual):
            let actualName = actual?.name ?? "无输入"
            return "蓝牙麦克风 \(expected.name) 未实际生效，当前输入为 \(actualName)。已拒绝使用手机麦克风录音。"
        }
    }
}

enum AudioRoutePolicy {
    static func selectPreferredBluetoothMicrophone(from inputs: [AudioInputRoute]) throws -> AudioInputRoute {
        let bluetoothMicrophones = inputs.filter { $0.kind == .bluetoothHFP }
        switch bluetoothMicrophones.count {
        case 0:
            throw AudioRouteError.noBluetoothMicrophone
        case 1:
            return bluetoothMicrophones[0]
        default:
            throw AudioRouteError.ambiguousBluetoothMicrophones(bluetoothMicrophones)
        }
    }

    static func requirePreferredInputAvailable(
        _ selected: AudioInputRoute,
        in currentInputs: [AudioInputRoute],
    ) throws -> AudioInputRoute {
        guard currentInputs.contains(selected) else {
            throw AudioRouteError.preferredInputUnavailable(selected.name)
        }
        return selected
    }

    static func verifyActiveInput(expected: AudioInputRoute, actual: AudioInputRoute?) throws -> AudioInputRoute {
        guard actual == expected, actual?.kind == .bluetoothHFP else {
            throw AudioRouteError.routeMismatch(expected: expected, actual: actual)
        }
        return expected
    }
}
