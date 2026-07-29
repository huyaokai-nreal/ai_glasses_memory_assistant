import AVFoundation
import Foundation

@MainActor
final class AssistantAudioController: NSObject, ObservableObject {
    enum State: Equatable {
        case idle, preparing, listening, pausedForSpeech, failed(String)

        var label: String {
            switch self {
            case .idle: return "未开始"
            case .preparing: return "正在验证蓝牙麦克风"
            case .listening: return "正在持续收音"
            case .pausedForSpeech: return "语音回复中，已暂停收音"
            case let .failed(message): return message
            }
        }
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var routeName = "未检测"
    @Published private(set) var frameCount = 0

    private let audioSession = AVAudioSession.sharedInstance()
    private let engine = AVAudioEngine()
    private let speech = AVSpeechSynthesizer()
    private var selectedInput: AVAudioSessionPortDescription?
    private var observers: [NSObjectProtocol] = []

    override init() {
        super.init()
        speech.delegate = self
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: AVAudioSession.routeChangeNotification, object: audioSession, queue: .main) { [weak self] _ in Task { @MainActor in self?.handleRouteChange() } },
            center.addObserver(forName: AVAudioSession.interruptionNotification, object: audioSession, queue: .main) { [weak self] _ in Task { @MainActor in self?.stop(reason: "音频会话被系统中断") } },
            center.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: audioSession, queue: .main) { [weak self] _ in Task { @MainActor in self?.stop(reason: "系统音频服务已重置") } },
        ]
    }

    deinit { observers.forEach(NotificationCenter.default.removeObserver) }

    func start() {
        guard state == .idle || isFailure else { return }
        state = .preparing
        frameCount = 0
        do {
            try audioSession.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetoothHFP])
            let candidates = audioSession.availableInputs?.filter { $0.portType == .bluetoothHFP } ?? []
            guard candidates.count == 1, let input = candidates.first else { throw AudioError.requiresUniqueHFP(candidates.map(\.portName)) }
            try audioSession.setPreferredInput(input)
            try audioSession.setActive(true)
            guard audioSession.currentRoute.inputs.contains(where: { $0.uid == input.uid && $0.portType == .bluetoothHFP }) else {
                throw AudioError.routeMismatch(expected: input.portName, actual: audioSession.currentRoute.inputs.first?.portName)
            }
            selectedInput = input
            routeName = input.portName
            try startEngine()
            state = .listening
        } catch { stop(reason: error.localizedDescription) }
    }

    func stop(reason: String? = nil) {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        speech.stopSpeaking(at: .immediate)
        selectedInput = nil
        try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
        state = reason.map(State.failed) ?? .idle
    }

    func speak(_ text: String) {
        guard state == .listening, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        engine.pause()
        state = .pausedForSpeech
        speech.speak(AVSpeechUtterance(string: text))
    }

    private var isFailure: Bool { if case .failed = state { return true }; return false }

    private func startEngine() throws {
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1_600, format: input.inputFormat(forBus: 0)) { [weak self] _, _ in
            Task { @MainActor in self?.frameCount += 1 }
        }
        engine.prepare()
        try engine.start()
    }

    private func handleRouteChange() {
        guard state == .listening || state == .pausedForSpeech else { return }
        guard let selectedInput, audioSession.currentRoute.inputs.contains(where: { $0.uid == selectedInput.uid && $0.portType == .bluetoothHFP }) else {
            stop(reason: "蓝牙 HFP 输入已变化，已停止收音")
            return
        }
        routeName = selectedInput.portName
    }

    private enum AudioError: LocalizedError {
        case requiresUniqueHFP([String])
        case routeMismatch(expected: String, actual: String?)

        var errorDescription: String? {
            switch self {
            case let .requiresUniqueHFP(names): return names.isEmpty ? "未检测到蓝牙通话麦克风 (HFP)" : "检测到多个蓝牙麦克风：\(names.joined(separator: "、"))"
            case let .routeMismatch(expected, actual): return "蓝牙麦克风 \(expected) 未实际生效，当前输入为 \(actual ?? "无输入")"
            }
        }
    }
}

extension AssistantAudioController: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            guard state == .pausedForSpeech else { return }
            handleRouteChange()
            guard state != .failed("蓝牙 HFP 输入已变化，已停止收音") else { return }
            do { try startEngine(); state = .listening } catch { stop(reason: error.localizedDescription) }
        }
    }
}
