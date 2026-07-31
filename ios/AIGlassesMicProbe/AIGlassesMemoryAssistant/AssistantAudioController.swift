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
    @Published private(set) var sampleRate = 0
    @Published private(set) var channelCount = 0

    /// Set by IOSNativeBridge before starting ambient listening.
    var pipeline: ModelPipeline?

    var onTtsFinished: (() -> Void)?

    private let audioSession = AVAudioSession.sharedInstance()
    private let engine = AVAudioEngine()
    private let speech = AVSpeechSynthesizer()
    private var selectedInput: AVAudioSessionPortDescription?
    private var resumeListeningAfterSpeech = false
    private var observers: [NSObjectProtocol] = []
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    /// Tracks the timestamp of the first captured frame for relative timing.
    private var firstFrameHostTime: UInt64 = 0

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
        firstFrameHostTime = 0
        do {
            try audioSession.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetoothHFP])
            let candidates = audioSession.availableInputs?.filter { $0.portType == .bluetoothHFP } ?? []
            guard candidates.count == 1, let input = candidates.first else { throw AudioError.requiresUniqueHFP(candidates.map(\.portName)) }
            try audioSession.setPreferredInput(input)
            try audioSession.setActive(true)
            selectedInput = input
            routeName = input.portName
            try startEngine()
            try verifyActiveHFPInput(expected: input)
            state = .listening
        } catch { stop(reason: error.localizedDescription) }
    }

    func stop(reason: String? = nil) {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        speech.stopSpeaking(at: .immediate)
        selectedInput = nil
        resumeListeningAfterSpeech = false
        converter = nil
        targetFormat = nil
        sampleRate = 0
        channelCount = 0
        try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
        state = reason.map(State.failed) ?? .idle
    }

    func speak(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        resumeListeningAfterSpeech = state == .listening
        if resumeListeningAfterSpeech {
            engine.pause()
        } else {
            do {
                try audioSession.setCategory(.playback, mode: .spokenAudio)
                try audioSession.setActive(true)
            } catch {
                state = .failed(error.localizedDescription)
                return
            }
        }
        state = .pausedForSpeech
        speech.speak(AVSpeechUtterance(string: text))
    }

    func webStatus() -> [String: Any] {
        let stateValue: String
        switch state {
        case .idle: stateValue = "idle"
        case .preparing: stateValue = "starting"
        case .listening: stateValue = "recording"
        case .pausedForSpeech: stateValue = "paused_tts"
        case .failed: stateValue = "error"
        }
        let model = BundledModelPackValidator.validate()
        return [
            "platform": "ios",
            "state": stateValue,
            "running": stateValue == "starting" || stateValue == "recording" || stateValue == "paused_tts",
            "input_device_name": routeName,
            "input_device_type": routeName == "未检测" ? "" : "bluetoothHFP",
            "input_device_source": routeName == "未检测" ? "" : "bluetooth_hfp",
            "sample_rate": sampleRate,
            "channels": channelCount,
            "encoding": "pcm_float32_native",
            "captured_samples": frameCount,
            "model_state": model.ready ? (pipeline != nil ? "pipeline_running" : "manifest_valid_pipeline_unavailable") : "invalid",
            "model_version": model.version ?? "",
            "model_self_test": ["state": pipeline != nil ? "pipeline_running" : "unavailable", "reason": pipeline != nil ? "" : "ios_sherpa_pipeline_not_initialized"],
            "transcription_ready": pipeline != nil,
            "last_error": {
                if case let .failed(message) = state { return message }
                return ""
            }(),
            "capabilities": iosAudioCapabilities(),
        ]
    }

    private var isFailure: Bool { if case .failed = state { return true }; return false }

    /// Report iOS audio capabilities to the web UI, matching JS's expected keys.
    /// Speaker enrollment needs both VAD and speaker-embedding models, which
    /// are verified during the bundled model-pack validation.
    private func iosAudioCapabilities() -> [String: Bool] {
        return ["speaker_enrollment_ready": BundledModelPackValidator.validate().ready]
    }

    private func startEngine() throws {
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        let inputFormat = input.inputFormat(forBus: 0)
        sampleRate = Int(inputFormat.sampleRate.rounded())
        channelCount = Int(inputFormat.channelCount)

        // Target format: 16kHz mono Float32 for sherpa-onnx.
        // AVAudioConverter will be used if input format differs.
        targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)
        if let tf = targetFormat, !tf.isEqual(inputFormat) {
            converter = AVAudioConverter(from: inputFormat, to: tf)
        } else {
            converter = nil
        }

        input.installTap(onBus: 0, bufferSize: 1_600, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            let frameCount = Int(buffer.frameLength)
            Task { @MainActor in self.frameCount += frameCount }

            let floats: [Float] = self.extractFloats(buffer: buffer, frameCount: frameCount)
            if !floats.isEmpty {
                let now = self.captureTimestamp()
                self.pipeline?.accept(pcmFloat32: floats, capturedAtNanos: now)
            }
        }
        engine.prepare()
        try engine.start()
    }

    /// Extract mono Float32 samples from the buffer, converting if needed.
    private func extractFloats(buffer: AVAudioPCMBuffer, frameCount: Int) -> [Float] {
        guard frameCount > 0 else { return [] }

        // If conversion is needed and available
        if let converter = converter, let targetFormat = targetFormat {
            guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(frameCount)) else { return [] }
            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            converter.convert(to: converted, error: &error, withInputFrom: inputBlock)
            guard error == nil, let channelData = converted.floatChannelData else { return [] }
            return Array(UnsafeBufferPointer(start: channelData[0], count: Int(converted.frameLength)))
        }

        // Direct extraction
        guard let channelData = buffer.floatChannelData else { return [] }
        let channelCount = Int(buffer.format.channelCount)
        if channelCount == 1 {
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }
        // Multi-channel: take first channel
        return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
    }

    /// Returns a monotonically-increasing nanosecond timestamp for the captured frame.
    private func captureTimestamp() -> UInt64 {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let raw = mach_absolute_time()
        if firstFrameHostTime == 0 { firstFrameHostTime = raw }
        let elapsed = raw - firstFrameHostTime
        return elapsed * UInt64(info.numer) / UInt64(info.denom)
    }

    private func handleRouteChange() {
        guard state == .listening || state == .pausedForSpeech else { return }
        guard let selectedInput, audioSession.currentRoute.inputs.contains(where: { $0.uid == selectedInput.uid && $0.portType == .bluetoothHFP }) else {
            stop(reason: "蓝牙 HFP 输入已变化，已停止收音")
            return
        }
        routeName = selectedInput.portName
    }

    private func verifyActiveHFPInput(expected input: AVAudioSessionPortDescription) throws {
        guard audioSession.currentRoute.inputs.contains(where: { $0.uid == input.uid && $0.portType == .bluetoothHFP }) else {
            throw AudioError.routeMismatch(expected: input.portName, actual: audioSession.currentRoute.inputs.first?.portName)
        }
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
            // Notify that TTS finished (e.g., for pipeline acknowledgement)
            onTtsFinished?()
            guard resumeListeningAfterSpeech, let selectedInput else {
                try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
                state = .idle
                return
            }
            do {
                try verifyActiveHFPInput(expected: selectedInput)
                try startEngine()
                try verifyActiveHFPInput(expected: selectedInput)
                state = .listening
            } catch {
                stop(reason: error.localizedDescription)
            }
        }
    }
}
