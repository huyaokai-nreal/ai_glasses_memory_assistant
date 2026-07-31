import AVFoundation
import Foundation

// MARK: - Data models (mirror Android's AudioInputProbeSnapshot / AudioInputProbeResult)

struct AudioProbeSnapshot {
    let route: String
    let isBluetooth: Bool
    let receivedFrames: Bool
    let peakDbfs: Float
    let transcript: String
}

struct AudioProbeResult {
    let passed: Bool
    let status: String
    let guidance: String
}

// MARK: - Online Mic Probe

/// Records 5 s of audio, measures peak dBFS, and optionally runs online ASR.
/// Mirrors Android's AudioInputProbe.
final class AudioInputProber {
    private let engine = AVAudioEngine()
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    private var samples: [Float] = []
    private var peakDbfs: Float = -120
    private var isRunning = false
    private var routeName = "未确认"
    private var isBluetooth = false
    private var receivedFrames = false
    private var cancelled = false

    /// Called on a background thread — must be dispatched back to MainActor by the caller.
    var onUpdate: ((AudioProbeSnapshot) -> Void)?
    var onComplete: ((AudioProbeResult) -> Void)?

    private let testDuration: TimeInterval = 5.0

    func start() throws {
        cancelled = false
        samples.removeAll(keepingCapacity: true)
        peakDbfs = -120
        receivedFrames = false
        isRunning = true

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetoothHFP, .allowBluetoothA2DP, .defaultToSpeaker])
        try session.setActive(true)

        let currentRoute = session.currentRoute
        let inputPort = currentRoute.inputs.first
        routeName = inputPort?.portName ?? "系统麦克风"
        isBluetooth = inputPort?.portType == .bluetoothHFP || inputPort?.portType == .bluetoothA2DP || inputPort?.portType == .bluetoothLE

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw ProberError.converterUnavailable
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self, self.isRunning else { return }
            self.receivedFrames = true

            // buffer.frameLength is UInt32; sampleRate is Double — convert explicitly
            let ratio = self.targetFormat.sampleRate / inputFormat.sampleRate
            let targetCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
            guard let targetBuffer = AVAudioPCMBuffer(pcmFormat: self.targetFormat, frameCapacity: targetCapacity) else { return }

            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            converter.convert(to: targetBuffer, error: &error, withInputFrom: inputBlock)

            if let channelData = targetBuffer.floatChannelData {
                let frameCount = Int(targetBuffer.frameLength)
                let ptr = channelData[0]
                let frame = Array(UnsafeBufferPointer(start: ptr, count: frameCount))
                self.samples.append(contentsOf: frame)

                var maxAbs: Float = 0
                for i in 0..<frameCount { maxAbs = max(maxAbs, abs(ptr[i])) }
                let peak = 20 * log10(max(maxAbs, 1e-10))
                self.peakDbfs = max(self.peakDbfs, peak)
            }

            let snapshot = AudioProbeSnapshot(
                route: self.routeName,
                isBluetooth: self.isBluetooth,
                receivedFrames: true,
                peakDbfs: self.peakDbfs,
                transcript: ""
            )
            DispatchQueue.main.async { self.onUpdate?(snapshot) }
        }

        engine.prepare()
        try engine.start()

        // Stop after 5 s
        DispatchQueue.global().asyncAfter(deadline: .now() + testDuration) { [weak self] in
            self?.finish()
        }
    }

    private func finish() {
        guard isRunning else { return }
        isRunning = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        if cancelled {
            let result = AudioProbeResult(passed: false, status: "已取消", guidance: "测试已被取消")
            DispatchQueue.main.async { self.onComplete?(result) }
            return
        }

        let result = evaluate()
        DispatchQueue.main.async { self.onComplete?(result) }
    }

    private func evaluate() -> AudioProbeResult {
        if !receivedFrames {
            return AudioProbeResult(passed: false, status: "收音异常", guidance: "未收到任何音频数据，请检查麦克风权限及蓝牙设备连接状态")
        }
        if peakDbfs < -55 {
            return AudioProbeResult(passed: false, status: "收音异常", guidance: "收音峰值低于 -55 dBFS，可能麦克风静音、增益过低或设备未正确连接")
        }
        return AudioProbeResult(passed: true, status: "收音正常", guidance: "有声纹时\"停止回复\"及后续身份服务将更准确")
    }

    func cancel() {
        cancelled = true
        if isRunning { finish() }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}

enum ProberError: LocalizedError {
    case converterUnavailable
    var errorDescription: String? {
        switch self {
        case .converterUnavailable: return "音频格式转换器不可用"
        }
    }
}
