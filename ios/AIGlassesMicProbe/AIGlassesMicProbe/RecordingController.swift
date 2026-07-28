import AVFoundation
import Combine
import Foundation
import UIKit

struct SavedRecording: Identifiable, Equatable {
    let url: URL
    let createdAt: Date
    let duration: TimeInterval

    var id: URL { url }
}

@MainActor
final class RecordingController: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var availableInputs: [AudioInputRoute] = []
    @Published private(set) var activeInput: AudioInputRoute?
    @Published private(set) var recordingState: RecordingState = .idle
    @Published private(set) var level: Double = 0
    @Published private(set) var elapsedSeconds: TimeInterval = 0
    @Published private(set) var recordings: [SavedRecording] = []
    @Published private(set) var playingRecordingID: URL?
    @Published private(set) var statusMessage = "请先连接蓝牙麦克风，再重新检测。"

    var isRecording: Bool { recordingState.isRecording }
    var isPreparing: Bool { recordingState.isPreparing }

    private let audioSession = AVAudioSession.sharedInstance()
    private var lifecycle = RecordingLifecycle()
    private var recorder: AVAudioRecorder?
    private var player: AVAudioPlayer?
    private var meterTimer: Timer?
    private var elapsedTimer: Timer?
    private var recordingStartedAt: Date?
    private var observers: [NSObjectProtocol] = []

    deinit {
        observers.forEach(NotificationCenter.default.removeObserver)
        meterTimer?.invalidate()
        elapsedTimer?.invalidate()
    }

    func startMonitoring() {
        guard observers.isEmpty else { return }
        let center = NotificationCenter.default
        observers = [
            center.addObserver(
                forName: AVAudioSession.routeChangeNotification,
                object: audioSession,
                queue: .main,
            ) { [weak self] _ in
                self?.handleRouteChange()
            },
            center.addObserver(
                forName: AVAudioSession.interruptionNotification,
                object: audioSession,
                queue: .main,
            ) { [weak self] notification in
                self?.handleInterruption(notification)
            },
            center.addObserver(
                forName: AVAudioSession.mediaServicesWereResetNotification,
                object: audioSession,
                queue: .main,
            ) { [weak self] _ in
                self?.stopForSystemEvent(.mediaServicesReset, message: "iOS 音频服务已重置，已停止录音。")
            },
            center.addObserver(
                forName: UIApplication.didEnterBackgroundNotification,
                object: nil,
                queue: .main,
            ) { [weak self] _ in
                self?.stopForSystemEvent(.enteredBackground, message: "应用进入后台，已停止录音。")
            },
        ]
        refreshRoute()
        reloadRecordings()
    }

    func refreshRoute() {
        guard !isRecording else { return }
        do {
            try configureForRecording()
            availableInputs = audioSession.availableInputs?.map { route(from: $0) } ?? []
            let preferred = try AudioRoutePolicy.selectPreferredBluetoothMicrophone(from: availableInputs)
            activeInput = nil
            statusMessage = "已找到蓝牙麦克风 \(preferred.name)。点击开始录音后将复核实际输入。"
        } catch {
            activeInput = nil
            statusMessage = message(for: error)
        }
    }

    func startRecording() {
        guard !isPreparing, !isRecording else { return }
        apply(.startRequested)
        switch audioSession.recordPermission {
        case .granted:
            activateAndStartRecording()
        case .denied:
            apply(.failed("麦克风权限未授予。请到 iPhone 设置中允许本应用使用麦克风。"))
        case .undetermined:
            audioSession.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.activateAndStartRecording()
                    } else {
                        self.apply(.failed("麦克风权限未授予。请到 iPhone 设置中允许本应用使用麦克风。"))
                    }
                }
            }
        @unknown default:
            apply(.failed("无法确定麦克风权限状态。"))
        }
    }

    func stopRecording() {
        guard apply(.stopRequested) else { return }
        finishRecording(message: "录音已保存到本机。")
    }

    func play(_ recording: SavedRecording) {
        if playingRecordingID == recording.id {
            stopPlayback()
            return
        }
        stopPlayback()
        do {
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
            let player = try AVAudioPlayer(contentsOf: recording.url)
            player.delegate = self
            player.prepareToPlay()
            guard player.play() else {
                throw NSError(domain: "AIGlassesMicProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "无法开始回放。"])
            }
            self.player = player
            playingRecordingID = recording.id
            statusMessage = "正在回放 \(recording.url.lastPathComponent)。"
        } catch {
            statusMessage = "回放失败：\(message(for: error))"
        }
    }

    func delete(_ recording: SavedRecording) {
        if playingRecordingID == recording.id {
            stopPlayback()
        }
        do {
            try FileManager.default.removeItem(at: recording.url)
            reloadRecordings()
            statusMessage = "已删除录音。"
        } catch {
            statusMessage = "删除失败：\(message(for: error))"
        }
    }

    func audioPlayerDidFinishPlaying(_: AVAudioPlayer, successfully _: Bool) {
        stopPlayback()
    }

    private func activateAndStartRecording() {
        do {
            try configureForRecording()
            let inputs = audioSession.availableInputs?.map { route(from: $0) } ?? []
            availableInputs = inputs
            let expectedRoute = try AudioRoutePolicy.selectPreferredBluetoothMicrophone(from: inputs)
            let preferredRoute = try AudioRoutePolicy.requirePreferredInputAvailable(expectedRoute, in: inputs)
            guard let preferredPort = audioSession.availableInputs?.first(where: { $0.uid == preferredRoute.id }) else {
                throw AudioRouteError.preferredInputUnavailable(preferredRoute.name)
            }
            try audioSession.setPreferredInput(preferredPort)
            try audioSession.setActive(true)

            let actualRoute = audioSession.currentRoute.inputs.first.map { route(from: $0) }
            activeInput = try AudioRoutePolicy.verifyActiveInput(expected: preferredRoute, actual: actualRoute)
            let recorder = try AVAudioRecorder(url: nextRecordingURL(), settings: recordingSettings())
            recorder.isMeteringEnabled = true
            recorder.prepareToRecord()
            guard recorder.record() else {
                throw NSError(domain: "AIGlassesMicProbe", code: 2, userInfo: [NSLocalizedDescriptionKey: "iOS 未能开始写入录音文件。"])
            }
            self.recorder = recorder
            recordingStartedAt = Date()
            apply(.routeVerified)
            startMeters()
            statusMessage = "正在使用 \(preferredRoute.name) 录音。"
        } catch {
            activeInput = nil
            apply(.failed(message(for: error)))
            deactivateRecordingSession()
        }
    }

    private func configureForRecording() throws {
        try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetooth])
    }

    private func recordingSettings() -> [String: Any] {
        let sessionSampleRate = audioSession.sampleRate
        let sampleRate = sessionSampleRate > 0 ? sessionSampleRate : 44_100
        return [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 96_000,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]
    }

    private func route(from port: AVAudioSessionPortDescription) -> AudioInputRoute {
        let kind: AudioInputKind
        switch port.portType {
        case .bluetoothHFP:
            kind = .bluetoothHFP
        case .bluetoothA2DP:
            kind = .bluetoothA2DP
        case .builtInMic:
            kind = .builtInMic
        case .headsetMic:
            kind = .wiredMic
        case .usbAudio:
            kind = .usbAudio
        default:
            kind = .other
        }
        return AudioInputRoute(id: port.uid, name: port.portName, kind: kind)
    }

    private func handleRouteChange() {
        let wasRecording = isRecording
        if apply(.routeChanged) {
            finishRecording(message: "输入设备已变化，已停止录音。请重新检测后再开始。")
        }
        refreshRoute()
        if wasRecording {
            statusMessage = "输入设备已变化，已停止录音。请重新检测后再开始。"
        }
    }

    private func handleInterruption(_ notification: Notification) {
        guard let rawType = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              AVAudioSession.InterruptionType(rawValue: rawType) == .began else {
            return
        }
        stopForSystemEvent(.interrupted, message: "音频会话被系统中断，已停止录音。")
    }

    private func stopForSystemEvent(_ event: RecordingEvent, message: String) {
        if apply(event) {
            finishRecording(message: message)
        } else if recordingState == .idle {
            statusMessage = message
        }
    }

    private func finishRecording(message: String) {
        recorder?.stop()
        recorder = nil
        meterTimer?.invalidate()
        elapsedTimer?.invalidate()
        meterTimer = nil
        elapsedTimer = nil
        level = 0
        elapsedSeconds = 0
        recordingStartedAt = nil
        activeInput = nil
        deactivateRecordingSession()
        reloadRecordings()
        statusMessage = message
    }

    private func deactivateRecordingSession() {
        try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startMeters() {
        meterTimer?.invalidate()
        elapsedTimer?.invalidate()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self, let recorder = self.recorder else { return }
            recorder.updateMeters()
            let decibels = Double(recorder.averagePower(forChannel: 0))
            self.level = min(1, max(0, (decibels + 60) / 60))
        }
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            guard let self, let recordingStartedAt = self.recordingStartedAt else { return }
            self.elapsedSeconds = Date().timeIntervalSince(recordingStartedAt)
        }
    }

    private func reloadRecordings() {
        let directory = recordingsDirectory()
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.creationDateKey],
            options: [.skipsHiddenFiles],
        )) ?? []
        recordings = urls
            .filter { $0.pathExtension.lowercased() == "m4a" }
            .compactMap { url in
                let values = try? url.resourceValues(forKeys: [.creationDateKey])
                let duration = (try? AVAudioPlayer(contentsOf: url).duration) ?? 0
                return SavedRecording(url: url, createdAt: values?.creationDate ?? .distantPast, duration: duration)
            }
            .sorted { $0.createdAt > $1.createdAt }
    }

    private func nextRecordingURL() throws -> URL {
        let timestamp = Int(Date().timeIntervalSince1970 * 1_000)
        return recordingsDirectory().appendingPathComponent("mic-pro-\(timestamp).m4a")
    }

    private func recordingsDirectory() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let directory = documents.appendingPathComponent("Recordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func stopPlayback() {
        player?.stop()
        player = nil
        playingRecordingID = nil
    }

    private func apply(_ event: RecordingEvent) -> Bool {
        let shouldStop = lifecycle.apply(event)
        recordingState = lifecycle.state
        if case let .failed(message) = recordingState {
            statusMessage = message
        }
        return shouldStop
    }

    private func message(for error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}
