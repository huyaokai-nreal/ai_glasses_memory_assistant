import CoreLocation
import Foundation
import WebKit

@MainActor
final class IOSNativeBridge: NSObject, WKScriptMessageHandlerWithReply {
    var openSettings: (() -> Void)?

    private let audio: AssistantAudioController
    private let settingsStore = RuntimeSettingsStore()
    private let locationManager = CLLocationManager()
    private var locationReply: (@MainActor @Sendable (String) -> Void)?
    private var pipeline: ModelPipeline?
    private var pendingReplyEvents: [(String, String)] = []
    private var consumeCompletion: (([String: Any]) -> Void)?

    init(audio: AssistantAudioController) {
        self.audio = audio
        super.init()
        locationManager.delegate = self
        // When TTS finishes, notify the pipeline to transition to waiting_query.
        audio.onTtsFinished = { [weak self] in
            self?.pipeline?.acknowledgementFinished()
        }
    }

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping @MainActor @Sendable (Any?, String?) -> Void
    ) {
        guard message.name == "aiGlassesNative",
              let request = message.body as? [String: Any],
              let method = request["method"] as? String else {
            replyHandler(nil, BridgeError.invalidRequest.localizedDescription)
            return
        }
        switch method {
        case "platform":
            replyHandler(encoded(["platform": "ios"]), nil)
        case "ownerId":
            replyHandler(result { ["owner_id": try self.settingsStore.ownerID()] }, nil)
        case "audioStatus", "audioUiStatus":
            replyHandler(encoded(audio.webStatus()), nil)
        case "startAmbient":
            startAmbient(replyHandler)
        case "startSpeakerEnrollment":
            startSpeakerEnrollment(replyHandler)
        case "stopAmbient", "cancelSpeakerEnrollment":
            stopAmbient()
            replyHandler(encoded(audio.webStatus()), nil)
        case "speak":
            let payload = request["payload"] as? [String: Any] ?? [:]
            audio.speak(String(payload["text"] as? String ?? ""))
            replyHandler(encoded(audio.webStatus()), nil)
        case "stopSpeaking":
            audio.stop(reason: nil)
            replyHandler(encoded(audio.webStatus()), nil)
        case "consumeCompletedReplies":
            consumeCompletedReplies(replyHandler)
        case "openSettings":
            openSettings?()
            replyHandler(encoded(["opened": true]), nil)
        case "location":
            requestLocation(replyHandler)
        default:
            replyHandler(nil, BridgeError.unsupportedMethod(method).localizedDescription)
        }
    }

    // MARK: - Pipeline Lifecycle

    private func startAmbient(_ reply: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        // If already running, report status.
        if pipeline != nil {
            reply(encoded(audio.webStatus()), nil)
            return
        }

        // Validate model pack on main thread (fast).
        guard let packDir = modelPackDirectory() else {
            reply(nil, "未找到本地模型包")
            return
        }
        let manifestURL = packDir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path),
              let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(IOSModelPackManifest.self, from: data),
              (try? manifest.validate()) != nil else {
            reply(nil, "本地模型包校验失败")
            return
        }

        let ownerId = (try? settingsStore.ownerID()) ?? "unknown"
        let captureId = "ios-\(UUID().uuidString)"

        // Reply immediately to prevent timeout, then load on background thread.
        reply(encoded(["status": "loading", "message": "正在加载本地语音模型…"]), nil)

        // Load 5 sherpa-onnx models (~290 MB) on a background thread to avoid
        // blocking the main thread and triggering the iOS watchdog (hang > 10 s).
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let p = try ModelPipeline(
                    ownerId: ownerId,
                    captureId: captureId,
                    packDir: packDir,
                    manifest: manifest,
                    onWakeAcknowledgement: { [weak self] in
                        Task { @MainActor in self?.audio.speak("我在，请说") }
                    },
                    onPartial: { _ in },
                    onEnrollmentProgress: { _, _, _, _, _ in },
                    onAssistantQuery: { [weak self] eventId, text, event, privateEvent in
                        Task { @MainActor in self?.pendingReplyEvents.append((eventId, text)) }
                    },
                    onAmbientEvent: { _, _ in },
                    onFailure: { [weak self] error in
                        Task { @MainActor in
                            self?.audio.stop(reason: error.localizedDescription)
                            self?.pipeline = nil
                        }
                    }
                )

                // Switch back to main thread for audio session operations.
                Task { @MainActor in
                    self.pipeline = p
                    self.audio.pipeline = p
                    self.audio.start()
                }
            } catch {
                Task { @MainActor in
                    self.audio.stop(reason: "无法启动本地模型管线：\(error.localizedDescription)")
                }
            }
        }
    }

    private func startSpeakerEnrollment(_ reply: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        guard let p = pipeline else {
            reply(nil, "请先启动 ambient 收音")
            return
        }
        let sessionId = "ios-enroll-\(UUID().uuidString)"
        // Ensure audio is running.
        if audio.state != .listening {
            audio.pipeline = p
            audio.start()
        }
        p.startEnrollment(sessionId: sessionId)
        reply(encoded([
            "session_id": sessionId,
            "status": "recording",
        ]), nil)
    }

    private func stopAmbient() {
        pipeline?.close()
        pipeline = nil
        audio.pipeline = nil
        audio.stop()
        pendingReplyEvents.removeAll()
    }

    private func consumeCompletedReplies(_ reply: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        guard !pendingReplyEvents.isEmpty else {
            reply(encoded(["events": []]), nil)
            return
        }
        let events = pendingReplyEvents.map { (eventId, text) in
            ["event_id": eventId, "text": text] as [String: Any]
        }
        pendingReplyEvents.removeAll()
        reply(encoded(["events": events]), nil)
    }

    // MARK: - Model Pack Path

    private func modelPackDirectory() -> URL? {
        // Models are copied to the app bundle under "Models/" during build.
        guard let modelsURL = Bundle.main.resourceURL?.appendingPathComponent("Models") else {
            return nil
        }
        return modelsURL
    }

    // MARK: - Location

    private func requestLocation(_ reply: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        locationReply = { raw in reply(raw, nil) }
        switch locationManager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse: locationManager.requestLocation()
        case .notDetermined: locationManager.requestWhenInUseAuthorization()
        default: finishLocation(["status": "denied"])
        }
    }

    private func finishLocation(_ payload: [String: Any]) {
        let reply = locationReply
        locationReply = nil
        reply?(encoded(payload))
    }

    private func result(_ body: () throws -> [String: Any]) -> String {
        do { return encoded(try body()) }
        catch { return encoded(["error": error.localizedDescription]) }
    }

    private func encoded(_ payload: [String: Any]) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: payload, options: [])) ?? Data("{}".utf8)
        return String(decoding: data, as: UTF8.self)
    }

    private enum BridgeError: LocalizedError {
        case invalidRequest, pipelineUnavailable, unsupportedMethod(String)
        var errorDescription: String? {
            switch self {
            case .invalidRequest: return "原生桥接请求无效"
            case .pipelineUnavailable: return "iOS 本地语音模型尚未完成自检，已拒绝开始收音"
            case let .unsupportedMethod(method): return "iOS 原生桥接不支持：\(method)"
            }
        }
    }
}

extension IOSNativeBridge: @preconcurrency CLLocationManagerDelegate {
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        guard locationReply != nil else { return }
        if manager.authorizationStatus == .authorizedWhenInUse || manager.authorizationStatus == .authorizedAlways {
            manager.requestLocation()
        } else if manager.authorizationStatus == .denied || manager.authorizationStatus == .restricted {
            finishLocation(["status": "denied", "source": "ios_core_location"])
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else {
            finishLocation(["status": "unavailable", "source": "ios_core_location"])
            return
        }
        finishLocation([
            "status": "available",
            "latitude": location.coordinate.latitude,
            "longitude": location.coordinate.longitude,
            "accuracy": location.horizontalAccuracy,
            "timestamp": location.timestamp.timeIntervalSince1970,
            "source": "ios_core_location",
        ])
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        finishLocation(["status": "unavailable", "reason": error.localizedDescription, "source": "ios_core_location"])
    }
}
