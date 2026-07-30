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

    init(audio: AssistantAudioController) {
        self.audio = audio
        super.init()
        locationManager.delegate = self
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
        case "startAmbient", "startSpeakerEnrollment":
            replyHandler(nil, BridgeError.pipelineUnavailable.localizedDescription)
        case "stopAmbient", "cancelSpeakerEnrollment":
            audio.stop()
            replyHandler(encoded(audio.webStatus()), nil)
        case "speak":
            let payload = request["payload"] as? [String: Any] ?? [:]
            audio.speak(String(payload["text"] as? String ?? ""))
            replyHandler(encoded(audio.webStatus()), nil)
        case "stopSpeaking":
            audio.stop(reason: nil)
            replyHandler(encoded(audio.webStatus()), nil)
        case "openSettings":
            openSettings?()
            replyHandler(encoded(["opened": true]), nil)
        case "location":
            requestLocation(replyHandler)
        default:
            replyHandler(nil, BridgeError.unsupportedMethod(method).localizedDescription)
        }
    }

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
