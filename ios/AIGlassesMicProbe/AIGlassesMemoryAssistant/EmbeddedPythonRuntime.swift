import Foundation

/// Deliberately blocks persistence until a bundled CPython POC proves SQLite, audit, and memory writes on device.
struct EmbeddedRuntimeEndpoint: Equatable {
    let baseURL: URL
    let localToken: String
    let ownerID: String
}

final class EmbeddedPythonRuntime {
    enum RuntimeError: LocalizedError {
        case missingConfiguration, resourceMissing, invalidEndpoint
        var errorDescription: String? {
            switch self {
            case .missingConfiguration: return "请先在设置中配置联网 LLM"
            case .resourceMissing: return "CPython 本地运行时未打包"
            case .invalidEndpoint: return "CPython 本地运行时返回了无效地址"
            }
        }
    }

    func start(settings: RuntimeSettings) throws -> EmbeddedRuntimeEndpoint {
        guard !settings.apiKey.isEmpty else { throw RuntimeError.missingConfiguration }
        guard let pythonHome = Bundle.main.resourceURL?.appendingPathComponent("PythonRuntime", isDirectory: true),
              let moduleRoot = pythonHome.appendingPathComponent("python", isDirectory: true) as URL?,
              FileManager.default.fileExists(atPath: pythonHome.path),
              FileManager.default.fileExists(atPath: moduleRoot.path) else {
            throw RuntimeError.resourceMissing
        }
        let appHome = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("AIGlassesMemoryAssistant", isDirectory: true)
        try FileManager.default.createDirectory(at: appHome, withIntermediateDirectories: true)
        let staticDirectory = Bundle.main.resourceURL?.appendingPathComponent("Web/static", isDirectory: true)
        let config: [String: String] = [
            "app_home": appHome.path,
            "static_dir": staticDirectory?.path ?? "",
            "provider": settings.provider,
            "model": settings.model,
            "base_url": settings.baseURL,
            "api_key": settings.apiKey,
            // The Python service must use this iPhone's stable Keychain identity,
            // never an Android owner ID or a process-local placeholder.
            "owner_id": try RuntimeSettingsStore().ownerID(),
            "platform": "ios",
        ]
        let payload = try JSONSerialization.data(withJSONObject: config)
        let raw: String
        do {
            raw = try PythonRuntimeBridge.start(
                withPythonHome: pythonHome,
                moduleRoot: moduleRoot,
                configJSON: String(decoding: payload, as: UTF8.self)
            )
        } catch {
            NSLog("[EmbeddedPythonRuntime] Python start error: %@", error.localizedDescription)
            throw error
        }
        guard let object = try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any],
              let value = object["base_url"] as? String,
              let endpoint = URL(string: value),
              let token = object["local_token"] as? String,
              let ownerID = object["owner_id"] as? String,
              endpoint.scheme == "http" else { throw RuntimeError.invalidEndpoint }
        return EmbeddedRuntimeEndpoint(baseURL: endpoint, localToken: token, ownerID: ownerID)
    }

    func stop() { PythonRuntimeBridge.stop() }

    func call(_ function: String, arguments: [String]) throws -> String {
        return try PythonRuntimeBridge.callFunction(function, arguments: arguments)
    }
}
