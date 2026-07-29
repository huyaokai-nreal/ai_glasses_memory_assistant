import Foundation

/// Deliberately blocks persistence until a bundled CPython POC proves SQLite, audit, and memory writes on device.
final class EmbeddedPythonRuntime {
    enum RuntimeError: LocalizedError {
        case missingConfiguration, binaryNotBundled
        var errorDescription: String? { self == .missingConfiguration ? "请先在设置中配置联网 LLM" : "CPython POC 尚未打包；收音不会写入记忆或 Timeline" }
    }

    func start(settings: RuntimeSettings) throws -> URL {
        guard !settings.apiKey.isEmpty else { throw RuntimeError.missingConfiguration }
        throw RuntimeError.binaryNotBundled
    }
}
