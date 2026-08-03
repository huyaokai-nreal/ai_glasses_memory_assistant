import CryptoKit
import Foundation

enum IOSModelPackError: LocalizedError, Equatable {
    case invalidManifest(String)
    case invalidFile(String)
    case hashMismatch(String)

    var errorDescription: String? {
        switch self {
        case let .invalidManifest(message): return message
        case let .invalidFile(name): return "模型文件无效：\(name)"
        case let .hashMismatch(name): return "模型文件 SHA-256 校验失败：\(name)"
        }
    }
}

/// Mirrors Android's verified model_pack.v1 so both platforms use the same ONNX files.
struct IOSModelPackManifest: Codable, Equatable {
    static let schema = "model_pack.v1"
    static let sherpaOnnxVersion = "1.13.4"
    static let requiredComponents: Set<String> = ["vad", "kws", "online_asr", "ambient_asr", "speaker"]

    struct Component: Codable, Equatable {
        let engine: String
        let roles: [String: String]
        let options: [String: JSONValue]
    }

    struct File: Codable, Equatable {
        let path: String
        let sizeBytes: Int
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case path, sha256
            case sizeBytes = "size_bytes"
        }
    }

    let schema: String
    let version: String
    let sherpaOnnxVersion: String
    let installMode: String?
    let files: [File]
    let components: [String: Component]

    enum CodingKeys: String, CodingKey {
        case schema, version, files, components
        case sherpaOnnxVersion = "sherpa_onnx_version"
        case installMode = "install_mode"
    }

    func validate() throws {
        guard schema == Self.schema else { throw IOSModelPackError.invalidManifest("模型清单 schema 必须是 \(Self.schema)") }
        guard !version.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw IOSModelPackError.invalidManifest("模型清单缺少版本号") }
        guard sherpaOnnxVersion == Self.sherpaOnnxVersion else {
            throw IOSModelPackError.invalidManifest("模型要求 sherpa-onnx \(sherpaOnnxVersion)，App 仅提供 \(Self.sherpaOnnxVersion)")
        }
        guard !files.isEmpty else { throw IOSModelPackError.invalidManifest("模型清单没有文件") }
        guard Self.requiredComponents.isSubset(of: Set(components.keys)) else {
            throw IOSModelPackError.invalidManifest("模型清单缺少组件：\(Self.requiredComponents.subtracting(components.keys).sorted().joined(separator: ", "))")
        }
        let declaredPaths = Set(files.map(\.path))
        guard declaredPaths.count == files.count else { throw IOSModelPackError.invalidManifest("模型清单包含重复文件") }
        for file in files {
            guard Self.isSafeRelativePath(file.path), file.sizeBytes > 0,
                  file.sha256.range(of: "^[a-fA-F0-9]{64}$", options: .regularExpression) != nil else {
                throw IOSModelPackError.invalidFile(file.path)
            }
        }
        for (name, component) in components {
            guard !component.engine.isEmpty, !component.roles.isEmpty else {
                throw IOSModelPackError.invalidManifest("模型组件无效：\(name)")
            }
            guard component.roles.values.allSatisfy(declaredPaths.contains) else {
                throw IOSModelPackError.invalidManifest("模型组件 \(name) 引用了未声明文件")
            }
        }
    }

    private static func isSafeRelativePath(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return !path.hasPrefix("/") && !parts.isEmpty && parts.allSatisfy {
            !$0.isEmpty && $0 != "." && $0 != ".." && $0.allSatisfy { $0.isLetter || $0.isNumber || "._-".contains($0) }
        }
    }
}

enum JSONValue: Codable, Equatable {
    case string(String), number(Double), bool(Bool), object([String: JSONValue]), array([JSONValue]), null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { self = .array(try container.decode([JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .bool(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

struct BundledModelPackStatus: Equatable {
    let version: String?
    let message: String
    let ready: Bool
}

enum BundledModelPackValidator {
    private static let lock = NSLock()
    private static var cache: [String: BundledModelPackStatus] = [:]

    static func validate(bundle: Bundle = .main, forceRefresh: Bool = false) -> BundledModelPackStatus {
        let key = bundle.bundlePath
        lock.lock()
        if !forceRefresh, let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let result = validateUncached(bundle: bundle)
        lock.lock()
        cache[key] = result
        lock.unlock()
        return result
    }

    private static func validateUncached(bundle: Bundle) -> BundledModelPackStatus {
        guard let manifestURL = bundle.url(forResource: "manifest", withExtension: "json", subdirectory: "Models") else {
            return BundledModelPackStatus(version: nil, message: "未打包本地模型", ready: false)
        }
        do {
            let manifest = try JSONDecoder().decode(IOSModelPackManifest.self, from: Data(contentsOf: manifestURL))
            try manifest.validate()
            let root = manifestURL.deletingLastPathComponent()
            for file in manifest.files {
                let url = root.appendingPathComponent(file.path)
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                guard attributes[.size] as? Int == file.sizeBytes else { throw IOSModelPackError.invalidFile(file.path) }
                let digest = try sha256(url)
                guard digest.caseInsensitiveCompare(file.sha256) == .orderedSame else { throw IOSModelPackError.hashMismatch(file.path) }
            }
            return BundledModelPackStatus(version: manifest.version, message: "本地模型 \(manifest.version) 已校验", ready: true)
        } catch {
            return BundledModelPackStatus(version: nil, message: error.localizedDescription, ready: false)
        }
    }

    private static func sha256(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
