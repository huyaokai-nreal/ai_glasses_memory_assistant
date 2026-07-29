import CryptoKit
import Foundation

enum IOSModelPackError: LocalizedError, Equatable {
    case invalidManifest(String)
    case invalidDownloadURL(String)
    case invalidFile(String)
    case hashMismatch(String)

    var errorDescription: String? {
        switch self {
        case let .invalidManifest(message): return message
        case let .invalidDownloadURL(value): return "模型下载地址必须使用 HTTPS：\(value)"
        case let .invalidFile(name): return "模型文件无效：\(name)"
        case let .hashMismatch(name): return "模型文件 SHA-256 校验失败：\(name)"
        }
    }
}

struct IOSModelPackManifest: Codable, Equatable {
    static let schema = "ios_model_pack.v1"
    static let requiredComponents: Set<String> = ["vad", "kws", "streaming_asr", "ambient_asr", "speaker"]

    struct Component: Codable, Equatable {
        let files: [File]
        let options: [String: String]
    }

    struct File: Codable, Equatable {
        let name: String
        let url: String
        let sizeBytes: Int
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case name, url, sha256
            case sizeBytes = "size_bytes"
        }
    }

    let schema: String
    let version: String
    let components: [String: Component]

    func validate() throws {
        guard schema == Self.schema else { throw IOSModelPackError.invalidManifest("模型清单 schema 必须是 \(Self.schema)") }
        guard !version.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw IOSModelPackError.invalidManifest("模型清单缺少版本号") }
        guard Self.requiredComponents.isSubset(of: Set(components.keys)) else {
            throw IOSModelPackError.invalidManifest("模型清单缺少组件：\(Self.requiredComponents.subtracting(components.keys).sorted().joined(separator: ", "))")
        }
        for component in components.values {
            guard !component.files.isEmpty else { throw IOSModelPackError.invalidManifest("模型组件不能没有文件") }
            for file in component.files {
                guard !file.name.isEmpty, file.name == URL(fileURLWithPath: file.name).lastPathComponent,
                      file.sizeBytes > 0,
                      file.sha256.range(of: "^[a-fA-F0-9]{64}$", options: .regularExpression) != nil else {
                    throw IOSModelPackError.invalidFile(file.name)
                }
                guard let url = URL(string: file.url), url.scheme?.lowercased() == "https" else {
                    throw IOSModelPackError.invalidDownloadURL(file.url)
                }
            }
        }
    }
}

final class IOSModelPackInstaller {
    private let fileManager: FileManager
    private let session: URLSession

    init(fileManager: FileManager = .default, session: URLSession = .shared) {
        self.fileManager = fileManager
        self.session = session
    }

    func fetchManifest(from url: URL) async throws -> IOSModelPackManifest {
        guard url.scheme?.lowercased() == "https" else { throw IOSModelPackError.invalidDownloadURL(url.absoluteString) }
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw IOSModelPackError.invalidManifest("无法下载模型清单")
        }
        let manifest = try JSONDecoder().decode(IOSModelPackManifest.self, from: data)
        try manifest.validate()
        return manifest
    }

    func install(_ manifest: IOSModelPackManifest, into root: URL) async throws -> URL {
        try manifest.validate()
        let staging = root.appendingPathComponent(".staging-\(UUID().uuidString)", isDirectory: true)
        let final = root.appendingPathComponent(manifest.version, isDirectory: true)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }

        for (componentName, component) in manifest.components {
            let componentDirectory = staging.appendingPathComponent(componentName, isDirectory: true)
            try fileManager.createDirectory(at: componentDirectory, withIntermediateDirectories: true)
            for file in component.files {
                guard let url = URL(string: file.url), url.scheme?.lowercased() == "https" else { throw IOSModelPackError.invalidDownloadURL(file.url) }
                let (data, response) = try await session.data(from: url)
                guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), data.count == file.sizeBytes else {
                    throw IOSModelPackError.invalidFile(file.name)
                }
                let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                guard digest.caseInsensitiveCompare(file.sha256) == .orderedSame else { throw IOSModelPackError.hashMismatch(file.name) }
                try data.write(to: componentDirectory.appendingPathComponent(file.name), options: .atomic)
            }
        }
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        try? fileManager.removeItem(at: final)
        try fileManager.moveItem(at: staging, to: final)
        return final
    }
}
