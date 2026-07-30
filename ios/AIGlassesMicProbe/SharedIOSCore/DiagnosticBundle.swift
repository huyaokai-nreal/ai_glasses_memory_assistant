import CommonCrypto
import CryptoKit
import Foundation
import Security

enum DiagnosticBundleError: LocalizedError {
    case invalidPassword
    case keyDerivationFailed

    var errorDescription: String? {
        switch self {
        case .invalidPassword: return "导出密码至少需要 8 个字符"
        case .keyDerivationFailed: return "无法生成诊断包加密密钥"
        }
    }
}

enum DiagnosticBundleExporter {
    static let magic = Data("AIGDIAG1".utf8)
    static let iterations: UInt32 = 210_000

    /// Matches the Android AIGDIAG1 header: magic, big-endian iterations, salt size/value, IV size/value, ciphertext and GCM tag.
    static func encrypt(snapshot: LocalAssistantSnapshot, passphrase: String) throws -> Data {
        guard passphrase.count >= 8 else { throw DiagnosticBundleError.invalidPassword }
        let json = try JSONEncoder.diagnostic.encode(snapshot)
        let zip = ZipArchive.stored(entries: ["diagnostics.json": json])
        let salt = random(count: 16)
        let nonceData = random(count: 12)
        let key = try deriveKey(passphrase: Array(passphrase.utf8), salt: salt)
        let nonce = try AES.GCM.Nonce(data: nonceData)
        let sealed = try AES.GCM.seal(zip, using: SymmetricKey(data: key), nonce: nonce)

        var output = Data()
        output.append(magic)
        output.appendBigEndian(iterations)
        output.append(UInt8(salt.count))
        output.append(salt)
        output.append(UInt8(nonceData.count))
        output.append(nonceData)
        output.append(sealed.ciphertext)
        output.append(sealed.tag)
        return output
    }

    private static func random(count: Int) -> Data {
        var bytes = [UInt8](repeating: 0, count: count)
        _ = SecRandomCopyBytes(kSecRandomDefault, count, &bytes)
        return Data(bytes)
    }

    private static func deriveKey(passphrase: [UInt8], salt: Data) throws -> Data {
        var key = [UInt8](repeating: 0, count: 32)
        let status = salt.withUnsafeBytes { saltBuffer in
            CCKeyDerivationPBKDF(
                CCPBKDFAlgorithm(kCCPBKDF2),
                passphrase.map { Int8(bitPattern: $0) },
                passphrase.count,
                saltBuffer.bindMemory(to: UInt8.self).baseAddress,
                salt.count,
                CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256),
                iterations,
                &key,
                key.count
            )
        }
        guard status == kCCSuccess else { throw DiagnosticBundleError.keyDerivationFailed }
        return Data(key)
    }
}

private extension JSONEncoder {
    static var diagnostic: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

private extension Data {
    mutating func appendBigEndian(_ value: UInt32) {
        var value = value.bigEndian
        append(Data(bytes: &value, count: MemoryLayout<UInt32>.size))
    }
}

/// Minimal stored ZIP writer keeps the exported diagnostic interoperable without a third-party archive dependency.
private enum ZipArchive {
    static func stored(entries: [String: Data]) -> Data {
        var archive = Data()
        var central = Data()
        for (name, payload) in entries.sorted(by: { $0.key < $1.key }) {
            let nameData = Data(name.utf8)
            let offset = UInt32(archive.count)
            let crc = CRC32.checksum(payload)
            archive.appendUInt32(0x04034b50)
            archive.appendUInt16(20)
            archive.appendUInt16(0)
            archive.appendUInt16(0)
            archive.appendUInt16(0)
            archive.appendUInt16(0)
            archive.appendUInt32(crc)
            archive.appendUInt32(UInt32(payload.count))
            archive.appendUInt32(UInt32(payload.count))
            archive.appendUInt16(UInt16(nameData.count))
            archive.appendUInt16(0)
            archive.append(nameData)
            archive.append(payload)

            central.appendUInt32(0x02014b50)
            central.appendUInt16(20)
            central.appendUInt16(20)
            central.appendUInt16(0)
            central.appendUInt16(0)
            central.appendUInt16(0)
            central.appendUInt16(0)
            central.appendUInt32(crc)
            central.appendUInt32(UInt32(payload.count))
            central.appendUInt32(UInt32(payload.count))
            central.appendUInt16(UInt16(nameData.count))
            central.appendUInt16(0)
            central.appendUInt16(0)
            central.appendUInt16(0)
            central.appendUInt16(0)
            central.appendUInt32(0)
            central.appendUInt32(offset)
            central.append(nameData)
        }
        let centralOffset = UInt32(archive.count)
        archive.append(central)
        archive.appendUInt32(0x06054b50)
        archive.appendUInt16(0)
        archive.appendUInt16(0)
        archive.appendUInt16(UInt16(entries.count))
        archive.appendUInt16(UInt16(entries.count))
        archive.appendUInt32(UInt32(central.count))
        archive.appendUInt32(centralOffset)
        archive.appendUInt16(0)
        return archive
    }
}

private enum CRC32 {
    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xffffffff
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 { crc = (crc & 1) == 0 ? crc >> 1 : (crc >> 1) ^ 0xedb88320 }
        }
        return crc ^ 0xffffffff
    }
}

private extension Data {
    mutating func appendUInt16(_ value: UInt16) {
        var value = value.littleEndian
        append(Data(bytes: &value, count: MemoryLayout<UInt16>.size))
    }

    mutating func appendUInt32(_ value: UInt32) {
        var value = value.littleEndian
        append(Data(bytes: &value, count: MemoryLayout<UInt32>.size))
    }
}
