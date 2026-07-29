import Foundation
import Security

struct RuntimeSettings: Equatable {
    var provider = "deepseek"
    var model = "deepseek-v4-flash"
    var baseURL = "https://api.deepseek.com"
    var apiKey = ""
}

final class RuntimeSettingsStore {
    private let service = "com.aiglasses.memoryassistant.runtime"
    private let account = "llm-api-key"
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    func load() throws -> RuntimeSettings {
        var value = RuntimeSettings()
        value.provider = defaults.string(forKey: "runtime.provider") ?? value.provider
        value.model = defaults.string(forKey: "runtime.model") ?? value.model
        value.baseURL = defaults.string(forKey: "runtime.baseURL") ?? value.baseURL
        value.apiKey = try readKey() ?? ""
        return value
    }

    func save(_ value: RuntimeSettings) throws {
        guard value.baseURL.hasPrefix("https://") else { throw NSError(domain: "RuntimeSettings", code: 1, userInfo: [NSLocalizedDescriptionKey: "Base URL 必须使用 HTTPS"]) }
        defaults.set(value.provider, forKey: "runtime.provider")
        defaults.set(value.model, forKey: "runtime.model")
        defaults.set(value.baseURL, forKey: "runtime.baseURL")
        let query: [CFString: Any] = [kSecClass: kSecClassGenericPassword, kSecAttrService: service, kSecAttrAccount: account]
        let attributes: [CFString: Any] = [kSecValueData: Data(value.apiKey.utf8), kSecAttrAccessible: kSecAttrAccessibleWhenUnlockedThisDeviceOnly]
        let update = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if update == errSecItemNotFound {
            var creation = query
            attributes.forEach { creation[$0.key] = $0.value }
            guard SecItemAdd(creation as CFDictionary, nil) == errSecSuccess else { throw NSError(domain: "RuntimeSettings", code: 2) }
        } else if update != errSecSuccess { throw NSError(domain: "RuntimeSettings", code: Int(update)) }
    }

    private func readKey() throws -> String? {
        let query: [CFString: Any] = [kSecClass: kSecClassGenericPassword, kSecAttrService: service, kSecAttrAccount: account, kSecReturnData: true]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = item as? Data else { throw NSError(domain: "RuntimeSettings", code: Int(status)) }
        return String(data: data, encoding: .utf8)
    }
}
