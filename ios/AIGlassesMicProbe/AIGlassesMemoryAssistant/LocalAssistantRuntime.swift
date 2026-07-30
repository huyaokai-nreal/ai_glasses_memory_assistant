import Foundation

@MainActor
final class LocalAssistantRuntime: ObservableObject {
    @Published private(set) var status = "本机 POC 尚未初始化"
    @Published private(set) var timelineCount = 0
    @Published private(set) var memoryCount = 0

    private var store: LocalAssistantStore?

    func bootstrap() {
        do {
            let root = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                .appendingPathComponent("AIGlassesMemoryAssistant", isDirectory: true)
            store = try LocalAssistantStore(directory: root)
            try refreshCounts()
            status = "本机 SQLite POC 已就绪"
        } catch {
            status = error.localizedDescription
        }
    }

    func verifyFinalGate() {
        guard let store else { bootstrap(); return }
        do {
            let result = try store.ingest(LocalAudioEvent(
                kind: .final,
                transcript: "请记住我喜欢黑咖啡",
                speakerState: .owner,
                overlapState: .none,
                privacyApproved: true
            ))
            timelineCount = result.timelineCount
            memoryCount = result.memoryCount
            status = result.accepted ? "final 已通过门控并写入 Timeline 和记忆" : "final 被拒绝：\(result.reason)"
        } catch {
            status = error.localizedDescription
        }
    }

    func verifyPartialIsolation() {
        guard let store else { bootstrap(); return }
        do {
            let before = try store.snapshot()
            let result = try store.ingest(LocalAudioEvent(
                kind: .partial,
                transcript: "这段 partial 不得落库",
                speakerState: .owner,
                overlapState: .none,
                privacyApproved: true
            ))
            let after = try store.snapshot()
            timelineCount = after.timeline.count
            memoryCount = after.memories.count
            status = !result.accepted && before == after ? "partial 已验证：仅 UI/Debug，不写 Timeline、记忆或审计" : "partial 隔离验证失败"
        } catch {
            status = error.localizedDescription
        }
    }

    func exportDiagnostic(passphrase: String) -> URL? {
        guard let store else { status = "请先初始化本机 POC"; return nil }
        do {
            let encrypted = try DiagnosticBundleExporter.encrypt(snapshot: store.snapshot(), passphrase: passphrase)
            let file = FileManager.default.temporaryDirectory.appendingPathComponent("ai-glasses-diagnostic-\(UUID().uuidString).aigdiag")
            try encrypted.write(to: file, options: .atomic)
            status = "已生成脱敏加密诊断包"
            return file
        } catch {
            status = error.localizedDescription
            return nil
        }
    }

    private func refreshCounts() throws {
        guard let store else { return }
        let snapshot = try store.snapshot()
        timelineCount = snapshot.timeline.count
        memoryCount = snapshot.memories.count
    }
}
