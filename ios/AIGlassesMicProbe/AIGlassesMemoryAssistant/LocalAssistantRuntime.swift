import Foundation

@MainActor
final class LocalAssistantRuntime: ObservableObject {
    @Published private(set) var status = "正在准备本机运行时…"
    @Published private(set) var endpoint: EmbeddedRuntimeEndpoint?
    @Published private(set) var starting = false
    @Published private(set) var timelineCount = 0
    @Published private(set) var memoryCount = 0

    private var store: LocalAssistantStore?
    private let embedded = EmbeddedPythonRuntime()
    private let callQueue = DispatchQueue(label: "com.aiglasses.memoryassistant.runtime-calls", qos: .userInitiated)
    private var generation = 0
    private var activeSettings: RuntimeSettings?
    private(set) var captureID = ""

    func bootstrap(settings supplied: RuntimeSettings? = nil) {
        let settings: RuntimeSettings
        do { settings = try supplied ?? RuntimeSettingsStore().load() }
        catch { status = error.localizedDescription; return }
        guard !settings.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            status = "请先在高级设置中配置 HTTPS LLM 和 API 密钥"
            return
        }
        if let activeSettings, endpoint != nil, activeSettings != settings {
            reconfigure(settings: settings)
            return
        }
        if starting || endpoint != nil { return }
        generation += 1
        let launchGeneration = generation
        activeSettings = settings
        starting = true
        status = "正在启动本机 Python 服务…"
        callQueue.async { [weak self] in
            guard let self else { return }
            do {
                let endpoint = try self.embedded.start(settings: settings)
                DispatchQueue.main.async {
                    guard self.generation == launchGeneration else {
                        self.callQueue.async { self.embedded.stop() }
                        return
                    }
                    self.endpoint = endpoint
                    self.starting = false
                    self.bootstrapStore()
                    self.status = "本机聊天、记忆和审计服务已就绪"
                }
            } catch {
                DispatchQueue.main.async {
                    guard self.generation == launchGeneration else { return }
                    self.starting = false
                    self.activeSettings = nil
                    self.status = "本机 Python 服务启动失败：\(error.localizedDescription)"
                }
            }
        }
    }

    func reconfigure(settings: RuntimeSettings) {
        generation += 1
        let launchGeneration = generation
        starting = true
        endpoint = nil
        activeSettings = nil
        let oldCaptureID = captureID
        captureID = ""
        status = "正在应用新的本机运行时配置…"
        callQueue.async { [weak self] in
            guard let self else { return }
            if !oldCaptureID.isEmpty, let ownerID = try? RuntimeSettingsStore().ownerID() {
                _ = try? self.embedded.call("stop_capture", arguments: [ownerID, oldCaptureID])
            }
            self.embedded.stop()
            DispatchQueue.main.async {
                guard self.generation == launchGeneration else { return }
                self.starting = false
                self.bootstrap(settings: settings)
            }
        }
    }

    func shutdown() {
        generation += 1
        let oldCaptureID = captureID
        captureID = ""
        endpoint = nil
        starting = false
        activeSettings = nil
        callQueue.async { [weak self] in
            guard let self else { return }
            if !oldCaptureID.isEmpty, let ownerID = try? RuntimeSettingsStore().ownerID() {
                _ = try? self.embedded.call("stop_capture", arguments: [ownerID, oldCaptureID])
            }
            self.embedded.stop()
        }
    }

    func startCapture() -> String? {
        guard let endpoint else { return nil }
        do {
            let raw = try embedded.call("start_capture", arguments: [endpoint.ownerID])
            guard let data = raw.data(using: .utf8),
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = object["capture_id"] as? String, !id.isEmpty else { return nil }
            captureID = id
            return id
        } catch {
            status = "本机音频 capture 启动失败：\(error.localizedDescription)"
            return nil
        }
    }

    func ingestAudioEvent(_ event: [String: Any], privateEvent: [String: Any]) {
        guard let endpoint, !captureID.isEmpty else { return }
        guard JSONSerialization.isValidJSONObject(event), JSONSerialization.isValidJSONObject(privateEvent),
              let eventData = try? JSONSerialization.data(withJSONObject: event),
              let privateData = try? JSONSerialization.data(withJSONObject: privateEvent) else { return }
        let args = [endpoint.ownerID, captureID, String(decoding: eventData, as: UTF8.self), String(decoding: privateData, as: UTF8.self), "{}"]
            self.callQueue.async { [weak self] in
                do { _ = try self?.embedded.call("ingest_audio_event", arguments: args) }
            catch { DispatchQueue.main.async { self?.status = "音频事件提交失败：\(error.localizedDescription)" } }
        }
    }

    func submitAudioEvent(_ event: [String: Any], privateEvent: [String: Any], completion: @escaping ([String: Any]) -> Void) {
        guard let endpoint, !captureID.isEmpty else { return }
        guard JSONSerialization.isValidJSONObject(event), JSONSerialization.isValidJSONObject(privateEvent),
              let eventData = try? JSONSerialization.data(withJSONObject: event),
              let privateData = try? JSONSerialization.data(withJSONObject: privateEvent) else { return }
        let args = [endpoint.ownerID, captureID, String(decoding: eventData, as: UTF8.self), String(decoding: privateData, as: UTF8.self), "{}"]
        callQueue.async { [weak self] in
            guard let self else { return }
            do {
                let queuedRaw = try self.embedded.call("ingest_audio_event", arguments: args)
                guard let queuedData = queuedRaw.data(using: .utf8),
                      let queued = try JSONSerialization.jsonObject(with: queuedData) as? [String: Any],
                      let eventID = queued["event_id"] as? String else { return }
                let resultRaw = try self.embedded.call("wait_audio_event", arguments: [endpoint.ownerID, eventID, "30"])
                guard let resultData = resultRaw.data(using: .utf8),
                      let result = try JSONSerialization.jsonObject(with: resultData) as? [String: Any] else { return }
                DispatchQueue.main.async { completion(result) }
            } catch {
                DispatchQueue.main.async { self.status = "音频事件处理失败：\(error.localizedDescription)" }
            }
        }
    }

    func stopCapture() {
        guard let endpoint, !captureID.isEmpty else { return }
        let ownerID = endpoint.ownerID
        let id = captureID
        captureID = ""
        callQueue.async { [weak self] in
            do { _ = try self?.embedded.call("stop_capture", arguments: [ownerID, id]) }
            catch { DispatchQueue.main.async { self?.status = "本机音频 capture 停止失败：\(error.localizedDescription)" } }
        }
    }

    /// Forward the native audio/model snapshot without blocking the main actor.
    func setDeviceState(_ state: [String: Any]) {
        guard endpoint != nil,
              JSONSerialization.isValidJSONObject(state),
              let data = try? JSONSerialization.data(withJSONObject: state) else { return }
        let payload = String(decoding: data, as: UTF8.self)
        callQueue.async { [weak self] in
            do { _ = try self?.embedded.call("set_device_state", arguments: [payload]) }
            catch {
                DispatchQueue.main.async { self?.status = "本机音频状态同步失败：\(error.localizedDescription)" }
            }
        }
    }

    private func bootstrapStore() {
        do {
            let root = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                .appendingPathComponent("AIGlassesMemoryAssistant", isDirectory: true)
            store = try LocalAssistantStore(directory: root)
            try refreshCounts()
            status = "本机聊天、记忆和审计服务已就绪"
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
