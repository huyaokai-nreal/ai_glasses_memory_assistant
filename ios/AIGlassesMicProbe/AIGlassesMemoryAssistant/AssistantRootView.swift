import AVFoundation
import CoreLocation
import SwiftUI
import UniformTypeIdentifiers
import WebKit

struct AssistantRootView: View {
    @ObservedObject var audio: AssistantAudioController
    @ObservedObject var runtime: LocalAssistantRuntime
    @State private var diagnosticFile: URL?
    @State private var showingSettings = false

    var body: some View {
        NavigationStack {
            AssistantWebView(audio: audio) { showingSettings = true }
                .ignoresSafeArea()
        }
        .navigationBarHidden(true)
        .sheet(isPresented: $showingSettings) {
            RuntimeSettingsView(audio: audio, runtime: runtime) { diagnosticFile = $0 }
        }
        .sheet(isPresented: Binding(get: { diagnosticFile != nil }, set: { if !$0 { diagnosticFile = nil } })) {
            if let diagnosticFile { ShareSheet(items: [diagnosticFile]) }
        }
    }
}

private struct AssistantWebView: UIViewRepresentable {
    let audio: AssistantAudioController
    let openSettings: () -> Void

    final class Coordinator {
        var bridge: IOSNativeBridge?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        let bridge = IOSNativeBridge(audio: audio)
        bridge.openSettings = openSettings
        configuration.userContentController.addScriptMessageHandler(
            bridge,
            contentWorld: .page,
            name: "aiGlassesNative"
        )
        configuration.userContentController.addUserScript(WKUserScript(
            source: "window.__AI_GLASSES_PLATFORM__ = 'ios'; window.AiGlassesNative = { request: function(method, payload) { return window.webkit.messageHandlers.aiGlassesNative.postMessage({ method: method, payload: payload || {} }); } };",
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        ))
        context.coordinator.bridge = bridge
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground
        guard let pageURL = Bundle.main.url(forResource: "index", withExtension: "html", subdirectory: "Web"),
              var html = try? String(contentsOf: pageURL, encoding: .utf8) else {
            webView.loadHTMLString("<p>本机网页资源未打包。</p>", baseURL: nil)
            return webView
        }
        // index.html uses absolute /static/... paths that work on the web server
        // but resolve to file:///static/... (nonexistent) under loadFileURL.
        // Replace them with relative paths so the WKWebView can find CSS/JS in
        // the same Web/ bundle directory.
        html = html.replacingOccurrences(of: "/static/", with: "")
        let baseURL = pageURL.deletingLastPathComponent()
        webView.loadHTMLString(html, baseURL: baseURL)
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        context.coordinator.bridge?.openSettings = openSettings
    }
}

private struct RuntimeSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    let audio: AssistantAudioController
    let runtime: LocalAssistantRuntime
    let onDiagnosticExport: (URL) -> Void
    @State private var settings = RuntimeSettings()
    @State private var modelStatus = BundledModelPackValidator.validate()
    @State private var errorMessage = ""
    @State private var diagnosticPassphrase = ""
    @State private var locationManager = CLLocationManager()
    private let store = RuntimeSettingsStore()

    // Online mic probe state
    @State private var probeStatus = "测试前请停止全天收音；蓝牙设备接入时会优先且仅使用蓝牙收音"
    @State private var probeRunning = false
    private let prober = AudioInputProber()

    // Offline audio test state
    @State private var offlineStatus = "选择 PCM16 WAV 或 AAC M4A 后，可直接测试本机 VAD 与 SenseVoice；不会写入记忆或审计"
    @State private var offlineFileURL: URL?
    @State private var offlineFileName = ""
    @State private var offlineTesting = false
    @State private var offlineResult: OfflineTestResult?
    @State private var offlineGainEnabled = false
    @State private var offlineGainDecibels = 0.0
    @State private var offlineFilePickerPresented = false

    var body: some View {
        NavigationStack {
            Form {
                Section("联网模型") {
                    TextField("服务提供商", text: $settings.provider)
                    TextField("模型名称", text: $settings.model)
                    TextField("HTTPS 地址", text: $settings.baseURL)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                    SecureField("API 密钥", text: $settings.apiKey)
                        .textInputAutocapitalization(.never)
                }
                Section("本地语音模型") {
                    LabeledContent("状态", value: modelStatus.message)
                    if let version = modelStatus.version { LabeledContent("版本", value: version) }
                    LabeledContent("五项自检", value: "推理管线尚未初始化，待验证")
                    Button("重新校验模型") {
                        // 在后台执行 SHA-256 校验（~290MB 模型文件），避免阻塞主线程 1-3 秒
                        DispatchQueue.global(qos: .userInitiated).async {
                            let result = BundledModelPackValidator.validate()
                            DispatchQueue.main.async { modelStatus = result }
                        }
                    }
                }
                Section("前台音频") {
                    LabeledContent("实际输入", value: audio.routeName)
                    LabeledContent("状态", value: audio.state.label)
                    Button("测试 TTS") { audio.speak("这是 iPhone 本机语音回复测试") }
                }
                // MARK: Online mic probe
                Section("收音设备") {
                    Text(probeStatus)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button(probeRunning ? "测试中…" : "测试收音") { startMicProbe() }
                        .disabled(probeRunning || audio.state != .idle)
                }
                // MARK: Offline audio model test
                Section("离线音频模型测试") {
                    Text(offlineStatus)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Toggle("使用增益", isOn: $offlineGainEnabled)
                        .disabled(offlineTesting)
                    if offlineGainEnabled {
                        HStack {
                            Text("增益：\(String(format: "%+.0f", offlineGainDecibels)) dB")
                                .font(.caption)
                            Slider(value: $offlineGainDecibels, in: -24...24, step: 1) {
                                Text("增益")
                            }
                                .disabled(offlineTesting)
                        }
                        Button("恢复 0 dB") { offlineGainDecibels = 0 }
                    }
                    HStack {
                        Button("选择 WAV 或 M4A 文件") { offlineFilePickerPresented = true }
                            .disabled(offlineTesting)
                        if !offlineFileName.isEmpty {
                            Text(offlineFileName)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .fileImporter(isPresented: $offlineFilePickerPresented, allowedContentTypes: [.audio], allowsMultipleSelection: false) { result in
                        if case let .success(urls) = result, let url = urls.first {
                            guard url.startAccessingSecurityScopedResource() else { return }
                            offlineFileURL = url
                            offlineFileName = url.lastPathComponent
                            offlineResult = nil
                            offlineStatus = "已选择音频：\(url.lastPathComponent)"
                        }
                    }
                    if offlineTesting {
                        Button("取消离线测试") { cancelOfflineTest() }
                            .foregroundStyle(.red)
                    } else {
                        Button("运行离线 VAD/ASR 测试") { startOfflineTest() }
                            .disabled(offlineFileURL == nil || modelStatus.version == nil)
                    }
                    if let result = offlineResult {
                        Button("复制转写文本") {
                            UIPasteboard.general.string = result.transcript
                        }
                        Text(offlineResultText(result))
                            .font(.footnote)
                            .textSelection(.enabled)
                    }
                }
                Section("定位") {
                    LabeledContent("权限", value: locationAuthorizationLabel)
                    Button("请求定位授权") { locationManager.requestWhenInUseAuthorization() }
                }
                Section("加密诊断") {
                    SecureField("导出密码", text: $diagnosticPassphrase)
                        .textInputAutocapitalization(.never)
                    Button("生成加密诊断包") {
                        guard diagnosticPassphrase.count >= 8 else {
                            errorMessage = "诊断导出密码至少需要 8 个字符"
                            return
                        }
                        runtime.bootstrap()
                        guard let file = runtime.exportDiagnostic(passphrase: diagnosticPassphrase) else {
                            errorMessage = runtime.status
                            return
                        }
                        diagnosticPassphrase = ""
                        errorMessage = ""
                        onDiagnosticExport(file)
                    }
                    Text("诊断包不包含 API Key、原始 PCM 或声纹向量。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if !errorMessage.isEmpty { Section { Text(errorMessage).foregroundStyle(.red) } }
            }
            .navigationTitle("配置")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("关闭") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("保存") {
                        do { try store.save(settings); errorMessage = ""; dismiss() }
                        catch { errorMessage = error.localizedDescription }
                    }
                }
            }
            .onAppear {
                do { settings = try store.load() }
                catch { errorMessage = error.localizedDescription }
                locationManager = CLLocationManager()
            }
            .onDisappear { prober.cancel(); cancelOfflineTest() }
        }
    }

    // MARK: - Online Mic Probe

    private func startMicProbe() {
        guard !probeRunning else { return }
        probeRunning = true
        probeStatus = "正在测试收音，请对收音设备说话…"
        prober.onUpdate = { snapshot in
            probeStatus = micProbeText(snapshot)
        }
        prober.onComplete = { result in
            probeStatus = "测试结果：\(result.status)\n\(result.guidance)"
            probeRunning = false
        }
        DispatchQueue.global(qos: .userInitiated).async {
            do { try prober.start() }
            catch {
                DispatchQueue.main.async {
                    probeStatus = "测试失败：\(error.localizedDescription)"
                    probeRunning = false
                }
            }
        }
    }

    private func micProbeText(_ snapshot: AudioProbeSnapshot) -> String {
        var lines: [String] = []
        lines.append("当前收音设备：\(snapshot.route)")
        lines.append("蓝牙输入：\(snapshot.isBluetooth ? "是" : "否")")
        lines.append("当前峰值：\(String(format: "%.1f", snapshot.peakDbfs)) dBFS")
        if probeRunning { lines.append("测试状态：正在收音，请说话…") }
        return lines.joined(separator: "\n")
    }

    // MARK: - Offline Audio Test

    private func startOfflineTest() {
        guard let fileURL = offlineFileURL, !offlineTesting else { return }
        guard let packDir = Bundle.main.resourceURL?.appendingPathComponent("Models"),
              let data = try? Data(contentsOf: packDir.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode(IOSModelPackManifest.self, from: data),
              (try? manifest.validate()) != nil else {
            offlineStatus = "离线测试失败：模型包未安装或校验不通过"
            return
        }
        offlineTesting = true
        offlineResult = nil
        offlineStatus = "正在运行本机 VAD 与 SenseVoice，请稍候…"
        let tester = OfflineAudioTester(packDir: packDir, manifest: manifest)
        let gainEnabled = offlineGainEnabled
        let gainDb = Int(offlineGainDecibels)
        DispatchQueue.global(qos: .userInitiated).async {
            let result = Result { try tester.run(fileURL: fileURL, gainEnabled: gainEnabled, gainDecibels: gainDb) }
            DispatchQueue.main.async {
                offlineTesting = false
                switch result {
                case let .success(r):
                    offlineResult = r
                    offlineStatus = "离线音频测试完成；结果仅保留在当前设置页"
                case let .failure(error):
                    offlineStatus = "离线音频测试失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private func cancelOfflineTest() {
        offlineTesting = false
        offlineStatus = "离线音频测试已取消"
    }

    private func offlineResultText(_ result: OfflineTestResult) -> String {
        var lines: [String] = []
        lines.append("模型包：\(result.modelVersion)")
        lines.append("输入：\(result.sourceFormat) / \(Int(result.sourceSampleRate)) Hz / \(result.sourceChannelCount) 声道 / \(String(format: "%.1f", result.sourceDurationSec)) 秒")
        lines.append("规范化：16000 Hz 单声道 / \(result.normalizedSampleCount) 样本")
        lines.append("增益：\(result.gainEnabled ? "\(String(format: "%+d", result.gainDecibels)) dB" : "未启用，使用原音频")")
        lines.append("处理耗时：\(result.elapsedMillis) ms")
        if result.segments.isEmpty {
            lines.append("VAD 未检测到语音片段")
        } else {
            lines.append("VAD/ASR 片段：")
            result.segments.enumerated().forEach { i, seg in
                lines.append("\(i + 1). \(seg.startMillis)–\(seg.endMillis) ms：\(seg.text.isEmpty ? "（无文字）" : seg.text)")
            }
            lines.append("合并转写：\(result.transcript.isEmpty ? "（无文字）" : result.transcript)")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Location

    private var locationAuthorizationLabel: String {
        switch locationManager.authorizationStatus {
        case .authorizedAlways: return "始终允许"
        case .authorizedWhenInUse: return "使用 App 期间允许"
        case .denied: return "已拒绝"
        case .restricted: return "受限制"
        case .notDetermined: return "尚未请求"
        @unknown default: return "未知"
        }
    }
}
