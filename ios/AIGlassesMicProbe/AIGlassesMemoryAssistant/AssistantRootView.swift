import CoreLocation
import SwiftUI
import WebKit

struct AssistantRootView: View {
    @ObservedObject var audio: AssistantAudioController
    @ObservedObject var runtime: LocalAssistantRuntime
    @State private var diagnosticFile: URL?
    @State private var showingSettings = false

    var body: some View {
        NavigationStack {
            AssistantWebView(audio: audio) { showingSettings = true }
                .ignoresSafeArea(edges: .bottom)
            .navigationTitle("AI 眼镜记忆助手")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("配置", systemImage: "gearshape") { showingSettings = true }
                        .accessibilityLabel("打开配置")
                }
            }
            .sheet(isPresented: $showingSettings) {
                RuntimeSettingsView(audio: audio, runtime: runtime) { diagnosticFile = $0 }
            }
            .sheet(isPresented: Binding(get: { diagnosticFile != nil }, set: { if !$0 { diagnosticFile = nil } })) {
                if let diagnosticFile { ShareSheet(items: [diagnosticFile]) }
            }
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
        guard let page = Bundle.main.url(forResource: "index", withExtension: "html", subdirectory: "Web") else {
            webView.loadHTMLString("<p>本机网页资源未打包。</p>", baseURL: nil)
            return webView
        }
        webView.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent())
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
                    Button("重新校验模型") { modelStatus = BundledModelPackValidator.validate() }
                }
                Section("前台音频") {
                    LabeledContent("实际输入", value: audio.routeName)
                    LabeledContent("状态", value: audio.state.label)
                    Button("测试 TTS") { audio.speak("这是 iPhone 本机语音回复测试") }
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
        }
    }

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
