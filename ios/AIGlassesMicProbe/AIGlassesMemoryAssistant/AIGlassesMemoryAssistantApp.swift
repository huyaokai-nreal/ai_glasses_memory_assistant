import SwiftUI

@main
struct AIGlassesMemoryAssistantApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var audio = AssistantAudioController()
    @StateObject private var runtime = LocalAssistantRuntime()

    var body: some Scene {
        WindowGroup {
            AssistantRootView(audio: audio, runtime: runtime)
                .task { runtime.bootstrap() }
                .onChange(of: scenePhase) { phase in
                    if phase == .active {
                        // Restore Python runtime and page connection after
                        // returning from background.
                        runtime.bootstrap()
                        return
                    }
                    audio.stop(reason: "App 已离开前台，已停止收音")
                    runtime.shutdown()
                }
        }
    }

    init() {
        let env = ProcessInfo.processInfo.environment["AI_GLASSES_AUTO_AMBIENT"] ?? "nil"
        let args = CommandLine.arguments
        NSLog("[AIGlassesMemoryAssistantApp] init env AI_GLASSES_AUTO_AMBIENT=\(env) args=\(args)")
        // Debug helper: set env var `AI_GLASSES_AUTO_AMBIENT=1` to auto-trigger
        // startAmbient for testing without physical taps.
        if env == "1" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 8.0) {
                NSLog("[AIGlassesMemoryAssistantApp] auto-ambient triggered")
                NotificationCenter.default.post(name: .autoAmbientTest, object: nil)
            }
        }
    }
}

extension Notification.Name {
    static let autoAmbientTest = Notification.Name("autoAmbientTest")
}
