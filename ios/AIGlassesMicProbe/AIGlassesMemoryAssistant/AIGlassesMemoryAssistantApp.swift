import SwiftUI

@main
struct AIGlassesMemoryAssistantApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var audio = AssistantAudioController()
    @StateObject private var runtime = LocalAssistantRuntime()

    var body: some Scene {
        WindowGroup {
            AssistantRootView(audio: audio, runtime: runtime)
                .onChange(of: scenePhase) { phase in
                    guard phase != .active else { return }
                    audio.stop(reason: "App 已离开前台，已停止收音")
                }
        }
    }
}
