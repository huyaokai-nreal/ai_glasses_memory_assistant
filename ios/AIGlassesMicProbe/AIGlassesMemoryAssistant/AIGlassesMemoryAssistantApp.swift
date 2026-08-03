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
}
