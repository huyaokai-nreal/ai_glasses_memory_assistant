import SwiftUI

@main
struct AIGlassesMemoryAssistantApp: App {
    @StateObject private var audio = AssistantAudioController()

    var body: some Scene {
        WindowGroup { AssistantRootView(audio: audio) }
    }
}
