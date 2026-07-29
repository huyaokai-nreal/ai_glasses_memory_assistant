import SwiftUI

struct AssistantRootView: View {
    @ObservedObject var audio: AssistantAudioController

    var body: some View {
        NavigationStack {
            List {
                Section("持续收音") {
                    LabeledContent("状态", value: audio.state.label)
                    LabeledContent("实际输入", value: audio.routeName)
                    LabeledContent("已处理音频帧", value: "\(audio.frameCount)")
                    HStack {
                        Button("开始") { audio.start() }.disabled(audio.state == .preparing || audio.state == .listening || audio.state == .pausedForSpeech)
                        Button("停止", role: .destructive) { audio.stop() }.disabled(audio.state == .idle)
                    }
                }
                Section("本地模型") { Text("需要安装并通过 VAD、唤醒、流式 ASR、环境 ASR、声纹五项自检后，音频帧才会进入推理链路。") }
                Section("记忆与聊天") { Text("CPython POC 尚未安装，当前只验证本机音频路由。partial 不会写入 Timeline、记忆或审计。") }
                Section("语音回复") { Button("测试 TTS") { audio.speak("正在验证语音回复期间暂停收音") }.disabled(audio.state != .listening) }
            }
            .navigationTitle("AI 眼镜记忆助手")
        }
    }
}
