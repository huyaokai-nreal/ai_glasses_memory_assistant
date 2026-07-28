import SwiftUI

struct ContentView: View {
    @StateObject private var controller = RecordingController()

    var body: some View {
        NavigationStack {
            List {
                Section("实际输入") {
                    if let input = controller.activeInput {
                        inputDetails(input, state: "已验证")
                    } else {
                        Text("尚未验证蓝牙输入")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(controller.availableInputs) { input in
                        inputDetails(input, state: input.kind == .bluetoothHFP ? "可选" : "不接受")
                    }
                    Button {
                        controller.refreshRoute()
                    } label: {
                        Label("重新检测", systemImage: "arrow.clockwise")
                    }
                    .accessibilityHint("重新读取 iPhone 当前可用的音频输入设备")
                }

                Section("录音") {
                    HStack {
                        Text("输入电平")
                        ProgressView(value: controller.level)
                        Text("\(Int(controller.level * 100))%")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("录音时长")
                        Spacer()
                        Text(durationText(controller.elapsedSeconds))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    if controller.isRecording {
                        Button(role: .destructive) {
                            controller.stopRecording()
                        } label: {
                            Label("停止录音", systemImage: "stop.circle.fill")
                        }
                    } else {
                        Button {
                            controller.startRecording()
                        } label: {
                            Label("开始录音", systemImage: "record.circle")
                        }
                        .disabled(controller.isPreparing)
                    }
                }

                Section("状态") {
                    Text(controller.statusMessage)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("本机录音") {
                    if controller.recordings.isEmpty {
                        Text("还没有本机录音")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(controller.recordings) { recording in
                        HStack(spacing: 12) {
                            Button {
                                controller.play(recording)
                            } label: {
                                Image(systemName: controller.playingRecordingID == recording.id ? "stop.fill" : "play.fill")
                            }
                            .accessibilityLabel(controller.playingRecordingID == recording.id ? "停止回放" : "回放录音")
                            VStack(alignment: .leading, spacing: 3) {
                                Text(recording.createdAt.formatted(date: .abbreviated, time: .standard))
                                Text(durationText(recording.duration))
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                        }
                        .swipeActions {
                            Button(role: .destructive) {
                                controller.delete(recording)
                            } label: {
                                Label("删除", systemImage: "trash")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Mic Pro 收音验证")
            .task {
                controller.startMonitoring()
            }
        }
    }

    @ViewBuilder
    private func inputDetails(_ input: AudioInputRoute, state: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(input.name)
                Spacer()
                Text(state)
                    .font(.caption)
                    .foregroundStyle(input.kind == .bluetoothHFP ? .green : .secondary)
            }
            Text(input.kind.displayName)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private func durationText(_ duration: TimeInterval) -> String {
        let seconds = Int(duration.rounded(.down))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}
