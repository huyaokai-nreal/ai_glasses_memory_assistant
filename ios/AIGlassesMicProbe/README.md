# AIGlassesMicProbe

独立的 iOS 16+ SwiftUI 蓝牙麦克风验证器。它只在 iOS 的实际输入路由为唯一的 `BluetoothHFP` 时开始录音；蓝牙配对、蓝牙播放输出或手机内置麦克风都不能通过验证。

录音由用户显式开始，保存为 App Documents/Recordings 下的本地 AAC `.m4a` 文件，可在 App 内回放或删除。它不联网、不发送音频、不接入 Python、Timeline、SQLite、audit 或记忆流程；进入后台、音频中断、路由变化和媒体服务重置时会停止录音。

## Build later

当前 Mac 尚未安装完整 Xcode 或 iPhone SDK。安装 Xcode 后，用 Xcode 打开 `AIGlassesMicProbe.xcodeproj`，为 App target 选择个人 Development Team，再连接 iPhone 12 mini 运行。首次真机验收必须确认页面显示 `蓝牙通话麦克风 (HFP)`，并通过回放一段独特口述内容验证实际收音来源。

建议的构建与测试命令：

```bash
xcodebuild -project ios/AIGlassesMicProbe/AIGlassesMicProbe.xcodeproj \
  -scheme AIGlassesMicProbe \
  -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  test
```
