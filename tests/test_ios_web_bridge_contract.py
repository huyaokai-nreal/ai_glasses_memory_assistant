from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ios_webview_uses_the_shared_async_native_bridge() -> None:
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    styles = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")
    root_view = (
        ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/AssistantRootView.swift"
    ).read_text(encoding="utf-8")
    bridge = (
        ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/IOSNativeBridge.swift"
    ).read_text(encoding="utf-8")

    assert "window.AiGlassesNative" in root_view
    assert "WKScriptMessageHandlerWithReply" in bridge
    assert "addScriptMessageHandler(" in root_view
    assert "contentWorld: .page" in root_view
    assert "RuntimeSettingsView(audio: audio, runtime: runtime)" in root_view
    assert "生成加密诊断包" in root_view
    assert "请求定位授权" in root_view
    assert "async function callNativeBridge(method, payload = {})" in app
    assert "function isIOSNative()" in app
    assert 'callNativeBridge("location")' in app
    assert 'case "platform"' in bridge
    assert 'case "ownerId"' in bridge
    assert 'case "audioStatus", "audioUiStatus"' in bridge
    assert 'case "openSettings"' in bridge
    assert 'case "location"' in bridge
    assert 'case "startAmbient", "startSpeakerEnrollment"' in bridge
    assert "pipelineUnavailable" in bridge
    assert "native-only-setting" in html
    assert "body.android-native .setting-row.native-only-setting" in styles
    assert "body.ios-native .setting-row.native-only-setting" in styles


def test_ios_native_audio_stays_fail_closed_until_real_models_are_initialized() -> None:
    audio = (
        ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/AssistantAudioController.swift"
    ).read_text(encoding="utf-8")
    app = (
        ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/AIGlassesMemoryAssistantApp.swift"
    ).read_text(encoding="utf-8")

    assert '"model_state": model.ready ? "manifest_valid_pipeline_unavailable" : "invalid"' in audio
    assert '"transcription_ready": false' in audio
    assert "try startEngine()\n            try verifyActiveHFPInput(expected: input)" in audio
    assert 'audio.stop(reason: "App 已离开前台，已停止收音")' in app
