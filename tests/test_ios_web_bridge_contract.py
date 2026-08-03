import json
from pathlib import Path

from ai_glasses_memory_assistant import android_runtime


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
    assert 'case "startAmbient"' in bridge
    assert 'case "startSpeakerEnrollment"' in bridge
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

    assert '"model_state"' in audio
    assert '"transcription_ready": pipeline != nil' in audio
    assert "try startEngine()\n                try verifyActiveInput(expected: input)" in audio
    assert 'audio.stop(reason: "App 已离开前台，已停止收音")' in app


def test_python_runtime_accepts_ios_audio_route_state() -> None:
    payload = json.loads(android_runtime.set_device_state(json.dumps({
        "state": "recording",
        "running": True,
        "capture_id": "ios-capture",
        "sample_rate": 16_000,
        "channels": 1,
        "encoding": "pcm_float32_native",
        "captured_samples": 1600,
        "model_state": "ready",
        "model_version": "x4000-sherpa-1.13.4-v2",
        "model_self_test": {"state": "ready"},
        "transcription_ready": True,
        "input_device_type": "bluetoothHFP",
        "input_device_source": "bluetooth_hfp",
        "input_device_name": "Mic Pro",
        "last_error": "",
    })))
    assert payload["input_device_type"] == "bluetoothHFP"
    assert payload["input_device_source"] == "bluetooth_hfp"
    assert payload["input_device_name"] == "Mic Pro"
    assert payload["captured_samples"] == 1600


def test_ios_runtime_state_sync_and_restart_contract() -> None:
    runtime = (ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/LocalAssistantRuntime.swift").read_text(encoding="utf-8")
    bridge = (ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/IOSNativeBridge.swift").read_text(encoding="utf-8")
    audio = (ROOT / "ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/AssistantAudioController.swift").read_text(encoding="utf-8")

    assert 'embedded.call("set_device_state"' in runtime
    assert "private let callQueue = DispatchQueue" in runtime
    assert "private var generation = 0" in runtime
    assert "self.embedded.stop()" in runtime
    assert "audio.setCaptureID(captureId)" in bridge
    assert "private func syncDeviceState()" in bridge
    for field in ("capture_id", "sample_rate", "channels", "encoding", "captured_samples",
                  "model_state", "model_version", "model_self_test", "transcription_ready",
                  "input_device_type", "input_device_source", "input_device_name", "last_error"):
        assert f'"{field}"' in audio
    assert "requiresHFPOrExplicitFallback" in audio
    assert "routeSource = selectedInput.portType == .bluetoothHFP ? \"bluetooth_hfp\" : \"iphone_builtin_mic\"" in audio
