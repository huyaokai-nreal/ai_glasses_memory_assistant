from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mobile_app_navigation_contract() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")
    activity = (
        ROOT
        / "android/app/src/main/java/com/aiglasses/memoryassistant/demo/MainActivity.kt"
    ).read_text(encoding="utf-8")

    for element_id in (
        "settings-toggle",
        "settings-pane",
        "settings-close",
        "memory-close",
        "settings-audio-details",
        "native-settings-button",
    ):
        assert f'id="{element_id}"' in html

    assert "window.aiGlassesHandleBack" in app
    assert "window.aiGlassesHandleBack()" in activity
    assert html.count('aria-expanded="false"') >= 3
    assert 'role="switch" aria-checked="true"' in html
    assert "body.settings-open .topbar-actions" in styles
    assert "body.debug-open" in styles
    assert "body.speaker-open" in styles
    assert "width: 100vw" in styles


def test_reply_evidence_stays_inline_and_read_only() -> None:
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")
    start = app.index("function appendReplyEvidenceControls")
    end = app.index("function renderTimelinePanel", start)
    controls = app[start:end]

    assert 'toggle.textContent = "回答依据"' in controls
    assert 'panel.className = "reply-evidence-panel"' in controls
    assert "renderReplyEvidencePanel" in controls
    assert "renderTimelinePanel" not in controls
    assert "deleteTimeline" not in controls
    assert "可能含 ASR 错误" in app
    assert "说话人未确认" in app
    assert ".reply-evidence-panel" in styles
    assert ".reply-evidence-item summary" in styles


def test_speaker_enrollment_uses_three_distinct_phrases_without_hermes() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    phrases = (
        "你好小忆，现在开始录入我的声音。",
        "清晨的街道很安静，我准备出门散步。",
        "明天下午三点，我们一起讨论新的计划。",
    )

    positions = [app.index(phrase) for phrase in phrases]
    assert positions == sorted(positions)
    assert len(set(phrases)) == 3
    assert phrases[0] in html
    assert "Hermes" not in html
    assert "Hermes" not in app
    assert "固定短句" not in html
    assert "固定短句" not in app


def test_android_wake_query_is_visible_and_final_only() -> None:
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    bridge = (
        ROOT
        / "android/app/src/main/java/com/aiglasses/memoryassistant/demo/NativeAppBridge.kt"
    ).read_text(encoding="utf-8")
    pipeline = (
        ROOT
        / "android/app/src/main/java/com/aiglasses/memoryassistant/demo/NativeModelPipeline.kt"
    ).read_text(encoding="utf-8")

    assert 'fun audioUiStatus()' in bridge
    assert 'interactionState = INTERACTION_WAITING_QUERY' in pipeline
    assert 'NativeAudioState.markFinalQuery(eventId, text)' in pipeline
    assert 'showNativeFinalQuery(String(status.final_query_event_id' in app
    assert 'nativeQueryEventIds.has(eventId)' in app
    assert '"我在，请说"' not in app
