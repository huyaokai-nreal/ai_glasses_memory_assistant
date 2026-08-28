#!/usr/bin/env python3
"""Run privacy-preserving acceptance checks against one explicit Android device."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE = "com.aiglasses.memoryassistant.demo"
ACTIVITY = f"{PACKAGE}/.MainActivity"
PUBLIC_SNAPSHOT_EXPRESSION = r"""
(async () => {
  const native = state.audio.nativeStatus || {};
  const ambient = native.ambient_context || {};
  const runtime = await fetch('/api/runtime').then((response) => response.json());
  const audit = await fetch(`/api/debug/audit?user_id=${encodeURIComponent(state.userId)}&limit=100`)
    .then((response) => response.json());
  const auditTypes = {};
  for (const record of (audit.records || [])) {
    const type = String(record.record_type || 'unknown');
    auditTypes[type] = (auditTypes[type] || 0) + 1;
  }
  return {
    platform: String(window.AiGlassesAndroid?.platform?.() || ''),
    page_status: String(document.body.dataset.status || ''),
    audio: {
      state: String(native.state || 'idle'),
      running: Boolean(native.running),
      capture_id_present: Boolean(native.capture_id),
      captured_samples: Number(native.captured_samples || 0),
      network_online: Boolean(native.network_online),
      audio_rms_dbfs: Number(native.audio_rms_dbfs ?? -120),
      audio_peak_dbfs: Number(native.audio_peak_dbfs ?? -120),
      input_device_name: String(native.input_device_name || ''),
      input_device_type: String(native.input_device_type || ''),
      input_device_source: String(native.input_device_source || ''),
      vad_segment_count: Number(native.vad_segment_count || 0),
      ambient_final_count: Number(native.ambient_final_count || 0),
      speech_rejected_count: Number(native.speech_rejected_count || 0),
      last_final_at_ms: Number(native.last_final_at_ms || 0),
      inference_queue_depth: Number(native.inference_queue_depth || 0),
      model_state: String(native.model_state || ''),
      model_version: String(native.model_version || ''),
      model_components: Object.fromEntries(Object.entries(native.model_self_test?.components || {})
        .map(([name, value]) => [name, {state: String(value.state || ''), elapsed_ms: Number(value.elapsed_ms || 0)}])),
      event_queue: native.device_event_queue || runtime.device_event_queue || {},
      ambient_context: {
        status: String(ambient.status || ''),
        chunk_count: Number(ambient.chunk_count || 0),
        last_chunk_id_present: Boolean(ambient.last_chunk_id),
        last_segment_id_present: Boolean(ambient.last_segment_id),
        last_captured_at: Number(ambient.last_captured_at || 0),
      },
      ui_context_count: Number(state.ambient.chunkCount || 0),
    },
    location: state.location ? {
      status: String(state.location.status || ''),
      error: String(state.location.error || ''),
      accuracy: Number(state.location.accuracy || 0),
    } : null,
    audit_types: auditTypes,
  };
})()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="ADB device serial")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument("--scenario", default="default", help="Operator-defined connection scenario recorded in the report")
    parser.add_argument(
        "--expected-input",
        choices=("bluetooth", "usb", "system"),
        help="Fail the route check unless Android reports this actual input source",
    )
    parser.add_argument("--speech", action="store_true", help="Play the fixed Mandarin Mac TTS set")
    parser.add_argument(
        "--speech-source",
        choices=("mac", "device"),
        default="mac",
        help="Use Mac speakers, or the device speaker as a controlled acoustic fallback",
    )
    parser.add_argument("--location", action="store_true", help="Temporarily grant location and request one fix")
    parser.add_argument("--background-minutes", type=int, default=0, help="Keep capture running with the screen off")
    parser.add_argument("--stop-after", action="store_true", help="Stop capture if this run started it")
    return parser.parse_args()


class Adb:
    def __init__(self, executable: str, serial: str):
        self.executable = executable
        self.serial = serial

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, "-s", self.serial, *args],
            check=check,
            text=True,
            capture_output=True,
        )

    def shell(self, *args: str, check: bool = True) -> str:
        return self.run("shell", *args, check=check).stdout.strip()


class DevToolsSocket:
    def __init__(self, url: str):
        match = re.fullmatch(r"ws://([^/:]+):(\d+)(/.*)", url)
        if not match:
            raise ValueError("unsupported DevTools WebSocket URL")
        host, port, path = match.group(1), int(match.group(2)), match.group(3)
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = self._read_http_headers()
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if " 101 " not in response.splitlines()[0] or expected.lower() not in response.lower():
            self.close()
            raise RuntimeError("DevTools WebSocket handshake failed")
        self.next_id = 1

    def close(self) -> None:
        if getattr(self, "sock", None) is not None:
            self.sock.close()
            self.sock = None

    def evaluate(self, expression: str) -> Any:
        request_id = self.next_id
        self.next_id += 1
        self._send_json({
            "id": request_id,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "awaitPromise": True, "returnByValue": True},
        })
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            payload = self._receive_json()
            if payload.get("id") != request_id:
                continue
            if payload.get("exceptionDetails"):
                raise RuntimeError(_devtools_error_message("DevTools expression failed", payload))
            result = payload.get("result", {}).get("result", {})
            if result.get("subtype") == "error":
                raise RuntimeError(_devtools_error_message("DevTools returned an error", payload))
            return result.get("value")
        raise TimeoutError("DevTools evaluation timed out")

    def _read_http_headers(self) -> str:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            data.extend(self.sock.recv(4096))
            if len(data) > 32_768:
                raise RuntimeError("oversized WebSocket handshake")
        return data.decode("latin-1")

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        mask = secrets.token_bytes(4)
        header = bytearray((0x81,))
        if len(data) < 126:
            header.append(0x80 | len(data))
        elif len(data) < 65_536:
            header.extend((0x80 | 126, *struct.pack("!H", len(data))))
        else:
            header.extend((0x80 | 127, *struct.pack("!Q", len(data))))
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        self.sock.sendall(bytes(header) + mask + masked)

    def _receive_json(self) -> dict[str, Any]:
        while True:
            first, second = self._read_exact(2)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            payload = self._read_exact(length)
            if opcode == 0x9:
                self.sock.sendall(bytes((0x8A, len(payload))) + payload)
                continue
            if opcode == 0x8:
                raise ConnectionError("DevTools WebSocket closed")
            if opcode == 0x1:
                return json.loads(payload)

    def _read_exact(self, length: int) -> bytes:
        data = bytearray()
        while len(data) < length:
            chunk = self.sock.recv(length - len(data))
            if not chunk:
                raise ConnectionError("DevTools WebSocket ended")
            data.extend(chunk)
        return bytes(data)


def _devtools_error_message(prefix: str, payload: dict[str, Any]) -> str:
    """Keep bridge failures actionable without dumping an unbounded DevTools payload."""

    details = payload.get("exceptionDetails") or {}
    exception = details.get("exception") or payload.get("result", {}).get("result", {})
    detail = exception.get("description") or exception.get("value") or details.get("text") or ""
    normalized = " ".join(str(detail).split())[:600]
    return f"{prefix}: {normalized}" if normalized else prefix


def connect_devtools(adb: Adb) -> tuple[DevToolsSocket, str]:
    pid = adb.shell("pidof", PACKAGE).split()[0]
    forwarded = adb.run("forward", "tcp:0", f"localabstract:webview_devtools_remote_{pid}").stdout.strip()
    port = forwarded.rsplit(":", 1)[-1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10) as response:
        targets = json.load(response)
    page = next((target for target in targets if target.get("type") == "page"), None)
    if page is None:
        raise RuntimeError("no debuggable WebView page")
    websocket_url = re.sub(r"ws://[^/]+", f"ws://127.0.0.1:{port}", page["webSocketDebuggerUrl"])
    return DevToolsSocket(websocket_url), port


def reconnect_devtools(adb: Adb, current: DevToolsSocket | None, forwarded_port: str) -> tuple[DevToolsSocket, str]:
    if current is not None:
        current.close()
    if forwarded_port:
        adb.run("forward", "--remove", f"tcp:{forwarded_port}", check=False)
    return connect_devtools(adb)


def device_metrics(adb: Adb) -> dict[str, Any]:
    meminfo = adb.shell("dumpsys", "meminfo", PACKAGE, check=False)
    pss_match = re.search(r"TOTAL PSS:\s*(\d+)", meminfo)
    battery = adb.shell("dumpsys", "battery", check=False)
    values = dict(re.findall(r"^\s*(level|temperature|status):\s*(\d+)", battery, re.MULTILINE))
    appops = adb.shell("cmd", "appops", "get", PACKAGE, "RECORD_AUDIO", check=False)
    return {
        "pss_kb": int(pss_match.group(1)) if pss_match else None,
        "battery_level": int(values["level"]) if "level" in values else None,
        "temperature_tenths_c": int(values["temperature"]) if "temperature" in values else None,
        "record_audio_running": "running" in appops.lower(),
    }


def permission_granted(adb: Adb, permission: str) -> bool:
    output = adb.shell("dumpsys", "package", PACKAGE, check=False)
    match = re.search(rf"^\s*{re.escape(permission)}: granted=(true|false)", output, re.MULTILINE)
    return bool(match and match.group(1) == "true")


def wait_for(predicate, timeout: float, interval: float = 1.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def music_volume(adb: Adb) -> int | None:
    output = adb.shell("cmd", "media_session", "volume", "--stream", "3", "--get", check=False)
    match = re.search(r"volume is (\d+)", output)
    return int(match.group(1)) if match else None


def play_speech(adb: Adb, phrase: str, source: str, index: int) -> None:
    if source == "mac":
        subprocess.run(["say", "-v", "Tingting", phrase], check=True)
        return
    with tempfile.TemporaryDirectory(prefix="android-acceptance-speech-") as temporary:
        aiff = Path(temporary) / "speech.aiff"
        wav = Path(temporary) / "speech.wav"
        remote = f"/data/local/tmp/ai-glasses-acceptance-{os.getpid()}-{index}.wav"
        subprocess.run(["say", "-v", "Tingting", "-o", str(aiff), phrase], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        try:
            adb.run("push", str(wav), remote)
            adb.shell("tinyplay", remote)
        finally:
            adb.shell("rm", "-f", remote, check=False)


def add_check(report: dict[str, Any], name: str, status: str, evidence: dict[str, Any]) -> None:
    report["checks"].append({"name": name, "status": status, "evidence": evidence})
    print(f"[{status.upper()}] {name}", flush=True)


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"x4000-acceptance-{stamp}.json"
    md_path = output_dir / f"x4000-acceptance-{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Android Device Acceptance", "", f"- Device: `{report['device_serial']}`", f"- Started: `{report['started_at']}`", ""]
    for item in report["checks"]:
        lines.append(f"- **{item['status'].upper()}** {item['name']}: `{json.dumps(item['evidence'], ensure_ascii=False)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    adb = Adb(args.adb, args.serial)
    report: dict[str, Any] = {
        "schema": "android_acceptance.v2",
        "device_serial": args.serial,
        "scenario": args.scenario,
        "expected_input": args.expected_input,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": [],
    }
    devtools: DevToolsSocket | None = None
    forwarded_port = ""
    started_capture = False
    original_music_volume: int | None = None
    screen_turned_off = False
    location_permissions = {
        permission: permission_granted(adb, permission)
        for permission in ("android.permission.ACCESS_COARSE_LOCATION", "android.permission.ACCESS_FINE_LOCATION")
    }
    try:
        state = adb.run("get-state").stdout.strip()
        add_check(report, "adb_device", "pass" if state == "device" else "fail", {"state": state})
        adb.shell("am", "start", "-n", ACTIVITY)
        time.sleep(3)
        devtools, forwarded_port = connect_devtools(adb)
        before = devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)
        components = before["audio"]["model_components"]
        model_ok = set(components) == {"vad", "kws", "online_asr", "ambient_asr", "speaker"} and all(
            item.get("state") == "ok" for item in components.values()
        )
        add_check(report, "webview_and_models", "pass" if before["platform"] == "android" and model_ok else "fail", {
            "platform": before["platform"],
            "model_state": before["audio"]["model_state"],
            "model_version": before["audio"]["model_version"],
            "components": components,
        })
        initially_running = bool(before["audio"]["running"])
        if not initially_running:
            devtools.evaluate("document.querySelector('#ambient-standby-toggle').click(); true")
            started_capture = True
        active = wait_for(
            lambda: (
                (snapshot := devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION))["audio"]["state"] == "recording"
                and bool(snapshot["audio"]["input_device_name"])
                and snapshot
            ),
            timeout=25,
        )
        add_check(report, "foreground_capture_start", "pass" if active else "fail", {
            "state": active["audio"]["state"] if active else "timeout",
            "capture_id_present": bool(active and active["audio"]["capture_id_present"]),
        })
        if active:
            route = {
                "name": active["audio"]["input_device_name"],
                "type": active["audio"]["input_device_type"],
                "source": active["audio"]["input_device_source"],
            }
            route_matches = bool(route["name"]) and (
                args.expected_input is None or route["source"] == args.expected_input
            )
            add_check(report, "actual_input_route", "pass" if route_matches else "fail", {
                "scenario": args.scenario,
                "expected_source": args.expected_input or "any",
                **route,
            })
            samples_before = active["audio"]["captured_samples"]
            time.sleep(2)
            signal = devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)
            audio = signal["audio"]
            add_check(report, "microphone_signal", "pass" if audio["captured_samples"] > samples_before else "fail", {
                "sample_delta": audio["captured_samples"] - samples_before,
                "rms_dbfs": audio["audio_rms_dbfs"],
                "peak_dbfs": audio["audio_peak_dbfs"],
                "input_route": {
                    "name": audio["input_device_name"],
                    "type": audio["input_device_type"],
                    "source": audio["input_device_source"],
                },
                "queue_depth": audio["inference_queue_depth"],
            })
        if args.speech and active:
            speech_before = devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)["audio"]
            if args.speech_source == "device":
                original_music_volume = music_volume(adb)
                adb.shell("cmd", "media_session", "volume", "--stream", "3", "--set", "15")
            for index, phrase in enumerate(("今天天气不错。", "我下午三点要提交测试报告。", "请记录这次安卓语音测试。"), start=1):
                play_speech(adb, phrase, args.speech_source, index)
                time.sleep(5)
            speech_after = wait_for(
                lambda: (snapshot := devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION))["audio"]["vad_segment_count"] > speech_before["vad_segment_count"] and snapshot,
                timeout=30,
            ) or devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)
            audio = speech_after["audio"]
            context_matches = audio["ui_context_count"] == audio["ambient_context"]["chunk_count"]
            add_check(report, "speech_vad_and_context_sync", "pass" if audio["vad_segment_count"] > speech_before["vad_segment_count"] and context_matches else "fail", {
                "vad_delta": audio["vad_segment_count"] - speech_before["vad_segment_count"],
                "ambient_final_delta": audio["ambient_final_count"] - speech_before["ambient_final_count"],
                "rejected_delta": audio["speech_rejected_count"] - speech_before["speech_rejected_count"],
                "backend_chunk_count": audio["ambient_context"]["chunk_count"],
                "ui_context_count": audio["ui_context_count"],
                "queue": audio["event_queue"],
                "audit_types": speech_after["audit_types"],
                "speech_source": args.speech_source,
            })
        else:
            add_check(report, "speech_vad_and_context_sync", "blocked", {"reason": "run again with --speech"})
        if args.background_minutes > 0 and active:
            background_before = devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)
            metrics_before = device_metrics(adb)
            power = adb.shell("dumpsys", "power", check=False)
            screen_was_awake = "mWakefulness=Awake" in power
            if screen_was_awake:
                adb.shell("input", "keyevent", "223")
                screen_turned_off = True
            max_queue_depth = background_before["audio"]["inference_queue_depth"]
            background_error = ""
            background_after = background_before
            for minute in range(1, args.background_minutes + 1):
                time.sleep(60)
                try:
                    devtools, forwarded_port = reconnect_devtools(adb, devtools, forwarded_port)
                    background_after = devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)
                    max_queue_depth = max(max_queue_depth, background_after["audio"]["inference_queue_depth"])
                    print(f"[PROGRESS] background_lock {minute}/{args.background_minutes}", flush=True)
                except Exception as error:
                    background_error = type(error).__name__
                    break
            metrics_after = device_metrics(adb)
            sample_delta = background_after["audio"]["captured_samples"] - background_before["audio"]["captured_samples"]
            expected_samples = args.background_minutes * 60 * 16_000
            passed = (
                not background_error
                and background_after["audio"]["state"] == "recording"
                and sample_delta >= expected_samples * 0.8
                and max_queue_depth < 16
            )
            add_check(report, "locked_background_capture", "pass" if passed else "fail", {
                "minutes": args.background_minutes,
                "sample_delta": sample_delta,
                "expected_samples": expected_samples,
                "max_queue_depth": max_queue_depth,
                "connection_error": background_error,
                "pss_kb_before": metrics_before["pss_kb"],
                "pss_kb_after": metrics_after["pss_kb"],
                "battery_before": metrics_before["battery_level"],
                "battery_after": metrics_after["battery_level"],
                "temperature_before": metrics_before["temperature_tenths_c"],
                "temperature_after": metrics_after["temperature_tenths_c"],
            })
        elif args.background_minutes > 0:
            add_check(report, "locked_background_capture", "fail", {"reason": "capture_not_running"})
        if args.location:
            for permission in location_permissions:
                adb.shell("pm", "grant", PACKAGE, permission, check=False)
            devtools.evaluate("document.querySelector('#location-refresh').click(); true")
            location = wait_for(
                lambda: (snapshot := devtools.evaluate(PUBLIC_SNAPSHOT_EXPRESSION)).get("location") and snapshot,
                timeout=12,
            )
            location_status = (location or {}).get("location") or {}
            status = "pass" if location_status.get("status") == "available" else "blocked" if location_status.get("status") in {"timeout", "unavailable"} else "fail"
            add_check(report, "system_location", status, location_status or {"status": "timeout", "error": "no_device_fix"})
        else:
            add_check(report, "system_location", "blocked", {"reason": "run again with --location"})
        report["final_metrics"] = device_metrics(adb)
    finally:
        if devtools is not None and started_capture and args.stop_after:
            try:
                devtools.evaluate("document.querySelector('#ambient-standby-toggle').click(); true")
                time.sleep(2)
            except Exception:
                pass
            services = adb.shell("dumpsys", "activity", "services", PACKAGE, check=False)
            if ".AudioCaptureService" in services:
                adb.shell(
                    "am",
                    "startservice",
                    "-n",
                    f"{PACKAGE}/.AudioCaptureService",
                    "-a",
                    "com.aiglasses.memoryassistant.action.STOP_CAPTURE",
                    check=False,
                )
                time.sleep(2)
                services = adb.shell("dumpsys", "activity", "services", PACKAGE, check=False)
            released = ".AudioCaptureService" not in services and not device_metrics(adb)["record_audio_running"]
            add_check(report, "microphone_release", "pass" if released else "fail", {"released": released})
        if args.location:
            for permission, was_granted in location_permissions.items():
                action = "grant" if was_granted else "revoke"
                adb.shell("pm", action, PACKAGE, permission, check=False)
        if original_music_volume is not None:
            adb.shell("cmd", "media_session", "volume", "--stream", "3", "--set", str(original_music_volume), check=False)
        if screen_turned_off:
            adb.shell("input", "keyevent", "224", check=False)
        if devtools is not None:
            devtools.close()
        if forwarded_port:
            adb.run("forward", "--remove", f"tcp:{forwarded_port}", check=False)
        paths = write_report(report, args.output_dir)
        print("\n".join(str(path.resolve()) for path in paths), flush=True)


if __name__ == "__main__":
    main()
