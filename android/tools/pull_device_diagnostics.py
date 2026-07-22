#!/usr/bin/env python3
"""Pull one sanitized Android diagnostic snapshot over ADB for Codex analysis."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from run_device_acceptance import Adb, DevToolsSocket, connect_devtools


PACKAGE = "com.aiglasses.memoryassistant.demo"
ACTIVITY = f"{PACKAGE}/.MainActivity"
TERMINAL_MEMORY_JOB_STATUSES = {"saved", "skipped", "rejected", "failed"}
SAFE_SERIAL = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "captures" / "diagnostics"


class DeviceClient(Protocol):
    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]: ...

    def shell(self, *args: str, check: bool = True) -> str: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="ADB serial. Required only when multiple devices are connected.")
    parser.add_argument("--adb", default="adb", help="ADB executable")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


def connected_devices(adb: str) -> list[dict[str, str]]:
    result = subprocess.run([adb, "devices", "-l"], check=True, text=True, capture_output=True)
    devices: list[dict[str, str]] = []
    for raw_line in result.stdout.splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] != "device":
            continue
        attributes = {"serial": parts[0]}
        for item in parts[2:]:
            if ":" in item:
                key, value = item.split(":", 1)
                attributes[key] = value
        devices.append(attributes)
    return devices


def select_device(devices: list[dict[str, str]], requested_serial: str | None) -> dict[str, str]:
    if requested_serial:
        for device in devices:
            if device["serial"] == requested_serial:
                return device
        raise RuntimeError(f"ADB device is not ready: {requested_serial}")
    if not devices:
        raise RuntimeError("no ready ADB device found")
    if len(devices) > 1:
        choices = ", ".join(
            f"{item['serial']} ({item.get('model', 'unknown')})" for item in devices
        )
        raise RuntimeError(f"multiple ADB devices are connected; pass --serial. Choices: {choices}")
    return devices[0]


def evaluate_json(devtools: DevToolsSocket, expression: str) -> dict[str, Any]:
    value = devtools.evaluate(expression)
    if not isinstance(value, dict):
        raise RuntimeError("Android WebView returned an invalid diagnostic payload")
    return value


def read_runtime_state(devtools: DevToolsSocket) -> dict[str, Any]:
    return evaluate_json(
        devtools,
        """(async () => {
          const audio = JSON.parse(window.AiGlassesAndroid.audioStatus());
          const jobId = String(audio.last_stop_memory_job_id || '');
          let memoryJob = null;
          if (jobId) {
            const response = await fetch(`/api/memory/jobs?user_id=${encodeURIComponent(state.userId)}&job_id=${encodeURIComponent(jobId)}`);
            if (response.ok) memoryJob = (await response.json()).job || null;
          }
          return {audio, memory_job: memoryJob};
        })()""",
    )


def wait_for_capture_stop(
    devtools: DevToolsSocket,
    *,
    capture_id: str,
    timeout_seconds: float,
    poll_interval: float = 0.5,
) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    last = read_runtime_state(devtools)
    while time.monotonic() < deadline:
        audio = last.get("audio") or {}
        queue = audio.get("device_event_queue") or {}
        stopped_capture = str(audio.get("last_stopped_capture_id") or "")
        stop_status = str(audio.get("last_stop_status") or "")
        if stop_status == "failed" and stopped_capture == capture_id:
            detail = str(audio.get("last_stop_error") or "unknown stop failure")
            raise RuntimeError(f"Android capture stop failed: {detail}")
        stopped = (
            str(audio.get("state") or "") == "idle"
            and stopped_capture == capture_id
            and stop_status == "completed"
        )
        queue_drained = int(queue.get("pending") or 0) == 0 and int(queue.get("running") or 0) == 0
        memory_job = last.get("memory_job")
        job_id = str(audio.get("last_stop_memory_job_id") or "")
        job_status = str((memory_job or {}).get("status") or audio.get("last_stop_memory_job_status") or "")
        job_terminal = not job_id or job_status in TERMINAL_MEMORY_JOB_STATUSES
        if stopped and queue_drained and job_terminal:
            return last, ""
        time.sleep(poll_interval)
        last = read_runtime_state(devtools)
    return last, "capture_stop_or_memory_job_timeout"


def create_remote_snapshot(devtools: DevToolsSocket) -> dict[str, Any]:
    return evaluate_json(
        devtools,
        """(() => JSON.parse(window.AiGlassesAndroid.createAdbDiagnosticSnapshot()))()""",
    )


def delete_remote_snapshot(devtools: DevToolsSocket, relative_path: str) -> bool:
    expression = (
        "window.AiGlassesAndroid.deleteAdbDiagnosticSnapshot("
        + json.dumps(relative_path, ensure_ascii=True)
        + ")"
    )
    return bool(devtools.evaluate(expression))


def remove_remote_snapshot(
    devtools: DevToolsSocket,
    adb: DeviceClient,
    relative_path: str,
) -> bool:
    if not re.fullmatch(
        r"cache/adb-diagnostics/ai-glasses-diagnostic-[0-9a-f]{32}\.zip",
        relative_path,
    ):
        return False
    try:
        if delete_remote_snapshot(devtools, relative_path):
            return True
    except Exception:
        pass
    adb.shell("run-as", PACKAGE, "rm", "-f", relative_path, check=False)
    return True


def pull_remote_file(adb: str, serial: str, relative_path: str, destination: Path) -> None:
    if not re.fullmatch(
        r"cache/adb-diagnostics/ai-glasses-diagnostic-[0-9a-f]{32}\.zip",
        relative_path,
    ):
        raise ValueError("Android returned an unsafe diagnostic snapshot path")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    command = [adb, "-s", serial, "exec-out", "run-as", PACKAGE, "cat", relative_path]
    try:
        with temporary.open("wb") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ADB diagnostic pull failed: {detail or process.returncode}")
        if temporary.stat().st_size == 0:
            raise RuntimeError("ADB diagnostic pull returned an empty file")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_archive(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "diagnostics.json" not in names:
                raise ValueError("diagnostic archive is missing diagnostics.json")
            damaged = archive.testzip()
            if damaged:
                raise ValueError(f"diagnostic archive contains a damaged entry: {damaged}")
            for name in names:
                safe_archive_destination(Path("/tmp/diagnostic-root"), name)
            return names
    except zipfile.BadZipFile as exc:
        raise ValueError("Android diagnostic snapshot is not a valid ZIP file") from exc


def safe_archive_destination(root: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"unsafe diagnostic archive member: {member_name}")
    destination = root.joinpath(*member.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_root and resolved_root not in resolved_destination.parents:
        raise ValueError(f"unsafe diagnostic archive member: {member_name}")
    return destination


def extract_archive(path: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = safe_archive_destination(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(info.filename)
    return extracted


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_handoff(path: Path, collection: dict[str, Any]) -> None:
    audio = collection.get("final_state", {}).get("audio") or {}
    job = collection.get("final_state", {}).get("memory_job") or {}
    lines = [
        "# Android Diagnostic Handoff",
        "",
        f"- Collection status: `{collection['collection_status']}`",
        f"- Device: `{collection['device_serial']}` / `{collection.get('device_model', 'unknown')}`",
        f"- Capture: `{collection.get('capture_id') or 'none'}`",
        f"- Capture stop: `{audio.get('last_stop_status') or 'not_required'}`",
        f"- Memory job: `{audio.get('last_stop_memory_job_id') or 'none'}` / `{job.get('status') or audio.get('last_stop_memory_job_status') or 'none'}`",
        f"- Timeout stage: `{collection.get('timeout_stage') or 'none'}`",
        f"- Started: `{collection['started_at']}`",
        f"- Finished: `{collection['finished_at']}`",
        "",
        "## Evidence",
        "",
        "- `audit/chat_audit.redacted.jsonl`: audio final, chat, recall, write gate and failure evidence.",
        "- `database/timeline.db`: captures, chunks, raw turns, device audio events and memory jobs.",
        "- `database/events.db`: sanitized structured long-term memories.",
        "- `database/sessions.db`: sanitized model session messages when present.",
        "- `diagnostics.json`: device, model, queue and redaction summary.",
        "",
        "## Codex Prompt",
        "",
        "请使用 ai-glasses-audit-debug 工作流分析本目录中最近一次安卓测试，按证据、诊断、修复方向、验证方式和不确定性说明具体问题。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def public_collection_state(state: dict[str, Any]) -> dict[str, Any]:
    audio = state.get("audio") if isinstance(state.get("audio"), dict) else {}
    job = state.get("memory_job") if isinstance(state.get("memory_job"), dict) else {}
    audio_keys = (
        "state",
        "running",
        "capture_id",
        "vad_segment_count",
        "ambient_final_count",
        "speech_rejected_count",
        "last_final_at_ms",
        "network_online",
        "interaction_state",
        "inference_queue_depth",
        "model_state",
        "model_version",
        "last_error",
        "device_event_queue",
        "ambient_context",
        "last_stopped_capture_id",
        "last_stop_status",
        "last_stop_memory_job_id",
        "last_stop_memory_job_status",
        "last_stop_error",
    )
    job_keys = (
        "job_id",
        "status",
        "mode",
        "candidate_count",
        "saved_count",
        "rejected_count",
        "rejected_reasons",
        "error_type",
    )
    return {
        "audio": {key: audio[key] for key in audio_keys if key in audio},
        "memory_job": {key: job[key] for key in job_keys if key in job},
    }


def collection_directory(output_dir: Path, serial: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe_serial = SAFE_SERIAL.sub("_", serial).strip("._") or "device"
    return output_dir.resolve() / f"{stamp}-{safe_serial}"


def collect(args: argparse.Namespace) -> tuple[Path, int]:
    device = select_device(connected_devices(args.adb), args.serial)
    serial = device["serial"]
    adb = Adb(args.adb, serial)
    output_dir = collection_directory(args.output_dir, serial)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "bundle.zip"
    devtools: DevToolsSocket | None = None
    forwarded_port = ""
    remote_path = ""
    started_at = datetime.now(timezone.utc).isoformat()
    initial_state: dict[str, Any] = {}
    final_state: dict[str, Any] = {}
    capture_id = ""
    timeout_stage = ""
    try:
        adb.shell("am", "start", "-n", ACTIVITY)
        deadline = time.monotonic() + min(max(float(args.timeout_seconds), 5.0), 30.0)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                devtools, forwarded_port = connect_devtools(adb)
                break
            except Exception as error:
                last_error = error
                time.sleep(0.5)
        if devtools is None:
            raise RuntimeError(f"unable to connect to debuggable Android WebView: {last_error}")
        initial_state = read_runtime_state(devtools)
        initial_audio = initial_state.get("audio") or {}
        capture_id = str(initial_audio.get("capture_id") or "")
        if bool(initial_audio.get("running")):
            devtools.evaluate("window.AiGlassesAndroid.stopAmbient()")
            final_state, timeout_stage = wait_for_capture_stop(
                devtools,
                capture_id=capture_id,
                timeout_seconds=float(args.timeout_seconds),
            )
        else:
            final_state = initial_state
        snapshot = create_remote_snapshot(devtools)
        remote_path = str(snapshot.get("relative_path") or "")
        pull_remote_file(args.adb, serial, remote_path, bundle_path)
        validate_archive(bundle_path)
        extracted = extract_archive(bundle_path, output_dir)
        finished_at = datetime.now(timezone.utc).isoformat()
        collection = {
            "schema": "android_diagnostic_pull.v1",
            "collection_status": "partial" if timeout_stage else "complete",
            "timeout_stage": timeout_stage,
            "device_serial": serial,
            "device_model": device.get("model", ""),
            "capture_id": capture_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "initial_state": public_collection_state(initial_state),
            "final_state": public_collection_state(final_state),
            "snapshot": snapshot,
            "bundle_size_bytes": bundle_path.stat().st_size,
            "extracted_files": extracted,
        }
        write_json(output_dir / "collection.json", collection)
        write_handoff(output_dir / "codex_handoff.md", collection)
        latest = {
            "schema": "android_diagnostic_latest.v1",
            "collection_status": collection["collection_status"],
            "path": str(output_dir),
            "collection_file": str(output_dir / "collection.json"),
            "handoff_file": str(output_dir / "codex_handoff.md"),
            "updated_at": finished_at,
        }
        args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir.resolve() / "latest.json", latest)
        return output_dir, 2 if timeout_stage else 0
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise
    finally:
        if devtools is not None and remote_path:
            remove_remote_snapshot(devtools, adb, remote_path)
        if devtools is not None:
            devtools.close()
        if forwarded_port:
            adb.run("forward", "--remove", f"tcp:{forwarded_port}", check=False)


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be greater than zero")
    try:
        output_dir, exit_code = collect(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(output_dir)
    print(f"Codex: 分析 {output_dir / 'codex_handoff.md'} 对应的最近一次安卓测试，定位具体问题。")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
