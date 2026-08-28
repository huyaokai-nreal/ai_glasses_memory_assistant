#!/usr/bin/env python3
"""Pull one complete Android usage snapshot over ADB for local agent analysis.

The app creates a temporary, debug-only ZIP in its private cache. This tool stops
continuous capture, waits for pending audio and memory work, pulls and verifies that
ZIP, then removes only the temporary device ZIP. It never removes Android app data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
USAGE_SNAPSHOT_PATH = re.compile(
    r"cache/adb-usage-snapshots/ai-glasses-usage-[0-9a-f]{32}\.zip"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "captures" / "usage-data"


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
        parts = raw_line.strip().split()
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
        choices = ", ".join(f"{item['serial']} ({item.get('model', 'unknown')})" for item in devices)
        raise RuntimeError(f"multiple ADB devices are connected; pass --serial. Choices: {choices}")
    return devices[0]


def evaluate_json(devtools: DevToolsSocket, expression: str) -> dict[str, Any]:
    value = devtools.evaluate(expression)
    if not isinstance(value, dict):
        raise RuntimeError("Android WebView returned an invalid usage snapshot payload")
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
            raise RuntimeError(f"Android capture stop failed: {audio.get('last_stop_error') or 'unknown stop failure'}")
        stopped = (
            str(audio.get("state") or "") == "idle"
            and stopped_capture == capture_id
            and stop_status == "completed"
        )
        queue_drained = int(queue.get("pending") or 0) == 0 and int(queue.get("running") or 0) == 0
        memory_job = last.get("memory_job") or {}
        job_id = str(audio.get("last_stop_memory_job_id") or "")
        job_status = str(memory_job.get("status") or audio.get("last_stop_memory_job_status") or "")
        if stopped and queue_drained and (not job_id or job_status in TERMINAL_MEMORY_JOB_STATUSES):
            return last, ""
        time.sleep(poll_interval)
        last = read_runtime_state(devtools)
    return last, "capture_stop_or_memory_job_timeout"


def create_remote_snapshot(devtools: DevToolsSocket) -> dict[str, Any]:
    return evaluate_json(devtools, "(() => JSON.parse(window.AiGlassesAndroid.createAdbUsageSnapshot()))()")


def wait_for_discussion_archive(
    devtools: DevToolsSocket,
    *,
    timeout_seconds: float,
    poll_interval: float = 0.5,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = evaluate_json(
            devtools,
            "(() => JSON.parse(window.AiGlassesAndroid.prepareAdbUsageSnapshot()))()",
        )
        if bool(last.get("ready")):
            return last
        time.sleep(poll_interval)
    pending = len(last.get("pending_slice_ids") or [])
    raise RuntimeError(f"Android discussion archive did not finish before timeout ({pending} pending slices)")


def delete_remote_snapshot(devtools: DevToolsSocket, relative_path: str) -> bool:
    return bool(devtools.evaluate(
        "window.AiGlassesAndroid.deleteAdbUsageSnapshot(" + json.dumps(relative_path, ensure_ascii=True) + ")"
    ))


def remove_remote_snapshot(devtools: DevToolsSocket, adb: DeviceClient, relative_path: str) -> bool:
    if not USAGE_SNAPSHOT_PATH.fullmatch(relative_path):
        return False
    try:
        if delete_remote_snapshot(devtools, relative_path):
            return True
    except Exception:
        pass
    adb.shell("run-as", PACKAGE, "rm", "-f", relative_path, check=False)
    return True


def pull_remote_file(adb: str, serial: str, relative_path: str, destination: Path) -> None:
    if not USAGE_SNAPSHOT_PATH.fullmatch(relative_path):
        raise ValueError("Android returned an unsafe usage snapshot path")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    command = [adb, "-s", serial, "exec-out", "run-as", PACKAGE, "cat", relative_path]
    try:
        with temporary.open("wb") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ADB usage snapshot pull failed: {detail or process.returncode}")
        if temporary.stat().st_size == 0:
            raise RuntimeError("ADB usage snapshot pull returned an empty file")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_archive_destination(root: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"unsafe usage snapshot archive member: {member_name}")
    destination = root.joinpath(*member.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_root and resolved_root not in resolved_destination.parents:
        raise ValueError(f"unsafe usage snapshot archive member: {member_name}")
    return destination


def validate_archive(path: Path) -> list[str]:
    required = {"manifest.json", "feedback_index.json", "analysis_request.md"}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            missing = required.difference(names)
            if missing:
                raise ValueError("usage snapshot archive is missing " + ", ".join(sorted(missing)))
            damaged = archive.testzip()
            if damaged:
                raise ValueError(f"usage snapshot archive contains a damaged entry: {damaged}")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema") != "ai_glasses_usage_snapshot.v1":
                raise ValueError("usage snapshot archive has an unsupported manifest")
            for name, details in manifest.get("files", {}).items():
                if name not in names:
                    raise ValueError(f"usage snapshot manifest references a missing file: {name}")
                if hashlib.sha256(archive.read(name)).hexdigest() != details.get("sha256"):
                    raise ValueError(f"usage snapshot checksum mismatch: {name}")
            for name in names:
                safe_archive_destination(Path("/tmp/usage-snapshot-root"), name)
            return names
    except zipfile.BadZipFile as exc:
        raise ValueError("Android usage snapshot is not a valid ZIP file") from exc


def make_private(path: Path, *, directory: bool = False) -> None:
    try:
        os.chmod(path, 0o700 if directory else 0o600)
    except OSError:
        # The export remains local even on filesystems which do not support POSIX permissions.
        pass


def extract_archive(path: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = safe_archive_destination(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                make_private(target, directory=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            make_private(target.parent, directory=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            make_private(target)
            extracted.append(info.filename)
    return extracted


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_private(temporary)
    temporary.replace(path)
    make_private(path)


def public_collection_state(state: dict[str, Any]) -> dict[str, Any]:
    audio = state.get("audio") if isinstance(state.get("audio"), dict) else {}
    job = state.get("memory_job") if isinstance(state.get("memory_job"), dict) else {}
    audio_keys = (
        "state", "running", "capture_id", "vad_segment_count", "ambient_final_count",
        "speech_rejected_count", "last_final_at_ms", "network_online", "inference_queue_depth",
        "model_state", "model_version", "last_error", "device_event_queue", "ambient_context",
        "last_stopped_capture_id", "last_stop_status", "last_stop_memory_job_id",
        "last_stop_memory_job_status", "last_stop_error",
    )
    job_keys = ("job_id", "status", "mode", "candidate_count", "saved_count", "rejected_count", "rejected_reasons", "error_type")
    return {
        "audio": {key: audio[key] for key in audio_keys if key in audio},
        "memory_job": {key: job[key] for key in job_keys if key in job},
    }


def collection_directory(output_dir: Path, serial: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe_serial = SAFE_SERIAL.sub("_", serial).strip("._") or "device"
    return output_dir.resolve() / f"{stamp}-{safe_serial}"


def collect(args: argparse.Namespace) -> Path:
    device = select_device(connected_devices(args.adb), args.serial)
    serial = device["serial"]
    adb = Adb(args.adb, serial)
    output_dir = collection_directory(args.output_dir, serial)
    output_dir.mkdir(parents=True, exist_ok=False)
    make_private(output_dir, directory=True)
    bundle_path = output_dir / "bundle.zip"
    devtools: DevToolsSocket | None = None
    forwarded_port = ""
    remote_path = ""
    initial_state: dict[str, Any] = {}
    final_state: dict[str, Any] = {}
    capture_id = ""
    started_at = datetime.now(timezone.utc).isoformat()
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
                devtools, capture_id=capture_id, timeout_seconds=float(args.timeout_seconds)
            )
            if timeout_stage:
                raise RuntimeError("Android capture stop or memory archive did not finish before timeout; snapshot was not created")
        else:
            final_state = initial_state
        discussion_archive = wait_for_discussion_archive(
            devtools,
            timeout_seconds=float(args.timeout_seconds),
        )
        snapshot = create_remote_snapshot(devtools)
        remote_path = str(snapshot.get("relative_path") or "")
        pull_remote_file(args.adb, serial, remote_path, bundle_path)
        make_private(bundle_path)
        extracted = extract_archive(bundle_path, output_dir)
        finished_at = datetime.now(timezone.utc).isoformat()
        collection = {
            "schema": "android_usage_snapshot_pull.v1",
            "collection_status": "complete",
            "device_serial": serial,
            "device_model": device.get("model", ""),
            "capture_id": capture_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "initial_state": public_collection_state(initial_state),
            "final_state": public_collection_state(final_state),
            "discussion_archive": discussion_archive,
            "snapshot": snapshot,
            "bundle_size_bytes": bundle_path.stat().st_size,
            "extracted_files": extracted,
        }
        write_json(output_dir / "collection.json", collection)
        output_root = args.output_dir.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        make_private(output_root, directory=True)
        write_json(output_root / "latest.json", {
            "schema": "android_usage_snapshot_latest.v1",
            "collection_status": "complete",
            "path": str(output_dir),
            "analysis_request": str(output_dir / "analysis_request.md"),
            "updated_at": finished_at,
        })
        return output_dir
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
        output_dir = collect(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(output_dir)
    print(f"把此目录路径发给 Codex：{output_dir}")


if __name__ == "__main__":
    main()
