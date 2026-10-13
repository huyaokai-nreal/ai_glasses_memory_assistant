"""One new quality-only version; sealed parent requests still consume the same budget."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import run_live_memory_closeout as live


def configure(run):
    protocol = json.loads((run / "protocol.json").read_text())
    if protocol["config"] != live.CONFIG or protocol["limits"] != dict(
        global_requests=640, local_requests=120, per_variant=5, global_attempted_variants=124
    ):
        raise ValueError("Revision protocol configuration mismatch")
    if live.digest(run / "conditions.json") != protocol["conditions_sha256"]:
        raise ValueError("Condition hash mismatch")
    if json.loads((run / "conditions.json").read_text()) != protocol["conditions"]:
        raise ValueError("Condition content mismatch")
    live.RUN = run
    live.CASES = protocol["conditions"]
    if protocol["variants"] != [f'quality/{c["id"]}/after' for c in live.CASES] or len(live.CASES) != 24:
        raise ValueError("Only the frozen 24 new-version variants are allowed")
    return protocol


def ledger(run, protocol):
    prior = live.ROOT / protocol["parent_run"] / "requests.jsonl"
    result = live.Ledger(run, protocol["variants"],
                         prior_ledger=(prior, protocol["parent_ledger_sha256"]), local_limit=120)
    if (result.prior_total, result.prior_variants) != (protocol["parent_requests"], protocol["parent_used_variants"]):
        raise ValueError("Parent budget mismatch")
    return result


def freeze(run, protocol):
    if (run / "freeze.json").exists() or (run / "projects").exists():
        raise ValueError("Source version already frozen; refuse overwrite")
    ledger(run, protocol)
    project = run / "projects" / "after"
    for name in ["ai_glasses_memory_assistant", "static"]:
        shutil.copytree(live.ROOT / name, project / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    hashes = {str(p.relative_to(project)): live.digest(p) for p in project.rglob("*") if p.is_file()}
    live.dump(run / "sources" / "after.json", hashes)
    runners = ["run_live_memory_closeout.py", "run_memory_quality_revision.py", "report_memory_quality_revision.py"]
    tools = run / "tools"
    tools.mkdir()
    for name in runners:
        shutil.copy2(live.ROOT / "scripts" / name, tools / name)
    live.dump(run / "runner-hashes.json", {name: live.digest(tools / name) for name in runners})
    (run / "authority-token").write_text(secrets.token_hex(32))
    os.chmod(run / "authority-token", 0o600)
    live.dump(run / "freeze.json", dict(protocol_sha256=live.digest(run / "protocol.json"),
        config=protocol["config"], variants=protocol["variants"], conditions=live.CASES,
        parent_ledger_sha256=protocol["parent_ledger_sha256"], parent_requests=protocol["parent_requests"],
        source_manifest_sha256=live.digest(run / "sources" / "after.json"),
        runner_manifest_sha256=live.digest(run / "runner-hashes.json"), case_freeze="Previously inspected; not blind",
        baseline_evidence_hashes={name:live.digest(live.ROOT / protocol["parent_run"] / name)
                                 for name in ["quality-chain.json", "answer-review.json", "freeze.json", "sources/after.json"]},
        model_authorized=True, device_authorized=False, real_database_install_authorized=False))


def verify(run):
    frozen = json.loads((run / "freeze.json").read_text())
    if live.digest(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    if live.digest(run / "sources" / "after.json") != frozen["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    for name, expected in json.loads((run / "sources" / "after.json").read_text()).items():
        if live.digest(run / "projects" / "after" / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if live.digest(run / "runner-hashes.json") != frozen["runner_manifest_sha256"]:
        raise ValueError("Frozen runner manifest changed")
    for name, expected in json.loads((run / "runner-hashes.json").read_text()).items():
        if live.digest(run / "tools" / name) != expected or live.digest(live.ROOT / "scripts" / name) != expected:
            raise ValueError("Runner changed after freeze: " + name)
    protocol = json.loads((run / "protocol.json").read_text())
    for name, expected in frozen["baseline_evidence_hashes"].items():
        if live.digest(live.ROOT / protocol["parent_run"] / name) != expected:
            raise ValueError("Baseline evidence changed: " + name)


def quality(run, protocol):
    verify(run)
    for case in live.CASES:
        variant = f'quality/{case["id"]}/after'
        status = live.request("/status")
        if variant in status["used"]:
            continue  # A reserved failed/unknown variant must never be regenerated.
        live.request("/begin", dict(variant=variant))
        folder = run / "quality" / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        error = None
        with (folder / "after.log").open("x") as log:
            worker = subprocess.Popen([sys.executable, __file__, "worker", "--run", str(run), "--case", case["id"]],
                env=live.environment("after", run / "homes" / variant), cwd=run / "projects" / "after",
                stdout=log, stderr=subprocess.STDOUT)
            try:
                exit_code = worker.wait(timeout=390)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
                exit_code = -1
                error = "worker_timeout; stop; do not retry"
        status = live.request("/status")
        if error or status["pending"]:
            live.dump(folder / "stopped.json", dict(error=error, pending=status["pending"]))
            raise RuntimeError("Unresolved worker/request; no next variant")
        raw = json.loads((folder / "after.json").read_text()) if (folder / "after.json").exists() else {}
        if raw.get("background_pending") or exit_code:
            live.dump(folder / "stopped.json", dict(exit_code=exit_code, background_pending=raw.get("background_pending")))
            raise RuntimeError("Worker did not drain; no next variant")
        live.request("/end", dict(exit_code=exit_code))
        print(variant, "requests", status["counts"].get(variant, 0), "cumulative", status["total"], flush=True)


def worker(case_id):
    verify(live.RUN)
    sys.path.insert(0, str(live.RUN / "projects" / "after"))
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
    from ai_glasses_memory_assistant.acceptance_runtime import prepare
    service = GlassesChatService()
    case = next(c for c in live.CASES if c["id"] == case_id)
    output = dict(id=case_id, arm="after", condition=case)
    try:
        output["seed"] = prepare(service, dict(user_id="acceptance_" + case_id.replace("-", "_"),
                                             condition=case, projection=True))
        output["result"] = service.chat(case["question"], user_id=output["seed"]["user_id"], defer_memory_writes=False)
    except Exception as exc:
        output["error"] = dict(type=type(exc).__name__, detail=str(exc))
    finally:
        service.close(timeout=65)
        output["background_pending"] = [key for key, value in service._background_workers.items() if value.is_alive()]
        output["audit"] = [json.loads(line) for line in service.audit_path.read_text().splitlines()] if service.audit_path.exists() else []
        live.dump(live.RUN / "quality" / case_id / "after.json", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "authority", "quality", "worker", "verify"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = configure(run)
    if args.action == "freeze": freeze(run, protocol)
    elif args.action == "verify": verify(run)
    elif args.action == "authority":
        verify(run)
        live.authority_server(ledger(run, protocol))
    elif args.action == "quality": quality(run, protocol)
    elif args.action == "worker": worker(args.case)


if __name__ == "__main__":
    main()
