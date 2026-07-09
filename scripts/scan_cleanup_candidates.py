#!/usr/bin/env python3
"""Read-only cleanup candidate scanner for this repository."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CODE_PATTERNS = (
    "*.py",
    "*.js",
    "*.html",
    "*.css",
    "*.md",
    "*.json",
    "*.jsonl",
    "*.yaml",
    "*.yml",
)

HELPER_MODULES = {
    "audio_processing",
    "capture_helpers",
    "conversation_candidate_helpers",
    "conversation_helpers",
    "document_helpers",
    "import_helpers",
    "memory_job_helpers",
}

PROTECTED_POLICIES = (
    {
        "scope": "runtime_data",
        "paths": ["data/", "reports/", "*.db", "*.sqlite", "*.jsonl"],
        "reason": "may contain user data, eval evidence, or audit/runtime state",
    },
    {
        "scope": "secrets_and_local_config",
        "paths": [".env", "certs/", "*.pem", "*.key"],
        "reason": "may contain credentials or local HTTPS material",
    },
    {
        "scope": "public_contracts",
        "paths": ["ai_glasses_memory_assistant/server.py", "server.py", "static/"],
        "reason": "HTTP/API and frontend behavior must not be changed by cleanup scans",
    },
    {
        "scope": "memory_privacy_core",
        "paths": [
            "ai_glasses_memory_assistant/intent_policy.py",
            "ai_glasses_memory_assistant/memory_store.py",
            "ai_glasses_memory_assistant/timeline_store.py",
            "ai_glasses_memory_assistant/privacy_filter.py",
        ],
        "reason": "memory gate, schema, evidence, audit, and privacy changes require explicit review",
    },
)

ROOT_PLANNING_FILES = ("task_plan.md", "findings.md", "progress.md")


@dataclass(frozen=True)
class CleanupItem:
    path: str
    reason: str
    line: int | None = None
    symbol: str | None = None
    detail: str | None = None
    python_reference_count: int | None = None
    python_reference_files: list[str] | None = None


def run_git(repo: Path, *args: str) -> list[str]:
    try:
        output = subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return output.splitlines()


def is_repo_root(path: Path) -> bool:
    return (
        (path / "ai_glasses_memory_assistant" / "agent_bridge.py").exists()
        and (path / "PLANS.md").exists()
    )


def find_repo_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if is_repo_root(candidate):
            return candidate
    raise SystemExit(f"Not an ai_glasses_memory_assistant repository root or child: {start}")


def git_files(repo: Path, patterns: Iterable[str]) -> list[Path]:
    lines = run_git(repo, "ls-files", *patterns)
    return [repo / line for line in lines]


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def relative_path(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def safe_artifacts(repo: Path) -> list[CleanupItem]:
    items: list[CleanupItem] = []
    for root, dirs, files in os.walk(repo):
        root_path = Path(root)
        if ".git" in dirs:
            dirs.remove(".git")
        for dirname in list(dirs):
            if dirname in {"__pycache__", ".pytest_cache"}:
                path = root_path / dirname
                items.append(CleanupItem(
                    path=relative_path(repo, path),
                    reason="regenerable local cache; safe to remove after confirming no process is using it",
                ))
                dirs.remove(dirname)
        for filename in files:
            if filename == ".DS_Store" or filename.endswith(".pyc"):
                path = root_path / filename
                items.append(CleanupItem(
                    path=relative_path(repo, path),
                    reason="regenerable local artifact",
                ))
    return sorted(items, key=lambda item: item.path)


def root_planning_candidates(repo: Path) -> list[CleanupItem]:
    items: list[CleanupItem] = []
    for filename in ROOT_PLANNING_FILES:
        path = repo / filename
        if path.exists():
            items.append(CleanupItem(
                path=filename,
                reason="temporary planning file; merge durable content into PLANS.md/docs before deleting",
            ))
    return items


def helper_call_name(return_node: ast.Return) -> str | None:
    value = return_node.value
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if not isinstance(func, ast.Attribute):
        return None
    owner = func.value
    if not isinstance(owner, ast.Name) or owner.id not in HELPER_MODULES:
        return None
    return f"{owner.id}.{func.attr}"


def python_reference_index(repo: Path) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for path in sorted(repo.rglob("*.py")):
        rel = relative_path(repo, path)
        if rel.startswith(".git/") or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                entry = references.setdefault(node.attr, {"count": 0, "files": set()})
                entry["count"] += 1
                entry["files"].add(rel)
            elif isinstance(node, ast.Name):
                entry = references.setdefault(node.id, {"count": 0, "files": set()})
                entry["count"] += 1
                entry["files"].add(rel)
    return {
        name: {
            "count": int(entry["count"]),
            "files": sorted(entry["files"]),
        }
        for name, entry in references.items()
    }


def thin_helper_wrappers(repo: Path) -> list[CleanupItem]:
    path = repo / "ai_glasses_memory_assistant" / "agent_bridge.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    reference_index = python_reference_index(repo)
    items: list[CleanupItem] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "GlassesChatService":
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = [stmt for stmt in member.body if not isinstance(stmt, ast.Expr)]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            helper_call = helper_call_name(body[0])
            if not helper_call:
                continue
            items.append(CleanupItem(
                path=relative_path(repo, path),
                line=member.lineno,
                symbol=member.name,
                detail=helper_call,
                python_reference_count=reference_index.get(member.name, {}).get("count", 0),
                python_reference_files=reference_index.get(member.name, {}).get("files", []),
                reason="thin helper wrapper; review call sites before removing or inlining",
            ))
    return items


def largest_tracked_files(repo: Path, *, limit: int) -> list[dict[str, Any]]:
    files = git_files(repo, CODE_PATTERNS)
    rows = sorted(
        (
            {
                "path": relative_path(repo, path),
                "lines": count_lines(path),
            }
            for path in files
        ),
        key=lambda item: item["lines"],
        reverse=True,
    )
    return rows[:limit]


def grouped_thin_helper_wrappers(wrappers: list[CleanupItem]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "zero_python_refs": [],
        "internal_only_python_refs": [],
        "test_referenced": [],
    }
    for wrapper in wrappers:
        item = asdict(wrapper)
        files = wrapper.python_reference_files or []
        if int(wrapper.python_reference_count or 0) <= 0:
            groups["zero_python_refs"].append(item)
        elif any(path.startswith("tests/") for path in files):
            groups["test_referenced"].append(item)
        else:
            groups["internal_only_python_refs"].append(item)
    return groups


def build_report(repo: Path, *, largest_limit: int) -> dict[str, Any]:
    safe = safe_artifacts(repo)
    planning = root_planning_candidates(repo)
    wrappers = thin_helper_wrappers(repo)
    wrapper_groups = grouped_thin_helper_wrappers(wrappers)
    status = run_git(repo, "status", "--short")
    return {
        "repo": str(repo),
        "read_only": True,
        "summary": {
            "git_status": status or ["clean"],
            "safe_artifact_count": len(safe),
            "review_required_count": len(planning) + len(wrappers),
            "zero_reference_wrapper_count": len(wrapper_groups["zero_python_refs"]),
            "test_referenced_wrapper_count": len(wrapper_groups["test_referenced"]),
            "protected_policy_count": len(PROTECTED_POLICIES),
        },
        "safe_artifacts": [asdict(item) for item in safe],
        "review_required": {
            "root_planning_files": [asdict(item) for item in planning],
            "thin_helper_wrappers": [asdict(item) for item in wrappers],
        },
        "review_groups": {
            "thin_helper_wrappers": wrapper_groups,
        },
        "protected_policies": list(PROTECTED_POLICIES),
        "largest_files": largest_tracked_files(repo, limit=largest_limit),
    }


def print_markdown(report: dict[str, Any]) -> None:
    print("# Cleanup Candidate Scan")
    print()
    print(f"Repo: `{report['repo']}`")
    print("Mode: read-only; this script does not delete or modify files.")
    print()

    print("## Summary")
    for key, value in report["summary"].items():
        if key == "git_status":
            print(f"- git_status: {', '.join(f'`{line}`' for line in value)}")
        else:
            print(f"- {key}: {value}")
    print()

    print("## Safe Artifacts")
    safe_items = report["safe_artifacts"]
    if not safe_items:
        print("- none")
    for item in safe_items:
        print(f"- `{item['path']}`: {item['reason']}")
    print()

    print("## Review Required")
    planning_items = report["review_required"]["root_planning_files"]
    print("Root planning files:")
    if not planning_items:
        print("- none")
    for item in planning_items:
        print(f"- `{item['path']}`: {item['reason']}")
    print()
    print("Thin helper wrappers:")
    wrapper_groups = report["review_groups"]["thin_helper_wrappers"]
    group_titles = (
        ("zero_python_refs", "Zero Python references"),
        ("internal_only_python_refs", "Internal-only Python references"),
        ("test_referenced", "Referenced by tests"),
    )
    if not any(wrapper_groups.values()):
        print("- none")
    for group_key, title in group_titles:
        items = wrapper_groups[group_key]
        print(f"\n{title}:")
        if not items:
            print("- none")
            continue
        for item in items:
            reference_files = item.get("python_reference_files") or []
            reference_note = f"; python_refs={item['python_reference_count']}"
            if reference_files:
                reference_note += f" in {', '.join(f'`{path}`' for path in reference_files[:4])}"
                if len(reference_files) > 4:
                    reference_note += f", ... {len(reference_files) - 4} more"
            print(
                f"- `{item['path']}:{item['line']}` `{item['symbol']}` -> "
                f"`{item['detail']}`: {item['reason']}{reference_note}"
            )
    print()

    print("## Protected Policies")
    for policy in report["protected_policies"]:
        paths = ", ".join(f"`{path}`" for path in policy["paths"])
        print(f"- {policy['scope']}: {paths} - {policy['reason']}")
    print()

    print("## Largest Tracked Files")
    for item in report["largest_files"]:
        print(f"- {item['lines']:5d} `{item['path']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only cleanup candidate scanner.")
    parser.add_argument("--repo", default=".", help="Repository root or a path inside it.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--largest-limit", type=int, default=20, help="Number of largest files to show.")
    args = parser.parse_args()

    repo = find_repo_root(Path(args.repo))
    report = build_report(repo, largest_limit=max(args.largest_limit, 0))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
