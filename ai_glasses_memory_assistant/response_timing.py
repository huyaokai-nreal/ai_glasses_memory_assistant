from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any


class AssistantResponseTiming:
    _CALLBACK_NAMES = (
        "step_callback",
        "tool_progress_callback",
        "tool_start_callback",
        "tool_complete_callback",
    )

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.perf_counter
        self._started_at: float | None = None
        self._api_calls: list[dict[str, Any]] = []
        self._current_api: dict[str, Any] | None = None
        self._tool_calls: list[dict[str, Any]] = []
        self._active_tools: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        self._started_at = self._clock()

    # 挂接模型客户端回调，用于拆分主模型等待和工具调用耗时。
    def install(self, agent: Any) -> dict[str, Any]:
        previous = {name: getattr(agent, name, None) for name in self._CALLBACK_NAMES}

        def step_callback(api_call_count: int, previous_tools: list[dict[str, Any]]) -> None:
            self.record_api_call_start(api_call_count, previous_tools)
            self._call_previous(previous["step_callback"], api_call_count, previous_tools)

        def tool_progress_callback(event: str, name: str, preview: Any, args: Any, **kwargs: Any) -> None:
            self.record_tool_progress(event, name, **kwargs)
            self._call_previous(previous["tool_progress_callback"], event, name, preview, args, **kwargs)

        def tool_start_callback(tool_call_id: str, name: str, args: dict[str, Any]) -> None:
            self.record_tool_start(tool_call_id, name, args)
            self._call_previous(previous["tool_start_callback"], tool_call_id, name, args)

        def tool_complete_callback(tool_call_id: str, name: str, args: dict[str, Any], result: Any) -> None:
            self.record_tool_complete(tool_call_id, name, args, result)
            self._call_previous(previous["tool_complete_callback"], tool_call_id, name, args, result)

        agent.step_callback = step_callback
        agent.tool_progress_callback = tool_progress_callback
        agent.tool_start_callback = tool_start_callback
        agent.tool_complete_callback = tool_complete_callback
        return previous

    @classmethod
    def restore(cls, agent: Any, previous: dict[str, Any]) -> None:
        for name in cls._CALLBACK_NAMES:
            setattr(agent, name, previous.get(name))

    def record_api_call_start(self, api_call_count: int, previous_tools: list[dict[str, Any]] | None) -> None:
        self._close_current_api("next_api_call_started")
        previous_tool_summaries = self._previous_tool_summaries(previous_tools)
        entry = {
            "index": api_call_count,
            "start_offset_seconds": self._elapsed(),
            "seconds": None,
            "ended_by": None,
            "phase": "initial_model_request" if not previous_tool_summaries else "model_after_tool_results",
            "previous_tool_result_count": len(previous_tool_summaries),
            "previous_tools": previous_tool_summaries,
        }
        self._api_calls.append(entry)
        self._current_api = entry

    def record_tool_start(self, tool_call_id: str, name: str, args: dict[str, Any] | None) -> None:
        self._close_current_api("tool_call_started")
        key = str(tool_call_id or f"{name}:{len(self._tool_calls) + 1}")
        entry = {
            "call_id": key,
            "name": name,
            "start_offset_seconds": self._elapsed(),
            "_started_at": self._clock(),
            "seconds": None,
            "argument_keys": self._argument_keys(args),
            "completed": False,
        }
        self._active_tools[key] = entry
        self._tool_calls.append(entry)

    def record_tool_progress(self, event: str, name: str, **kwargs: Any) -> None:
        if event != "tool.completed":
            return
        duration = kwargs.get("duration")
        if not isinstance(duration, (int, float)):
            return
        entry = self._latest_active_tool_by_name(name)
        if entry is None:
            return
        entry["seconds"] = round(float(duration), 6)
        if "is_error" in kwargs:
            entry["is_error"] = bool(kwargs.get("is_error"))

    def record_tool_complete(
        self,
        tool_call_id: str,
        name: str,
        args: dict[str, Any] | None,
        result: Any,
    ) -> None:
        key = str(tool_call_id or "")
        entry = self._active_tools.pop(key, None) if key else None
        if entry is None:
            entry = self._latest_active_tool_by_name(name)
            if entry is not None:
                self._active_tools.pop(entry["call_id"], None)
        if entry is None:
            entry = {
                "call_id": key or f"{name}:{len(self._tool_calls) + 1}",
                "name": name,
                "start_offset_seconds": self._elapsed(),
                "seconds": None,
                "argument_keys": self._argument_keys(args),
            }
            self._tool_calls.append(entry)
        if entry.get("seconds") is None:
            started_at = entry.get("_started_at")
            if isinstance(started_at, (int, float)):
                entry["seconds"] = round(max(0.0, self._clock() - started_at), 6)
            else:
                entry["seconds"] = 0.0
        entry["completed"] = True
        entry["result_chars"] = len(str(result or ""))

    def finish(self) -> None:
        self._close_current_api("run_completed")
        for key, entry in list(self._active_tools.items()):
            started_at = entry.get("_started_at")
            if entry.get("seconds") is None and isinstance(started_at, (int, float)):
                entry["seconds"] = round(max(0.0, self._clock() - started_at), 6)
            entry["completed"] = False
            self._active_tools.pop(key, None)

    # 将采集到的内部耗时转换成前端 debug 可以直接展示的结构。
    def debug_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        total_seconds: float,
    ) -> dict[str, Any]:
        self.finish()
        api_calls = [self._public_entry(item) for item in self._api_calls]
        response_shapes = self._assistant_response_shapes(messages)
        for index, shape in enumerate(response_shapes):
            if index >= len(api_calls):
                break
            api_calls[index].update(shape)
        for item in api_calls:
            item["phase_label"] = self._api_phase_label(item)
        tool_calls = [self._public_entry(item) for item in self._tool_calls]
        llm_wait_seconds = round(sum(item.get("seconds") or 0 for item in api_calls), 6)
        tool_seconds = round(sum(item.get("seconds") or 0 for item in tool_calls), 6)
        other_seconds = round(max(0.0, total_seconds - llm_wait_seconds - tool_seconds), 6)
        return {
            "total_seconds": round(total_seconds, 6),
            "llm_wait_seconds": llm_wait_seconds,
            "agent_tool_seconds": tool_seconds,
            "loop_overhead_seconds": other_seconds,
            "api_calls": api_calls,
            "agent_tool_calls": tool_calls,
            "bottleneck": self._bottleneck(api_calls, tool_calls),
            "note": (
                "assistant_response 包含模型请求准备与 provider 等待、客户端工具回调和少量调度开销；"
                "当模型请求工具时，工具结果会触发下一轮模型请求；如果 provider SDK 内部发生重试，会计入对应模型轮次耗时。"
            ),
        }

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return round(max(0.0, self._clock() - self._started_at), 6)

    def _close_current_api(self, ended_by: str) -> None:
        if self._current_api is None or self._current_api.get("seconds") is not None:
            return
        start_offset = self._current_api.get("start_offset_seconds") or 0.0
        self._current_api["seconds"] = round(max(0.0, self._elapsed() - start_offset), 6)
        self._current_api["ended_by"] = ended_by
        self._current_api = None

    def _latest_active_tool_by_name(self, name: str) -> dict[str, Any] | None:
        for entry in reversed(self._tool_calls):
            if entry.get("name") == name and not entry.get("completed"):
                return entry
        return None

    @staticmethod
    def _argument_keys(args: dict[str, Any] | None) -> list[str]:
        if not isinstance(args, dict):
            return []
        return sorted(str(key) for key in args.keys())

    @classmethod
    def _previous_tool_summaries(cls, previous_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        summaries = []
        for tool in previous_tools or []:
            if not isinstance(tool, dict):
                continue
            summary = {
                "name": tool.get("name"),
                "result_chars": len(str(tool.get("result") or "")),
            }
            arguments = cls._parse_tool_arguments(tool.get("arguments"))
            argument_keys = cls._argument_keys(arguments)
            if argument_keys:
                summary["argument_keys"] = argument_keys
            summaries.append(summary)
        return summaries

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str) or not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _assistant_response_shapes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        shapes: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                arguments = AssistantResponseTiming._parse_tool_arguments(
                    fn.get("arguments") or tc.get("arguments")
                )
                tool_calls.append({
                    "name": fn.get("name") or tc.get("name"),
                    "argument_keys": AssistantResponseTiming._argument_keys(arguments),
                })
            shapes.append({
                "finish_type": "tool_calls" if tool_calls else "final_response",
                "requested_tool_count": len(tool_calls),
                "requested_tools": tool_calls,
            })
        return shapes

    @staticmethod
    def _api_phase_label(entry: dict[str, Any]) -> str:
        if entry.get("phase") == "model_after_tool_results":
            names = [tool.get("name") for tool in entry.get("previous_tools") or [] if tool.get("name")]
            suffix = f"（工具结果：{', '.join(names)}）" if names else ""
            return f"工具结果后的模型续写{suffix}"
        if entry.get("finish_type") == "tool_calls":
            return "首轮模型判断并请求工具"
        if entry.get("finish_type") == "final_response":
            return "首轮模型直接生成最终回复"
        return "首轮模型请求"

    @staticmethod
    def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in entry.items()
            if not key.startswith("_") and value is not None
        }

    @staticmethod
    def _bottleneck(api_calls: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for item in api_calls:
            candidates.append({
                "kind": "llm_api_call",
                "name": f"api_call_{item.get('index')}",
                "seconds": item.get("seconds") or 0,
            })
        for item in tool_calls:
            candidates.append({
                "kind": "agent_tool_call",
                "name": item.get("name") or "tool",
                "seconds": item.get("seconds") or 0,
            })
        return max(candidates, key=lambda item: item["seconds"], default={})

    @staticmethod
    def _call_previous(callback: Any, *args: Any, **kwargs: Any) -> None:
        if not callable(callback):
            return
        try:
            callback(*args, **kwargs)
        except Exception:
            return
