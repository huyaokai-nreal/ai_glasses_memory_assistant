import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import pytest

from ai_glasses_memory_assistant.acceptance_runtime import BASE_URL, MODEL, GuardedAcceptanceClient


def ledger_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_live_memory_closeout.py"
    spec = importlib.util.spec_from_file_location("live_closeout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reservation():
    return {"client": {"provider": "llama_cpp", "model": MODEL, "base_url": BASE_URL},
            "request": {"model": MODEL, "messages": []}}


def test_parallel_reservations_stop_at_five_and_survive_restart(tmp_path):
    module = ledger_module()
    ledger = module.Ledger(tmp_path, ["quality/test/after"])
    ledger.dispatch("/begin", {"variant": "quality/test/after"})
    def attempt(_):
        try:
            return ledger.dispatch("/reserve", reservation())["request_id"]
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=12) as pool:
        keys = list(pool.map(attempt, range(20)))
    assert len([key for key in keys if key]) == 5
    restored = module.Ledger(tmp_path, ["quality/test/after"])
    assert restored.dispatch("/status", {})["total"] == 5
    with pytest.raises(ValueError): restored.dispatch("/end", {})
    for key in filter(None, keys): restored.dispatch("/finish", {"request_id": key, "error": "timeout"})
    restored.dispatch("/end", {})
    with pytest.raises(ValueError): restored.dispatch("/begin", {"variant": "quality/test/after"})


def test_global_640_limit_and_destination_guard(tmp_path):
    module = ledger_module()
    ledger = module.Ledger(tmp_path, ["last"])
    ledger.count["earlier"] = 640
    ledger.dispatch("/begin", {"variant": "last"})
    with pytest.raises(ValueError): ledger.dispatch("/reserve", reservation())
    ledger.count.clear()
    bad = reservation(); bad["client"]["base_url"] = "https://cloud.invalid"
    with pytest.raises(ValueError): ledger.dispatch("/reserve", bad)
    assert ledger.dispatch("/status", {})["total"] == 0


def test_http_error_is_counted_once_and_never_retried():
    client = GuardedAcceptanceClient(model=MODEL, base_url=BASE_URL, api_key="ollama", provider="llama_cpp")
    with patch("ai_glasses_memory_assistant.acceptance_runtime.authority", return_value={"request_id": "one"}) as authority, \
            patch("ai_glasses_memory_assistant.llm_client.urlopen", side_effect=TimeoutError("offline")) as send:
        with pytest.raises(TimeoutError): client.run_conversation("synthetic")
    assert send.call_count == 1
    assert [call.args[0] for call in authority.call_args_list] == ["/reserve", "/finish"]
    assert authority.call_args_list[-1].args[1]["error"]["type"] == "TimeoutError"


def test_reservation_failure_never_sends_model_request():
    client = GuardedAcceptanceClient(model=MODEL, base_url=BASE_URL, api_key="ollama", provider="llama_cpp")
    with patch("ai_glasses_memory_assistant.acceptance_runtime.authority", side_effect=RuntimeError("budget")), \
            patch("ai_glasses_memory_assistant.llm_client.urlopen") as send:
        with pytest.raises(RuntimeError): client.run_conversation("synthetic")
    send.assert_not_called()


def test_wrong_configuration_never_reserves_or_sends():
    client = GuardedAcceptanceClient(model=MODEL, base_url="https://cloud.invalid", api_key="ollama", provider="llama_cpp")
    with patch("ai_glasses_memory_assistant.acceptance_runtime.authority") as authority, \
            patch("ai_glasses_memory_assistant.llm_client.urlopen") as send:
        with pytest.raises(ValueError): client.run_conversation("synthetic")
    authority.assert_not_called(); send.assert_not_called()
