from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from ai_glasses_memory_assistant.server_config import parse_server_bind, startup_message


# server 配置测试锁定默认局域网可访问和非法端口拦截。
class ServerConfigTest(unittest.TestCase):
    def test_default_bind_allows_lan_access(self) -> None:
        bind = parse_server_bind([])

        self.assertEqual(bind.host, "0.0.0.0")
        self.assertEqual(bind.port, 8765)
        self.assertIn("Same LAN device", startup_message(bind))
        self.assertIn("http://192.168.1.23:8765", startup_message(bind, lan_ip="192.168.1.23"))
        self.assertIn("LAN HTTP note", startup_message(bind, lan_ip="192.168.1.23"))

    def test_local_asr_reports_insecure_lan_http_before_browser_unsupported(self) -> None:
        app_js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
        function_body = app_js.split("function getLocalASRUnsupportedMessage()", 1)[1].split(
            "function getLocalASRErrorMessage", 1
        )[0]

        self.assertLess(
            function_body.index("!window.isSecureContext"),
            function_body.index("!navigator.mediaDevices?.getUserMedia"),
        )

    def test_https_server_defers_tls_handshake_to_worker_thread(self) -> None:
        server_py = (Path(__file__).resolve().parents[1] / "ai_glasses_memory_assistant" / "server.py").read_text(encoding="utf-8")

        self.assertIn("class ThreadingHTTPSServer(ThreadingHTTPServer):", server_py)
        self.assertIn("def process_request_thread", server_py)
        self.assertIn("do_handshake_on_connect=False", server_py)
        self.assertNotIn("httpd.socket = context.wrap_socket", server_py)

    def test_https_bind_outputs_secure_lan_url(self) -> None:
        bind = parse_server_bind([
            "--certfile",
            "certs/cert.pem",
            "--keyfile",
            "certs/key.pem",
        ])

        self.assertEqual(bind.certfile, "certs/cert.pem")
        self.assertEqual(bind.keyfile, "certs/key.pem")
        self.assertIn("https://127.0.0.1:8765", startup_message(bind, lan_ip="192.168.1.23"))
        self.assertIn("https://192.168.1.23:8765", startup_message(bind, lan_ip="192.168.1.23"))
        self.assertNotIn("LAN HTTP note", startup_message(bind, lan_ip="192.168.1.23"))

    def test_custom_bind_keeps_mac_only_option(self) -> None:
        bind = parse_server_bind(["--host", "127.0.0.1", "--port", "9000"])

        self.assertEqual(bind.host, "127.0.0.1")
        self.assertEqual(bind.port, 9000)
        self.assertIn("http://127.0.0.1:9000", startup_message(bind))

    def test_rejects_incomplete_https_cert_pair(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_server_bind(["--certfile", "certs/cert.pem"])

    def test_rejects_invalid_port(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_server_bind(["--port", "70000"])


if __name__ == "__main__":
    unittest.main()
