from __future__ import annotations

import argparse
import ipaddress
import socket
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


@dataclass(frozen=True)
class ServerBind:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    certfile: str | None = None
    keyfile: str | None = None

    # 证书成对传入时，启动文案和服务入口统一切换到 HTTPS。
    @property
    def scheme(self) -> str:
        return "https" if self.certfile and self.keyfile else "http"


# server 和 app 共用启动参数，避免两个入口绑定行为不一致。
def parse_server_bind(argv: list[str] | None = None) -> ServerBind:
    parser = argparse.ArgumentParser(description="Run the AI glasses memory assistant demo server.")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Address to bind. Use 127.0.0.1 for Mac-only access.",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help="TCP port to listen on.",
    )
    parser.add_argument(
        "--certfile",
        default=None,
        help="TLS certificate file for HTTPS LAN testing.",
    )
    parser.add_argument(
        "--keyfile",
        default=None,
        help="TLS private key file for HTTPS LAN testing.",
    )
    args = parser.parse_args(argv)
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("--certfile and --keyfile must be provided together")
    return ServerBind(host=args.host, port=args.port, certfile=args.certfile, keyfile=args.keyfile)


# 默认提示同时给出本机和局域网访问方式，方便手机测试语音 UI。
def startup_message(bind: ServerBind, lan_ip: str | None = None) -> str:
    lines = [f"AI glasses memory assistant listening on {bind.host}:{bind.port}"]
    scheme = bind.scheme
    if bind.host in {"0.0.0.0", ""}:
        resolved_lan_ip = lan_ip or _resolve_lan_ip()
        lines.append(f"Mac local: {scheme}://127.0.0.1:{bind.port}")
        lines.append(f"Same LAN device: {scheme}://{resolved_lan_ip or '<mac-lan-ip>'}:{bind.port}")
        if scheme == "http":
            lines.append("LAN HTTP note: browser geolocation and microphone require HTTPS on phones/glasses.")
    else:
        lines.append(f"Open: {scheme}://{_format_host(bind.host)}:{bind.port}")
    return "\n".join(lines)


# argparse 类型函数在解析阶段直接拦截非法端口。
def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _format_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


# 通过系统路由表推断当前默认出口网卡的 IPv4，不会实际发送 UDP 数据。
def _resolve_lan_ip() -> str | None:
    candidates = [_route_probe_lan_ip()]
    candidates.extend(_hostname_lan_ips())
    candidates.extend(_ifconfig_lan_ips())
    return _select_lan_ip(candidate for candidate in candidates if candidate)


def _route_probe_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.0", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def _hostname_lan_ips() -> list[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return []
    return [str(info[4][0]) for info in infos]


# macOS 沙箱或离线环境可能禁用 route probe，fallback 到本机接口列表。
def _ifconfig_lan_ips() -> list[str]:
    try:
        result = subprocess.run(["ifconfig"], capture_output=True, text=True, check=False, timeout=1)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return _extract_ifconfig_ipv4s(result.stdout)


def _extract_ifconfig_ipv4s(output: str) -> list[str]:
    skipped_prefixes = ("lo", "awdl", "llw", "utun", "gif", "stf", "bridge")
    current_iface = ""
    ips: list[str] = []
    for raw_line in output.splitlines():
        if raw_line and not raw_line[0].isspace():
            current_iface = raw_line.split(":", 1)[0]
        if current_iface.startswith(skipped_prefixes):
            continue
        line = raw_line.strip()
        if line.startswith("inet "):
            parts = line.split()
            if len(parts) >= 2:
                ips.append(parts[1])
    return ips


def _select_lan_ip(candidates: Iterable[str]) -> str | None:
    valid_ips: list[str] = []
    for candidate in candidates:
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if ip.version == 4 and not ip.is_loopback and not ip.is_unspecified and not ip.is_link_local:
            valid_ips.append(str(ip))
    private_ips = [value for value in valid_ips if ipaddress.ip_address(value).is_private]
    return (private_ips or valid_ips or [None])[0]
