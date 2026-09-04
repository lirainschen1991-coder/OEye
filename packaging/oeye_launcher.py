from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap


APP_NAME = "OEye"
CHILD_PROCESSES: list[subprocess.Popen] = []


def app_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME


def log_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def subprocess_creationflags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "requirements-standard.txt").exists() or (script_dir / "protected_sources").exists():
        return script_dir
    return script_dir.parents[0]


def script_path(name: str) -> Path:
    root = resource_root()
    direct = root / "packaging" / name
    if direct.exists():
        return direct
    packaged = root / name
    if packaged.exists():
        return packaged
    return direct


def find_free_port(start: int = 26080, attempts: int = 200) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("未找到可用本地端口。")


def streamlit_flags(port: int) -> dict:
    return {
        "server.address": "127.0.0.1",
        "server.port": port,
        "server.headless": True,
        "server.runOnSave": False,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
        "client.toolbarMode": "viewer",
    }


def run_streamlit_server(entry: Path, port: int) -> int:
    log_name = "dependency-check.log" if "dependency" in entry.name else "streamlit-server.log"
    log_path = log_dir() / log_name
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    sys.stdout = log_handle
    sys.stderr = log_handle
    try:
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        flags = streamlit_flags(port)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] serve {entry} on {port}", flush=True)
        bootstrap.load_config_options(flags)
        bootstrap.run(str(entry), False, [], flags)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        log_handle.flush()


def streamlit_command(kind: str, port: int) -> list[str]:
    args = [f"--serve-{kind}", "--port", str(port)]
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, str(Path(__file__).resolve()), *args]


def start_streamlit(kind: str, entry: Path, port: int, log_name: str) -> subprocess.Popen:
    log_path = log_dir() / log_name
    handle = log_path.open("a", encoding="utf-8", errors="replace")
    handle.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start {entry} on {port}\n")
    handle.flush()
    process = subprocess.Popen(
        streamlit_command(kind, port),
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(resource_root()),
        stdin=subprocess.DEVNULL,
        creationflags=subprocess_creationflags(),
    )
    CHILD_PROCESSES.append(process)
    return process


def stop_children() -> None:
    for process in list(CHILD_PROCESSES):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()


def wait_for_health(port: int, timeout: int = 45) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def open_window(url: str, title: str) -> None:
    try:
        import webview

        window = webview.create_window(title, url, width=1320, height=860, min_size=(1100, 720))
        webview.start()
        return
    except Exception as exc:
        with (log_dir() / "launcher.log").open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"pywebview unavailable, fallback to browser: {exc}\n")
    webbrowser.open(url)


def launch_dependency_ui() -> int:
    atexit.register(stop_children)
    port = find_free_port()
    start_streamlit("dependency", script_path("oeye_dependency_entry.py"), port, "dependency-check.log")
    if not wait_for_health(port):
        print(f"依赖管理 UI 启动超时，请查看日志：{log_dir() / 'dependency-check.log'}", file=sys.stderr)
        return 1
    open_window(f"http://127.0.0.1:{port}", "OEye 依赖检测与启动")
    stop_children()
    return 0


def launch_main_app(open_browser: bool = True) -> int:
    port = find_free_port(start=26180)
    start_streamlit("main", script_path("oeye_app_entry.py"), port, "streamlit-server.log")
    if not wait_for_health(port):
        print(f"主应用启动超时，请查看日志：{log_dir() / 'streamlit-server.log'}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}"
    if open_browser:
        webbrowser.open(url)
    print(url)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OEye desktop launcher")
    parser.add_argument("--main-app", action="store_true", help="启动 OEye 主应用 Streamlit 服务")
    parser.add_argument("--serve-dependency", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--serve-main", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-import", default="", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--no-browser", action="store_true", help="启动服务但不自动打开浏览器")
    args = parser.parse_args()

    if args.check_import:
        __import__(args.check_import)
        print(f"import-ok: {args.check_import}")
        return 0
    if args.serve_dependency:
        return run_streamlit_server(script_path("oeye_dependency_entry.py"), args.port)
    if args.serve_main:
        return run_streamlit_server(script_path("oeye_app_entry.py"), args.port)
    if args.main_app:
        return launch_main_app(open_browser=not args.no_browser)
    return launch_dependency_ui()


if __name__ == "__main__":
    raise SystemExit(main())
