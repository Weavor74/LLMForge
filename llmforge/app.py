"""Launching LLMForge as a desktop application.

The goal is that starting it involves no terminal: a launcher icon brings up the
server and a dedicated window, and closing the window puts it away again.

Training is unaffected by any of this. Workers run in their own sessions, so quitting
the app leaves a long run going and reopening it reconnects to the run in progress.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from llmforge.core import paths

DEFAULT_PORT = 8765
STARTUP_TIMEOUT = 60.0

# Chromium-family browsers can open a window with no tab strip or address bar, which
# is what makes this feel like an application rather than a web page.
APP_MODE_BROWSERS = (
    "chromium",
    "chromium-browser",
    "google-chrome-stable",
    "google-chrome",
    "brave-browser",
    "microsoft-edge",
)


def health_url(port: int, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}/api/health"


def is_running(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Whether an LLMForge server is already answering on this port."""
    try:
        with urllib.request.urlopen(health_url(port, host), timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def choose_port(preferred: int = DEFAULT_PORT, host: str = "127.0.0.1") -> tuple[int, bool]:
    """Find a usable port. Returns (port, already_serving)."""
    if is_running(preferred, host):
        return preferred, True
    if port_is_free(preferred, host):
        return preferred, False

    # Something else holds the preferred port; step along rather than fail.
    for offset in range(1, 20):
        candidate = preferred + offset
        if is_running(candidate, host):
            return candidate, True
        if port_is_free(candidate, host):
            return candidate, False

    raise RuntimeError(f"no free port near {preferred}")


def start_server(port: int, host: str = "127.0.0.1") -> subprocess.Popen:
    """Launch the API server as a child process, logging to the workspace."""
    log_path = paths.workspace() / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handle = log_path.open("ab")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "llmforge.api.main:app",
            "--host", host, "--port", str(port), "--log-level", "warning",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def wait_for_server(port: int, host: str = "127.0.0.1", timeout: float = STARTUP_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running(port, host):
            return True
        time.sleep(0.2)
    return False


def find_app_browser() -> str | None:
    for name in APP_MODE_BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def open_window(url: str) -> subprocess.Popen | None:
    """Open a dedicated application window. Returns the process, if we own one.

    A separate profile directory is not cosmetic: without it a chromium that is
    already running adopts the request, the command returns immediately, and we would
    treat that as the window having been closed.
    """
    browser = find_app_browser()
    if browser:
        profile = paths.workspace() / "browser-profile"
        profile.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(
            [
                browser,
                f"--app={url}",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1400,900",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # No chromium-family browser: fall back to whatever handles http. This opens a
    # normal tab and returns immediately, so the caller cannot wait on it.
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener:
        subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None


# ---------------------------------------------------------------------------
# desktop integration
# ---------------------------------------------------------------------------

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="13" fill="#0a0a0a"/>
  <rect x="1.5" y="1.5" width="61" height="61" rx="11.5" fill="none"
        stroke="#262626" stroke-width="1.5"/>
  <!-- A loss curve: the thing you actually watch. -->
  <path d="M12 20 C 22 20, 26 40, 34 44 C 42 48, 46 47, 52 46"
        fill="none" stroke="#34d399" stroke-width="3.5"
        stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="52" cy="46" r="3.5" fill="#34d399"/>
  <path d="M12 52 L 52 52" stroke="#404040" stroke-width="2" stroke-linecap="round"/>
  <path d="M12 52 L 12 16" stroke="#404040" stroke-width="2" stroke-linecap="round"/>
</svg>
"""


def _launcher_command() -> str:
    """Absolute command the desktop entry should run.

    The console script is preferred; falling back to the interpreter keeps this
    working in a checkout that was never `pip install`ed.
    """
    script = Path(sys.executable).parent / "llmforge"
    if script.exists():
        return f"{script} app"
    return f"{sys.executable} -m llmforge.cli app"


def install_desktop_entry() -> tuple[Path, Path]:
    """Install the launcher into the desktop's application menu."""
    apps = Path.home() / ".local/share/applications"
    icons = Path.home() / ".local/share/icons/hicolor/scalable/apps"
    apps.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)

    icon_path = icons / "llmforge.svg"
    icon_path.write_text(ICON_SVG)

    entry_path = apps / "llmforge.desktop"
    entry_path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=LLMForge\n"
        "GenericName=Language Model Builder\n"
        "Comment=Point at a folder, get a model\n"
        f"Exec={_launcher_command()}\n"
        "Icon=llmforge\n"
        "Terminal=false\n"
        # One main category only, or the entry is listed twice in the menu.
        "Categories=Development;\n"
        "Keywords=LLM;training;fine-tuning;AI;\n"
        "StartupNotify=true\n"
        "StartupWMClass=llmforge\n"
    )
    entry_path.chmod(0o755)

    # Without this the entry may not appear until the session restarts.
    for command, argument in (
        ("update-desktop-database", str(apps)),
        ("gtk-update-icon-cache", str(icons.parent.parent)),
    ):
        binary = shutil.which(command)
        if binary:
            subprocess.run(
                [binary, argument], capture_output=True, check=False, timeout=30
            )

    return entry_path, icon_path


def uninstall_desktop_entry() -> list[Path]:
    removed = []
    for path in (
        Path.home() / ".local/share/applications/llmforge.desktop",
        Path.home() / ".local/share/icons/hicolor/scalable/apps/llmforge.svg",
    ):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed
