#!/usr/bin/env python3
"""
CouncilMed — 1-Click Complete Demo Launcher
==========================================
Starts all backend infrastructure in a single command:
1. Verifies/Starts Ollama local LLM service (http://127.0.0.1:11434).
2. Starts FastAPI backend (port 8000) with all 9 PyTorch council models.
3. Launches Cloudflare Tunnel to expose the backend securely on a public HTTPS URL.
4. Copies the public HTTPS URL to your Windows clipboard.
5. Saves the tunnel URL to data/active_tunnel.json.
6. Gracefully shuts down all child processes when you press Ctrl+C.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
CLOUDFLARED_PATH = PROJECT_ROOT / "scripts" / "cloudflared.exe"

OLLAMA_PATHS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
    Path("C:/Program Files/Ollama/ollama.exe"),
    Path("C:/Program Files (x86)/Ollama/ollama.exe"),
]


def is_service_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CouncilMed-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 204, 307)
    except Exception:
        return False


def find_ollama_executable() -> str | None:
    for p in OLLAMA_PATHS:
        if p.exists():
            return str(p)
    # Check PATH
    try:
        res = subprocess.run(["where", "ollama"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                return lines[0].strip()
    except Exception:
        pass
    return None


def copy_to_clipboard(text: str) -> bool:
    try:
        subprocess.run(["clip.exe"], input=text.strip().encode("utf-8"), check=True, timeout=2)
        return True
    except Exception:
        return False


def main() -> int:
    print("=" * 80)
    print("  COUNCILMED CLINICAL AI WORKSTATION — 1-CLICK DEMO LAUNCHER")
    print("=" * 80)

    processes: list[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        print("\n\nShutting down all CouncilMed services...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print("[OK] All processes terminated cleanly. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # -------------------------------------------------------------
    # 1. Start / Verify Ollama
    # -------------------------------------------------------------
    print("\n[1/3] Checking Ollama Local AI Infrastructure...")
    ollama_ready = is_service_ready("http://127.0.0.1:11434/api/tags", timeout=1.5)
    if not ollama_ready:
        ollama_bin = find_ollama_executable()
        if ollama_bin:
            print(f"  Starting Ollama daemon from {ollama_bin}...")
            ollama_proc = subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(ollama_proc)

            # Wait for Ollama to come online
            for _ in range(12):
                time.sleep(0.5)
                if is_service_ready("http://127.0.0.1:11434/api/tags"):
                    ollama_ready = True
                    break
        else:
            print("  [!] Ollama executable not found in standard paths.")
            print("  --> System will automatically utilize built-in deterministic clinical synthesis fallback.")

    if ollama_ready:
        print("  [OK] Ollama is active on http://127.0.0.1:11434")
    else:
        print("  [!] Ollama offline — deterministic local clinical agent will handle synthesis.")

    # -------------------------------------------------------------
    # 2. Start Python FastAPI Backend
    # -------------------------------------------------------------
    print("\n[2/3] Launching FastAPI Backend with 9 PyTorch Council Models...")
    python_exec = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)
    backend_cmd = [
        python_exec,
        "-m", "uvicorn",
        "backend.api.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "warning"
    ]

    backend_proc = subprocess.Popen(backend_cmd, cwd=str(PROJECT_ROOT))
    processes.append(backend_proc)

    # Poll until /api/health responds
    backend_online = False
    print("  Initializing 9 neural models into memory...", end="", flush=True)
    for _ in range(30):
        time.sleep(0.5)
        print(".", end="", flush=True)
        if is_service_ready("http://127.0.0.1:8000/api/health"):
            backend_online = True
            break
    print()

    if not backend_online:
        print("  [ERROR] Backend failed to initialize within 15 seconds.")
        cleanup()
        return 1

    print("  [OK] Backend online on http://127.0.0.1:8000 (All 9 Models Ready)")

    # -------------------------------------------------------------
    # 3. Start Cloudflare Tunnel
    # -------------------------------------------------------------
    print("\n[3/3] Establishing Secure Public Cloudflare Tunnel...")
    cf_bin = CLOUDFLARED_PATH if CLOUDFLARED_PATH.exists() else Path("C:/Program Files (x86)/cloudflared/cloudflared.exe")
    if not cf_bin.exists():
        cf_bin = Path("cloudflared")

    tunnel_cmd = [
        str(cf_bin),
        "tunnel",
        "--url", "http://127.0.0.1:8000"
    ]

    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    processes.append(tunnel_proc)

    # Read tunnel stdout/stderr to extract the public URL
    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for _ in range(40):
        line = tunnel_proc.stdout.readline()
        if not line:
            time.sleep(0.2)
            continue
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if not tunnel_url:
        print("  [!] Tunnel could not extract URL automatically.")
        print("  --> You can still access the application locally on http://localhost:8000")
    else:
        # Save tunnel to JSON
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        tunnel_file = data_dir / "active_tunnel.json"
        tunnel_file.write_text(
            f'{{"tunnel_url": "{tunnel_url}", "started_at": "{time.strftime("%Y-%m-%dT%H:%M:%SZ")}"}}\n',
            encoding="utf-8"
        )
        copied = copy_to_clipboard(tunnel_url)

        print("\n" + "=" * 80)
        print("  COUNCILMED IS LIVE & READY FOR DEMO!")
        print("=" * 80)
        print(f"  [+] Local API:           http://localhost:8000")
        print(f"  [+] Public HTTPS Tunnel: {tunnel_url}")
        if copied:
            print(f"  [+] Clipboard:           COPIED TO CLIPBOARD!")
        print("=" * 80)
        print("\nNext step on Vercel:")
        print(f"  1. Go to your Vercel Project Settings -> Environment Variables")
        print(f"  2. Set VITE_API_URL = {tunnel_url}")
        print(f"  3. Open your Vercel URL and run the demo on any device!")
        print("\nPress Ctrl+C at any time in this window to stop all services.")
        print("=" * 80 + "\n")

    # Read Vercel URL from file if user configured it
    vercel_url_file = PROJECT_ROOT / "vercel_url.txt"
    if vercel_url_file.exists():
        for line in vercel_url_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("http://") or line.startswith("https://"):
                print(f"Opening Vercel demo link in default browser: {line}")
                try:
                    webbrowser.open(line)
                except Exception:
                    pass
                break

    # Keep alive until Ctrl+C
    try:
        while True:
            time.sleep(1)
            # Check if any child died unexpectedly
            if backend_proc.poll() is not None:
                print("\n[!] Backend process exited unexpectedly.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
