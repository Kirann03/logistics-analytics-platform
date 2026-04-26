from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def venv_python() -> str:
    if sys.platform.startswith("win"):
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else sys.executable)


def main() -> None:
    python_bin = venv_python()
    env = os.environ.copy()
    backend = subprocess.Popen(
        [python_bin, "-m", "uvicorn", "live_ingest_api:app", "--reload"],
        cwd=ROOT,
        env=env,
    )
    time.sleep(4)
    frontend = subprocess.Popen(
        [python_bin, "-m", "streamlit", "run", "app.py"],
        cwd=ROOT,
        env=env,
    )
    print("Backend starting on http://127.0.0.1:8000")
    print("Frontend starting with Streamlit")
    print("Press Ctrl+C here to stop both services.")
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        for proc in (frontend, backend):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    proc.terminate()
        for proc in (frontend, backend):
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
