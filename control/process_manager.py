"""Start / stop / status / logs for product worker subprocesses."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import ProductInfo, REPO_ROOT

DEFAULT_RUNTIME_ROOT = Path("/var/lib/websub-workers")
STOP_WAIT_SECONDS = 5.0


def runtime_root() -> Path:
    env = os.environ.get("WEBSUB_WORKERS_RUNTIME_ROOT", "").strip()
    return Path(env).expanduser() if env else DEFAULT_RUNTIME_ROOT


def control_root() -> Path:
    return runtime_root() / "control"


def product_state_dir(product_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in product_id)
    return control_root() / safe


@dataclass
class WorkerStatus:
    product_id: str
    running: bool
    pid: Optional[int] = None
    hub: Optional[str] = None
    script: Optional[str] = None
    started_at: Optional[str] = None
    managed: bool = False
    log_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "running": self.running,
            "pid": self.pid,
            "hub": self.hub,
            "script": self.script,
            "started_at": self.started_at,
            "managed": self.managed,
            "log_path": self.log_path,
            "error": self.error,
        }


def _pid_path(product_id: str) -> Path:
    return product_state_dir(product_id) / "pid"


def _meta_path(product_id: str) -> Path:
    return product_state_dir(product_id) / "meta.json"


def log_path(product_id: str) -> Path:
    return product_state_dir(product_id) / "worker.log"


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _cmdline_matches_script(cmdline: str, script: Path) -> bool:
    if not cmdline:
        return False
    name = script.name
    try:
        rel = str(script.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(script)
    return name in cmdline or rel in cmdline or str(script) in cmdline


def _find_external_pid(product: ProductInfo) -> Optional[int]:
    """Find a running process whose cmdline matches this product's scripts."""
    scripts = product.match_scripts()
    if not scripts:
        return None
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        cmdline = _read_cmdline(pid)
        if not cmdline:
            continue
        # Ignore the control panel itself
        if "-m control" in cmdline or "control/__main__" in cmdline:
            continue
        if "control/app.py" in cmdline:
            continue
        for script in scripts:
            if _cmdline_matches_script(cmdline, script):
                return pid
    return None


def _load_meta(product_id: str) -> Dict[str, Any]:
    path = _meta_path(product_id)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_meta(product_id: str, meta: Dict[str, Any]) -> None:
    path = _meta_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read_managed_pid(product_id: str) -> Optional[int]:
    path = _pid_path(product_id)
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if not _is_pid_alive(pid):
        return None
    return pid


def status(product: ProductInfo) -> WorkerStatus:
    log = log_path(product.id)
    managed_pid = _read_managed_pid(product.id)
    meta = _load_meta(product.id)

    if managed_pid is not None:
        return WorkerStatus(
            product_id=product.id,
            running=True,
            pid=managed_pid,
            hub=meta.get("hub"),
            script=meta.get("script"),
            started_at=meta.get("started_at"),
            managed=True,
            log_path=str(log) if log.is_file() else None,
        )

    external = _find_external_pid(product)
    if external is not None:
        script_name = None
        cmdline = _read_cmdline(external)
        for script in product.match_scripts():
            if _cmdline_matches_script(cmdline, script):
                try:
                    script_name = str(script.relative_to(REPO_ROOT))
                except ValueError:
                    script_name = str(script)
                break
        hub = "local" if "--local" in cmdline else "cloud"
        return WorkerStatus(
            product_id=product.id,
            running=True,
            pid=external,
            hub=hub,
            script=script_name,
            started_at=None,
            managed=False,
            log_path=str(log) if log.is_file() else None,
        )

    return WorkerStatus(
        product_id=product.id,
        running=False,
        hub=meta.get("hub"),
        script=meta.get("script"),
        started_at=meta.get("started_at"),
        managed=False,
        log_path=str(log) if log.is_file() else None,
    )


def start(product: ProductInfo, hub: str = "cloud") -> WorkerStatus:
    if not product.startable or product.entry_script is None:
        return WorkerStatus(
            product_id=product.id,
            running=False,
            error="Product has no runnable entry script",
        )

    current = status(product)
    if current.running:
        return WorkerStatus(
            product_id=product.id,
            running=True,
            pid=current.pid,
            hub=current.hub,
            script=current.script,
            started_at=current.started_at,
            managed=current.managed,
            log_path=current.log_path,
            error="Already running",
        )

    hub_mode = "local" if hub == "local" else "cloud"
    state = product_state_dir(product.id)
    state.mkdir(parents=True, exist_ok=True)
    log = log_path(product.id)

    cmd = [sys.executable, str(product.entry_script)]
    if hub_mode == "local":
        cmd.append("--local")

    env = os.environ.copy()
    # Ensure repo imports (worker.py, runtime/, etc.) resolve.
    pythonpath = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT), str(REPO_ROOT / "runtime")]
    if pythonpath:
        parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    log_file = open(log, "a", encoding="utf-8")
    try:
        log_file.write(
            f"\n--- start {datetime.now(timezone.utc).isoformat()} "
            f"hub={hub_mode} cmd={' '.join(cmd)} ---\n"
        )
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        script_rel = str(product.entry_script.relative_to(REPO_ROOT))
    except ValueError:
        script_rel = str(product.entry_script)

    meta = {
        "hub": hub_mode,
        "script": script_rel,
        "started_at": started_at,
        "pid": proc.pid,
        "cmd": cmd,
    }
    _write_meta(product.id, meta)
    _pid_path(product.id).write_text(str(proc.pid) + "\n", encoding="utf-8")

    # Brief settle so immediate connect failures surface in status.
    time.sleep(0.3)
    if proc.poll() is not None:
        return WorkerStatus(
            product_id=product.id,
            running=False,
            pid=proc.pid,
            hub=hub_mode,
            script=script_rel,
            started_at=started_at,
            managed=True,
            log_path=str(log),
            error=f"Process exited immediately with code {proc.returncode}",
        )

    return WorkerStatus(
        product_id=product.id,
        running=True,
        pid=proc.pid,
        hub=hub_mode,
        script=script_rel,
        started_at=started_at,
        managed=True,
        log_path=str(log),
    )


def stop(product: ProductInfo) -> WorkerStatus:
    current = status(product)
    if not current.running or current.pid is None:
        return WorkerStatus(
            product_id=product.id,
            running=False,
            log_path=current.log_path,
            error=None,
        )

    pid = current.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            return WorkerStatus(
                product_id=product.id,
                running=True,
                pid=pid,
                hub=current.hub,
                script=current.script,
                started_at=current.started_at,
                managed=current.managed,
                log_path=current.log_path,
                error=f"Permission denied stopping pid {pid}: {exc}",
            )

    deadline = time.monotonic() + STOP_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # Clear managed pid file if it pointed at this process
    managed = _read_managed_pid(product.id)
    if managed is None or managed == pid:
        pid_file = _pid_path(product.id)
        if pid_file.is_file():
            try:
                pid_file.unlink()
            except OSError:
                pass

    still = _is_pid_alive(pid)
    return WorkerStatus(
        product_id=product.id,
        running=still,
        pid=pid if still else None,
        hub=current.hub,
        script=current.script,
        started_at=current.started_at,
        managed=False,
        log_path=current.log_path,
        error="Failed to stop process" if still else None,
    )


def read_log_tail(product_id: str, tail: int = 200) -> Tuple[str, Optional[str]]:
    """Return (text, path). Empty text if no log yet."""
    path = log_path(product_id)
    if not path.is_file():
        return "", None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(failed to read log: {exc})", str(path)
    if tail <= 0:
        return text, str(path)
    lines = text.splitlines()
    if len(lines) <= tail:
        return text, str(path)
    return "\n".join(lines[-tail:]) + "\n", str(path)


def log_file_size(product_id: str) -> int:
    path = log_path(product_id)
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def clear_log(product_id: str) -> str:
    """Truncate ``worker.log`` to empty; create the file if missing."""
    path = log_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return str(path)


def read_log_since(product_id: str, offset: int) -> Tuple[str, int]:
    """Read new log bytes from offset; return (chunk, new_offset)."""
    path = log_path(product_id)
    if not path.is_file():
        return "", 0
    try:
        size = path.stat().st_size
        if offset > size:
            offset = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            chunk = fh.read()
            return chunk, fh.tell()
    except OSError:
        return "", offset
