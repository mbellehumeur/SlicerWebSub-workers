"""Status of the independent Orchid docker compose stack for the control panel."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger("control.orchid_status")

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERRIDE_EXAMPLE = REPO_ROOT / "orchid" / "docker" / "compose.override.example.yml"
DEFAULT_API_URL = "http://127.0.0.1:8000"
HEALTH_TIMEOUT_SEC = 2.0
COMPOSE_TIMEOUT_SEC = 8.0


def orchid_src() -> Path:
    env = (os.environ.get("ORCHID_SRC") or "").strip()
    if env:
        return Path(env).expanduser()
    home = Path.home()
    return home / "src" / "orchid"


def api_url() -> str:
    return (os.environ.get("ORCHID_API_URL") or DEFAULT_API_URL).rstrip("/")


def _compose_argv(src: Path) -> List[str]:
    argv = ["docker", "compose", "-f", "docker-compose.yml"]
    local_override = src / "compose.override.yml"
    if local_override.is_file():
        argv.extend(["-f", "compose.override.yml"])
    elif OVERRIDE_EXAMPLE.is_file():
        argv.extend(["-f", str(OVERRIDE_EXAMPLE)])
    argv.extend(["ps", "--format", "json"])
    return argv


def _normalize_ports(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                published = item.get("published") or item.get("PublicPort")
                target = item.get("target") or item.get("PrivatePort")
                if published and target:
                    parts.append(f"{published}->{target}")
                elif published:
                    parts.append(str(published))
        return ", ".join(parts)
    return str(raw)


def _parse_compose_ps(stdout: str) -> List[Dict[str, Any]]:
    """Parse ``docker compose ps --format json`` (NDJSON or JSON array)."""
    text = (stdout or "").strip()
    if not text:
        return []
    rows: List[Any] = []
    # Prefer NDJSON (one object per line) — Compose v2 often emits this.
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                rows = parsed
        except json.JSONDecodeError:
            rows = []
    if not rows:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            elif isinstance(obj, list):
                rows.extend(obj)

    containers: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        service = str(
            row.get("Service") or row.get("service") or row.get("Name") or ""
        ).strip()
        name = str(row.get("Name") or row.get("name") or service).strip()
        state = str(
            row.get("State") or row.get("state") or row.get("Status") or ""
        ).strip().lower()
        status = str(row.get("Status") or row.get("status") or "").strip()
        health = str(row.get("Health") or row.get("health") or "").strip()
        ports = _normalize_ports(row.get("Publishers") or row.get("Ports") or row.get("ports"))
        containers.append(
            {
                "service": service or name,
                "name": name,
                "state": state,
                "status": status,
                "health": health,
                "ports": ports,
            }
        )
    containers.sort(key=lambda c: str(c.get("service") or "").lower())
    return containers


def _probe_api(base: str) -> Dict[str, Any]:
    url = f"{base}/api/health"
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=HEALTH_TIMEOUT_SEC) as resp:
            body = resp.read()
            status = int(resp.status)
    except HTTPError as exc:
        return {"api_ok": False, "api": None, "api_error": f"HTTP {exc.code}"}
    except URLError as exc:
        return {"api_ok": False, "api": None, "api_error": str(exc.reason or exc)}
    except Exception as exc:  # noqa: BLE001 — surface any probe failure
        return {"api_ok": False, "api": None, "api_error": str(exc)}

    if status >= 400:
        return {"api_ok": False, "api": None, "api_error": f"HTTP {status}"}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {"raw": body.decode("utf-8", errors="replace")}
    return {"api_ok": True, "api": payload, "api_error": None}


def collect_orchid_status() -> Dict[str, Any]:
    """Return Orchid compose + API status (always JSON-serializable)."""
    src = orchid_src()
    base = api_url()
    result: Dict[str, Any] = {
        "orchid_src": str(src),
        "api_url": base,
        "api_ok": False,
        "api": None,
        "api_error": None,
        "compose_ok": False,
        "error": None,
        "containers": [],
    }

    api_info = _probe_api(base)
    result["api_ok"] = bool(api_info.get("api_ok"))
    result["api"] = api_info.get("api")
    result["api_error"] = api_info.get("api_error")

    compose_yml = src / "docker-compose.yml"
    if not compose_yml.is_file():
        result["error"] = (
            f"Orchid checkout not found at {src} "
            "(clone https://github.com/kalsiva/orchid or set ORCHID_SRC)"
        )
        return result

    argv = _compose_argv(src)
    try:
        proc = subprocess.run(
            argv,
            cwd=str(src),
            capture_output=True,
            text=True,
            timeout=COMPOSE_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError:
        result["error"] = "docker not found on PATH"
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "docker compose ps timed out"
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        # Truncate noisy docker permission errors for the UI.
        if len(err) > 400:
            err = err[:400] + "…"
        result["error"] = err
        LOGGER.warning("orchid compose ps failed: %s", err)
        return result

    containers = _parse_compose_ps(proc.stdout)
    result["containers"] = containers
    result["compose_ok"] = True
    if not containers:
        # Compose succeeded but nothing is up — not an error, just empty.
        result["error"] = None
    return result
