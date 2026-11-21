"""HTTP / DICOM helpers for a locally installed Orchid platform."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pydicom

LOGGER = logging.getLogger("ORCHID.client")
StatusFn = Callable[[str], None]

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_DICOM_HOST = "127.0.0.1"
DEFAULT_DICOM_PORT = 11112
DEFAULT_DICOM_AET = "ORCHID"
DEFAULT_CALLING_AET = "WEBSUB"
DEFAULT_SETTLE_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT = 300.0
DEFAULT_POLL_INTERVAL = 1.5

_TERMINAL_STATUSES = frozenset({"complete", "failed", "no_route"})
_DICOM_SUFFIXES = (".dcm", ".dicom", ".DCM", ".DICOM")


def api_base() -> str:
    return (os.environ.get("ORCHID_API_URL") or DEFAULT_API_URL).rstrip("/")


def api_key() -> str:
    return (os.environ.get("ORCHID_API_KEY") or "").strip()


def dicom_host() -> str:
    return (os.environ.get("ORCHID_DICOM_HOST") or DEFAULT_DICOM_HOST).strip()


def dicom_port() -> int:
    raw = (os.environ.get("ORCHID_DICOM_PORT") or str(DEFAULT_DICOM_PORT)).strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_DICOM_PORT


def dicom_aet() -> str:
    return (os.environ.get("ORCHID_DICOM_AET") or DEFAULT_DICOM_AET).strip()


def calling_aet() -> str:
    return (os.environ.get("ORCHID_CALLING_AET") or DEFAULT_CALLING_AET).strip()


def settle_seconds() -> float:
    raw = (os.environ.get("ORCHID_SETTLE_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_SETTLE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_SETTLE_SECONDS


def poll_timeout() -> float:
    raw = (os.environ.get("ORCHID_POLL_TIMEOUT") or "").strip()
    if not raw:
        return DEFAULT_POLL_TIMEOUT
    try:
        return max(5.0, float(raw))
    except ValueError:
        return DEFAULT_POLL_TIMEOUT


def ingest_mode() -> str:
    """``auto`` | ``http`` | ``cstore``."""
    raw = (os.environ.get("ORCHID_INGEST") or "auto").strip().lower()
    if raw in ("http", "cstore", "auto"):
        return raw
    return "auto"


def _headers(*, json_body: bool = False) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    key = api_key()
    if key:
        headers["X-Api-Key"] = key
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _request(
    method: str,
    path: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 60.0,
) -> Tuple[int, bytes]:
    url = f"{api_base()}{path}"
    hdrs = dict(_headers())
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method.upper())
    try:
        with urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except HTTPError as exc:
        body = exc.read() if exc.fp is not None else b""
        return int(exc.code), body


def _request_json(
    method: str,
    path: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    timeout: float = 60.0,
) -> Any:
    data = None
    headers: Dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = _headers(json_body=True)
    status, body = _request(method, path, data=data, headers=headers, timeout=timeout)
    if status >= 400:
        raise RuntimeError(
            f"Orchid API {method} {path} → HTTP {status}: "
            f"{body[:300].decode('utf-8', errors='replace')}"
        )
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def health_check() -> Dict[str, Any]:
    """``GET /api/health``. Raises if Orchid is unreachable."""
    try:
        status, body = _request("GET", "/api/health", timeout=5.0)
    except URLError as exc:
        raise RuntimeError(
            f"Orchid not reachable at {api_base()} ({exc}). "
            "Start the independent platform first: ./orchid/docker/up.sh"
        ) from exc
    if status >= 400:
        raise RuntimeError(
            f"Orchid health check failed HTTP {status} at {api_base()}"
        )
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {"status": "ok", "raw": body.decode("utf-8", errors="replace")}


def ensure_orchid(*, on_status: Optional[StatusFn] = None) -> Dict[str, Any]:
    info = health_check()
    msg = (
        f"Orchid online at {api_base()} "
        f"(aet={info.get('aet', '?')} dicom_port={info.get('dicom_port', '?')})"
    )
    LOGGER.info(msg)
    if on_status:
        on_status(msg)
    return info


def list_dicom_files(input_dir: Path) -> List[Path]:
    """Collect DICOM instances under ``input_dir`` (extension or pydicom-readable)."""
    if not input_dir.is_dir():
        return []
    found: List[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in _DICOM_SUFFIXES or path.suffix == "":
            try:
                pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception:
                if path.suffix not in _DICOM_SUFFIXES:
                    continue
            found.append(path)
    return found


def study_instance_uid(dicom_files: Sequence[Path]) -> Optional[str]:
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        uid = getattr(ds, "StudyInstanceUID", None)
        if uid:
            return str(uid).strip()
    return None


def _multipart_upload(files: Sequence[Path], timeout: float = 300.0) -> Dict[str, Any]:
    """POST /api/studies/upload with multipart form ``files``."""
    import uuid

    boundary = f"----WebSubOrchid{uuid.uuid4().hex}"
    parts: List[bytes] = []
    for path in files:
        raw = path.read_bytes()
        name = path.name or "instance.dcm"
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="files"; '
                f'filename="{name}"\r\n'
                f"Content-Type: application/dicom\r\n\r\n"
            ).encode("utf-8")
            + raw
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    headers = _headers()
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    status, resp = _request(
        "POST",
        "/api/studies/upload",
        data=body,
        headers=headers,
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(
            f"Orchid upload failed HTTP {status}: "
            f"{resp[:300].decode('utf-8', errors='replace')}"
        )
    return json.loads(resp.decode("utf-8"))


def _cstore_available() -> bool:
    try:
        import pynetdicom  # noqa: F401
    except ImportError:
        return False
    return True


def cstore_send(
    dicom_files: Sequence[Path],
    *,
    on_status: Optional[StatusFn] = None,
) -> int:
    """C-STORE instances to Orchid. Requires ``pynetdicom``."""
    from pynetdicom import AE

    host = dicom_host()
    port = dicom_port()
    aet = dicom_aet()
    calling = calling_aet()
    datasets = [pydicom.dcmread(str(p)) for p in dicom_files]
    ae = AE(ae_title=calling)
    for ds in datasets:
        ae.add_requested_context(
            ds.SOPClassUID, ds.file_meta.TransferSyntaxUID
        )
    if on_status:
        on_status(f"C-STORE → {aet}@{host}:{port} ({len(datasets)} instances)")
    assoc = ae.associate(host, port, ae_title=aet)
    if not assoc.is_established:
        raise RuntimeError(f"could not associate with {aet}@{host}:{port}")
    sent = 0
    try:
        for ds, path in zip(datasets, dicom_files):
            status = assoc.send_c_store(ds)
            ok = status is not None and int(status.Status) == 0x0000
            if ok:
                sent += 1
            else:
                LOGGER.warning("C-STORE failed for %s status=%s", path.name, status)
    finally:
        assoc.release()
    LOGGER.info("C-STORE sent %s/%s", sent, len(datasets))
    return sent


def ingest_study(
    dicom_files: Sequence[Path],
    *,
    on_status: Optional[StatusFn] = None,
) -> str:
    """Ingest DICOM into Orchid; return StudyInstanceUID."""
    if not dicom_files:
        raise ValueError("no DICOM files to ingest")
    uid = study_instance_uid(dicom_files)
    if not uid:
        raise ValueError("could not read StudyInstanceUID from DICOM")

    mode = ingest_mode()
    use_cstore = mode == "cstore" or (mode == "auto" and _cstore_available())
    if mode == "cstore" and not _cstore_available():
        raise RuntimeError(
            "ORCHID_INGEST=cstore but pynetdicom is not installed "
            "(pip install pynetdicom)"
        )

    if use_cstore:
        try:
            sent = cstore_send(dicom_files, on_status=on_status)
            if sent <= 0:
                raise RuntimeError("C-STORE sent 0 instances")
            return uid
        except Exception as exc:
            if mode == "cstore":
                raise
            LOGGER.warning("C-STORE failed (%s); falling back to HTTP upload", exc)

    if on_status:
        on_status(f"HTTP upload → {api_base()} ({len(dicom_files)} files)")
    result = _multipart_upload(dicom_files)
    stored = int(result.get("stored") or 0)
    errors = result.get("errors") or []
    if stored <= 0:
        raise RuntimeError(f"Orchid upload stored 0 files; errors={errors}")
    if errors:
        LOGGER.warning("Orchid upload partial errors: %s", errors)
    LOGGER.info("HTTP upload stored=%s study_uid=%s", stored, uid)
    return uid


def find_study_by_uid(
    study_uid: str,
    *,
    timeout: Optional[float] = None,
    interval: float = DEFAULT_POLL_INTERVAL,
) -> Dict[str, Any]:
    """Poll study list until ``study_uid`` appears; return deep study dict."""
    deadline = time.monotonic() + (timeout if timeout is not None else poll_timeout())
    while True:
        rows = _request_json("GET", "/api/studies?limit=200")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and str(row.get("study_uid") or "") == study_uid:
                    study_id = row.get("id")
                    if study_id is not None:
                        deep = _request_json("GET", f"/api/studies/{int(study_id)}")
                        if isinstance(deep, dict):
                            return deep
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Orchid study {study_uid} not found within timeout"
            )
        time.sleep(interval)


def wait_for_study(
    study_id: int,
    *,
    timeout: Optional[float] = None,
    interval: float = DEFAULT_POLL_INTERVAL,
    on_status: Optional[StatusFn] = None,
) -> Dict[str, Any]:
    """Poll until study status is terminal."""
    # Give Orchid settle time before expecting routing.
    settle = settle_seconds()
    if settle > 0:
        time.sleep(settle)
    deadline = time.monotonic() + (timeout if timeout is not None else poll_timeout())
    last_status = ""
    while True:
        study = _request_json("GET", f"/api/studies/{int(study_id)}")
        if not isinstance(study, dict):
            raise RuntimeError(f"unexpected study payload: {study!r}")
        status = str(study.get("status") or "")
        if status != last_status:
            last_status = status
            line = f"Orchid study {study_id} status={status}"
            LOGGER.info(line)
            if on_status:
                on_status(line)
        if status in _TERMINAL_STATUSES:
            return study
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Orchid study {study_id} still {status!r} after timeout"
            )
        time.sleep(interval)


def collect_outputs(
    study: Mapping[str, Any],
    output_dir: Path,
    *,
    app_slugs: Optional[Sequence[str]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Download SR files + write findings.json. Returns (files, findings)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_file():
            old.unlink()

    results = [
        e for e in (study.get("results") or []) if isinstance(e, dict)
    ]

    slug_filter = {s.strip() for s in (app_slugs or []) if s and s.strip()}
    selected: List[Dict[str, Any]] = results
    if slug_filter:
        apps = _request_json("GET", "/api/apps")
        id_by_slug: Dict[str, int] = {}
        if isinstance(apps, list):
            for app in apps:
                if (
                    isinstance(app, dict)
                    and app.get("slug")
                    and app.get("id") is not None
                ):
                    id_by_slug[str(app["slug"])] = int(app["id"])
        wanted_ids = {id_by_slug[s] for s in slug_filter if s in id_by_slug}
        if wanted_ids:
            selected = [e for e in results if e.get("app_id") in wanted_ids]
        else:
            # Fall back to loose name match if slug not in marketplace yet.
            selected = [
                e
                for e in results
                if any(
                    slug.lower() in str(e.get("app_name") or "").lower()
                    for slug in slug_filter
                )
            ]
        if not selected:
            LOGGER.warning(
                "No Orchid results matched app slugs %s; keeping all %d",
                sorted(slug_filter),
                len(results),
            )
            selected = results

    findings: Dict[str, Any] = {
        "study_id": study.get("id"),
        "study_uid": study.get("study_uid"),
        "status": study.get("status"),
        "results": selected,
    }
    findings_path = output_dir / "findings.json"
    findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    out_files: List[Path] = [findings_path]
    for entry in selected:
        result_id = entry.get("id")
        if result_id is None or not entry.get("has_sr"):
            continue
        status, body = _request(
            "GET", f"/api/results/{int(result_id)}/sr", timeout=60.0
        )
        if status >= 400 or not body:
            LOGGER.warning(
                "SR download failed result_id=%s HTTP %s", result_id, status
            )
            continue
        sr_path = output_dir / f"ai-sr-{int(result_id)}.dcm"
        sr_path.write_bytes(body)
        out_files.append(sr_path)

    return out_files, findings


def run_orchid_job(
    input_dir: Path,
    output_dir: Path,
    *,
    app_slugs: Optional[Sequence[str]] = None,
    on_status: Optional[StatusFn] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Full Hub→Orchid→artifacts path for one study directory."""
    ensure_orchid(on_status=on_status)
    dicom_files = list_dicom_files(input_dir)
    if not dicom_files:
        raise ValueError(f"no DICOM files under {input_dir}")

    study_uid = ingest_study(dicom_files, on_status=on_status)
    if on_status:
        on_status(f"Waiting for Orchid study {study_uid}…")
    study = find_study_by_uid(study_uid)
    study_id = int(study["id"])
    study = wait_for_study(study_id, on_status=on_status)

    status = str(study.get("status") or "")
    if status == "failed":
        raise RuntimeError(f"Orchid study {study_id} failed")
    if status == "no_route":
        LOGGER.warning(
            "Orchid study %s has no_route — still collecting any results",
            study_id,
        )

    files, findings = collect_outputs(
        study, output_dir, app_slugs=app_slugs
    )
    if on_status:
        on_status(
            f"Orchid complete: {len(files)} artifact(s), "
            f"{len(findings.get('results') or [])} result(s)"
        )
    return files, findings
