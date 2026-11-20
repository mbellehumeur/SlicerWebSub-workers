#!/usr/bin/env python3
"""Orchid ``/analyze`` adapter wrapping TorchXRayVision (CPU).

Contract: https://github.com/kalsiva/orchid/blob/main/docs/INTEGRATING_AI_APPS.md

Env:
  TXRV_WEIGHTS — pretrained id (default densenet121-res224-all)
  TXRV_DEVICE  — cpu | cuda (default cpu)
  PORT         — listen port (default 9201)
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("orchid.adapter.txrv")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_WEIGHTS = "densenet121-res224-all"
# Pathologies commonly treated as "positive / abnormal" for a triage-style flag.
_POSITIVE_KEYS = frozenset(
    {
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    }
)

app = FastAPI(title="WebSub Orchid TXRV adapter", version="1.0.0")

_model = None
_pathologies: List[str] = []
_device = "cpu"
_weights = DEFAULT_WEIGHTS


class StudyMeta(BaseModel):
    uid: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    description: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None


class AnalyzeRequest(BaseModel):
    study: StudyMeta = Field(default_factory=StudyMeta)
    image_png_b64: str
    num_instances: int = 1


def _resolve_device(requested: str) -> str:
    device = (requested or "cpu").strip().lower()
    if device not in ("cuda", "cpu"):
        device = "cpu"
    if device != "cuda":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; using cpu")
            return "cpu"
    except ImportError:
        return "cpu"
    return "cuda"


def _load_model() -> None:
    global _model, _pathologies, _device, _weights
    if _model is not None:
        return

    import torch
    import torchvision
    import torchxrayvision as xrv

    _weights = (
        os.environ.get("TXRV_WEIGHTS") or DEFAULT_WEIGHTS
    ).strip() or DEFAULT_WEIGHTS
    _device = _resolve_device(os.environ.get("TXRV_DEVICE") or "cpu")
    LOGGER.info("Loading TXRV weights=%s device=%s", _weights, _device)
    model = xrv.models.get_model(_weights)
    model.eval()
    if _device == "cuda":
        model = model.cuda()
    _model = model
    _pathologies = list(
        getattr(model, "pathologies", None) or xrv.datasets.default_pathologies
    )
    # Keep transforms module refs for analyze().
    app.state.torch = torch
    app.state.torchvision = torchvision
    app.state.xrv = xrv


@app.on_event("startup")
def _startup() -> None:
    _load_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "adapter": "websub-txrv",
        "weights": _weights,
        "device": _device,
        "pathologies": len(_pathologies),
    }


def _scores_from_png(png_bytes: bytes) -> Dict[str, float]:
    _load_model()
    torch = app.state.torch
    torchvision = app.state.torchvision
    xrv = app.state.xrv

    with tempfile.TemporaryDirectory(prefix="txrv-orchid-") as tmp:
        png_path = Path(tmp) / "frame.png"
        png_path.write_bytes(png_bytes)
        img = xrv.utils.load_image(str(png_path))
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )
        img = transform(img)
        with torch.no_grad():
            tensor = torch.from_numpy(img).unsqueeze(0)
            if _device == "cuda":
                tensor = tensor.cuda()
            preds = _model(tensor).cpu()[0].detach().numpy()

    return {
        str(name): float(score)
        for name, score in zip(_pathologies, preds)
    }


def _triage(scores: Dict[str, float]) -> Tuple[float, str, bool, List[Dict[str, Any]]]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_score = ranked[0] if ranked else ("Unknown", 0.0)
    # Confidence = max score among "positive" findings, else overall max.
    positive_scores = [
        (n, s) for n, s in ranked if n in _POSITIVE_KEYS or "Lesion" in n
    ]
    if positive_scores:
        conf_name, confidence = positive_scores[0]
    else:
        conf_name, confidence = top_name, top_score
    positive = confidence >= 0.5
    label = conf_name if positive else "No high-confidence finding"
    findings = [
        {
            "type": "label",
            "label": name,
            "score": round(score, 4),
        }
        for name, score in ranked[:8]
    ]
    return float(confidence), label, positive, findings


@app.post("/analyze")
def analyze(body: AnalyzeRequest) -> Dict[str, Any]:
    try:
        png_bytes = base64.b64decode(body.image_png_b64)
    except Exception as exc:
        raise HTTPException(400, f"invalid image_png_b64: {exc}") from exc
    if not png_bytes:
        raise HTTPException(400, "empty image_png_b64")

    try:
        scores = _scores_from_png(png_bytes)
    except Exception as exc:
        LOGGER.exception("TXRV inference failed")
        raise HTTPException(500, f"inference failed: {exc}") from exc

    confidence, label, positive, findings = _triage(scores)
    summary = (
        f"TorchXRayVision ({_weights}): {label} "
        f"(confidence={confidence:.3f})"
    )
    return {
        "kind": "classification",
        "positive": positive,
        "confidence": confidence,
        "label": label,
        "summary": summary,
        "findings": findings,
        "measurements": {
            k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:18]
        },
    }


def main() -> None:
    import uvicorn

    host = (os.environ.get("ORCHID_ADAPTER_HOST") or "0.0.0.0").strip()
    port = int((os.environ.get("PORT") or "9201").strip() or "9201")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
