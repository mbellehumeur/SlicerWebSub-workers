"""Discover product cards (*.info.json) and map them to worker entry scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Preferred entry scripts keyed by info.json ``id``. Docker variants preferred.
# ``None`` means no runnable Hub worker (shown as unavailable).
ENTRYPOINTS: Dict[str, Optional[str]] = {
    "mhub": "mhub/mhub.py",
    "eucaim": "eucaim/eucaim_docker.py",
    "dental": "dental_segmentator/dental_segmentator_docker.py",
    "totalseg": "total_segmentator/total_segmentator_docker.py",
    "vox2seglam": "vox2seglam/vox2seglam_docker.py",
    "txrv": "torchxrayvision/torchxrayvision_docker.py",
    "lung": "lung_screening/lung_screening.py",
    "neuro": "neuro_seg/neuro_seg.py",
    "flexray": None,
}

# Alternate scripts that still count as "this product" when scanning ps.
ALTERNATE_SCRIPTS: Dict[str, List[str]] = {
    "dental": ["dental_segmentator/dental_segmentator.py"],
    "totalseg": ["total_segmentator/total_segmentator.py"],
    "vox2seglam": ["vox2seglam/vox2seglam.py"],
    "eucaim": ["eucaim/eucaim.py"],
    "txrv": ["torchxrayvision/torchxrayvision.py"],
}


@dataclass(frozen=True)
class ProductInfo:
    id: str
    product: str
    title: str
    summary: str
    enabled: bool
    info_path: Path
    entry_script: Optional[Path]
    alternate_scripts: tuple[Path, ...]

    @property
    def startable(self) -> bool:
        return self.entry_script is not None and self.entry_script.is_file()

    def match_scripts(self) -> List[Path]:
        """Scripts whose cmdline means this product is running."""
        out: List[Path] = []
        if self.entry_script is not None:
            out.append(self.entry_script)
        out.extend(self.alternate_scripts)
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "product": self.product,
            "title": self.title,
            "summary": self.summary,
            "enabled": self.enabled,
            "startable": self.startable,
            "entry_script": (
                str(self.entry_script.relative_to(REPO_ROOT))
                if self.entry_script is not None
                else None
            ),
        }


def _is_product_info(path: Path) -> bool:
    """Top-level ``<product>/<name>.info.json`` only (skip nested skill assets)."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if len(parts) != 2:
        return False
    return parts[1].endswith(".info.json")


def load_products(repo_root: Path = REPO_ROOT) -> List[ProductInfo]:
    """Load product cards sorted by title."""
    products: List[ProductInfo] = []
    for path in sorted(repo_root.glob("*/*.info.json")):
        if not _is_product_info(path):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        product_id = str(data.get("id") or "").strip()
        if not product_id:
            continue
        entry_rel = ENTRYPOINTS.get(product_id)
        entry_script: Optional[Path] = None
        if entry_rel:
            candidate = (repo_root / entry_rel).resolve()
            entry_script = candidate
        alts = tuple(
            (repo_root / rel).resolve()
            for rel in ALTERNATE_SCRIPTS.get(product_id, [])
        )
        products.append(
            ProductInfo(
                id=product_id,
                product=str(data.get("product") or product_id).strip(),
                title=str(data.get("title") or product_id).strip(),
                summary=str(data.get("summary") or "").strip(),
                enabled=bool(data.get("enabled", True)),
                info_path=path.resolve(),
                entry_script=entry_script,
                alternate_scripts=alts,
            )
        )
    products.sort(key=lambda p: (not p.startable, p.title.lower()))
    return products


def get_product(product_id: str, repo_root: Path = REPO_ROOT) -> Optional[ProductInfo]:
    for product in load_products(repo_root):
        if product.id == product_id:
            return product
    return None
