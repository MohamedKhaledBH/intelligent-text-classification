"""Project paths and configuration persistence.

Nothing in this repository is tied to the machine the notebook was written on.
All locations are resolved from the repository root and can be overridden with
environment variables:

==========================  ===============================  =========================
Variable                    Meaning                          Default
==========================  ===============================  =========================
``ITC_DATA_DIR``            local datasets (not versioned)   ``<repo>/data``
``ITC_RESULTS_DIR``         tables / figures output          ``<repo>/results``
``ITC_CACHE_DIR``           Hugging Face / embedding cache   ``<repo>/cache``
``ITC_CHECKPOINT_DIR``      model checkpoints                ``<repo>/checkpoints``
==========================  ===============================  =========================
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    cache_dir: Path
    checkpoints_dir: Path

    def ensure(self) -> "ProjectPaths":
        """Create every directory (idempotent) and return ``self``."""
        for path in (
            self.data_dir,
            self.results_dir,
            self.tables_dir,
            self.figures_dir,
            self.cache_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: Optional[os.PathLike] = None, create: bool = False) -> ProjectPaths:
    """Resolve project paths, honouring the ``ITC_*`` environment variables."""
    root_path = Path(root).resolve() if root is not None else REPO_ROOT

    def _resolve(env_name: str, default: Path) -> Path:
        value = os.environ.get(env_name)
        return Path(value).expanduser().resolve() if value else default

    results = _resolve("ITC_RESULTS_DIR", root_path / "results")
    paths = ProjectPaths(
        root=root_path,
        data_dir=_resolve("ITC_DATA_DIR", root_path / "data"),
        results_dir=results,
        tables_dir=results / "tables",
        figures_dir=results / "figures",
        cache_dir=_resolve("ITC_CACHE_DIR", root_path / "cache"),
        checkpoints_dir=_resolve("ITC_CHECKPOINT_DIR", root_path / "checkpoints"),
    )
    return paths.ensure() if create else paths


def save_json_config(config: Mapping[str, Any], path: os.PathLike) -> Path:
    """Persist an experiment configuration (dicts / dataclasses) as JSON.

    Mirrors the original notebook's ``config_v3.json`` export; non-serialisable
    values (``Path``, NumPy scalars…) are stringified instead of raising.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        key: asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
        for key, value in config.items()
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2, default=str)
    return out
