"""Static checks of the notebooks: schema, no stored outputs, syntax, resolvable `src` imports,
and execution of the setup cell from both documented working directories."""

import ast
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

from conftest import ROOT

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
EXPECTED = {"english_emotion.ipynb", "french_emotion.ipynb", "personality_prediction.ipynb", "llm_inference_finetuning.ipynb"}


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def _module_file(modname: str):
    try:
        spec = importlib.util.find_spec(modname)
    except (ModuleNotFoundError, ValueError):  # e.g. "pkg.module.function" is not a module
        return None
    return Path(spec.origin) if spec and spec.origin else None


def _defined_names(path: Path):
    """Top-level names defined or re-exported by a module (static, no import → works without torch)."""
    names = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def test_expected_notebooks_exist():
    assert EXPECTED <= {p.name for p in NOTEBOOKS}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_valid_clean_and_parsable(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for c in _code_cells(nb):
        assert not c.outputs and c.execution_count is None, "notebooks must be committed without outputs"
        ast.parse(c.source)  # syntax (no magics are used)
    blob = json.dumps(nb)
    for forbidden in ("/content/", "/root/", "PANDORA_Project", "drive/MyDrive", "wandb: Currently logged in", ".netrc"):
        assert forbidden not in blob, f"machine/identity-specific string {forbidden!r} in {path.name}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_src_imports_resolve(path):
    nb = nbformat.read(path, as_version=4)
    for cell in _code_cells(nb):
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "src":
                f = _module_file(node.module)
                assert f is not None, f"module {node.module} not found"
                defined = _defined_names(f)
                for alias in node.names:
                    sub = _module_file(f"{node.module}.{alias.name}")
                    assert alias.name in defined or sub is not None, f"{node.module}.{alias.name} not defined"


@pytest.mark.parametrize("cwd", ["root", "notebooks"])
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_setup_cell_runs_from_documented_directories(path, cwd, tmp_path, monkeypatch):
    """Execute the `setup`-tagged cell in a subprocess with cwd=repo root or notebooks/."""
    import subprocess
    import sys

    nb = nbformat.read(path, as_version=4)
    setup = next(c for c in _code_cells(nb) if "setup" in c.metadata.get("tags", []))
    wd = ROOT if cwd == "root" else ROOT / "notebooks"
    env = {"PATH": "/usr/bin:/bin", "ITC_RESULTS_DIR": str(tmp_path / "res"), "ITC_DATA_DIR": str(tmp_path / "d"),
           "ITC_CACHE_DIR": str(tmp_path / "c"), "ITC_CHECKPOINT_DIR": str(tmp_path / "k"), "MPLBACKEND": "Agg",
           "PYTHONPATH": ""}
    r = subprocess.run([sys.executable, "-c", setup.source], cwd=wd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    assert "repo root:" in r.stdout
