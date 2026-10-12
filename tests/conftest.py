"""Shared test helpers.

Equivalence tests compare the refactored code with the *original notebook's own
source code*, extracted from ``archive/original_master_notebook.ipynb`` and
executed.  When that file is absent (it is git-ignored by default) those tests
are skipped, not passed.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_NOTEBOOK = ROOT / "archive" / "original_master_notebook.ipynb"


@pytest.fixture(scope="session")
def original_nb():
    if not ORIGINAL_NOTEBOOK.is_file():
        pytest.skip("original notebook not present in archive/")
    with open(ORIGINAL_NOTEBOOK, encoding="utf-8") as handle:
        return json.load(handle)


def cell_source(nb, idx: int) -> str:
    cell = nb["cells"][idx]
    assert cell["cell_type"] == "code", f"cell {idx} is not a code cell"
    return "".join(cell["source"])


def extract_nodes(nb, idx: int, kinds=(ast.FunctionDef, ast.ClassDef), names=None) -> str:
    """Return the source of selected top-level definitions of a notebook cell."""
    src = cell_source(nb, idx)
    tree = ast.parse(src)
    chunks = [
        ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, kinds) and (names is None or node.name in names)
    ]
    assert chunks, f"nothing extracted from cell {idx} for {names}"
    return "\n\n".join(chunks)


def exec_definitions(nb, idx: int, names, extra_globals=None) -> dict:
    ns = dict(extra_globals or {})
    exec(compile(extract_nodes(nb, idx, names=set(names)), f"<orig cell {idx}>", "exec"), ns)
    return ns


def find_calls(nb, idx: int, func_name: str):
    """All ``func_name(...)`` calls of a cell as ``(args, {kw: literal})`` tuples."""
    tree = ast.parse(cell_source(nb, idx))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == func_name:
                kws = {}
                for kw in node.keywords:
                    try:
                        kws[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        kws[kw.arg] = ast.unparse(kw.value)
                found.append(kws)
    return found
