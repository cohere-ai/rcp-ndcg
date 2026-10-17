"""The torch-free import surface of ``rcp_ndcg_core.irt``.

Importing ``rcp_ndcg_core.irt`` needs numpy and pydantic only, so the rubric
scoring and the insertion primitive work on a bare ``rcp-ndcg-core`` install;
the two torch-backed estimator classes (``BradleyTerryEstimator``,
``RaschEstimator``) are read lazily and import torch when first touched, exactly
as the tournament engine and Bradley-Terry do. Nothing else guards that
property: an eager ``import torch`` creeping into the package would pass every
torch-ful test while breaking bare installs. This test blocks torch in a fresh
subprocess and asserts the split.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: The light package's source in THIS checkout (tests/core/irt/ -> repo root). The
#: probe runs with this as its cwd, so `python -c` puts it first on sys.path and the
#: probe imports the tree under test -- a hard-coded checkout path probed a
#: different tree from every worktree and every CI runner.
CORE_SRC = Path(__file__).resolve().parents[3] / "rcp-ndcg-core" / "src"

TORCH_BLOCKED_PROBE = r"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"torch blocked for this probe: {name}")
        return None

sys.meta_path.insert(0, _Block())

import rcp_ndcg_core.metric  # noqa: F401 - numpy-only, must import
import rcp_ndcg_core
import rcp_ndcg_core.irt as irt
from rcp_ndcg_core.schemas import ItemParams

print("CORE-FILE", rcp_ndcg_core.__file__)

for name in irt.__all__:
    try:
        getattr(irt, name)
    except ImportError as exc:
        assert name in ("BradleyTerryEstimator", "RaschEstimator"), (name, exc)
        assert "torch" in str(exc) or "blocked" in str(exc), (name, exc)
items = ItemParams(gamma=(4.0, 5.0, 3.5), beta=(1.0, 2.0, 3.0))
score = irt.score_document(items, 4, [4, 2, 0])
assert 1.0 < score.theta < 3.0, score
try:
    irt.fit_bradley_terry([("a", "b", 1.0, 1.0)], l2=1e-4)
except ImportError:
    pass
else:
    raise AssertionError("Bradley-Terry ran while torch was blocked")
print("TORCH-FREE OK")
"""


def test_criteria_2pl_imports_without_torch_and_torch_names_raise() -> None:
    assert (CORE_SRC / "rcp_ndcg_core" / "__init__.py").is_file(), CORE_SRC
    proc = subprocess.run(
        [sys.executable, "-c", TORCH_BLOCKED_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=CORE_SRC,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "TORCH-FREE OK" in proc.stdout
    # The probe must have imported the light package from this checkout, not from
    # whatever tree an editable install or a stale cwd happens to point at.
    imported = next(line.split(" ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("CORE-FILE "))
    assert Path(imported).resolve().is_relative_to(CORE_SRC), (imported, CORE_SRC)
