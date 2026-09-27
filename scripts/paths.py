import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TABLES = RESULTS / "tables"
FIGURE_INPUTS = REPO / "figures" / "inputs"
FIGURE_OUT = REPO / "figures" / "out"
CHECKPOINTS = Path(os.environ.get("CHECKPOINT_ROOT", REPO / "checkpoints")).resolve()
MEDTOKENIZERS_ROOT = Path(
    os.environ.get("MEDTOKENIZERS_ROOT", REPO.parent / "medtokenizers")
).resolve()
