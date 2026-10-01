"""Keep frozen inputs separate from generated paper output."""

import os
from pathlib import Path
import tempfile

PAPER_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PAPER_DIR.parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"


def validate_output_path(value):
    """Require an absolute path outside the repository and OneDrive."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Paper output paths must be absolute")
    path = path.resolve()
    if path == REPOSITORY_DIR or REPOSITORY_DIR in path.parents:
        raise ValueError("Paper output must be outside the checkout")
    if any(part.casefold().startswith("onedrive") for part in path.parts):
        raise ValueError("Paper output must be outside OneDrive")
    return path


def output_root():
    """Return a separate run directory for each paper."""
    if os.environ.get("GBA_PAPER_OUTPUT_PATH"):
        base = validate_output_path(os.environ["GBA_PAPER_OUTPUT_PATH"])
    elif os.environ.get("AGENT_LOCAL_ROOT"):
        base = validate_output_path(Path(os.environ["AGENT_LOCAL_ROOT"]) / "outputs")
    else:
        base = validate_output_path(Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "GoalBasedAllocation")
    return base / PAPER_DIR.name


def figure_directory(value=None):
    """Create an external output directory without replacing approved figures."""
    path = validate_output_path(value) if value is not None else output_root() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
