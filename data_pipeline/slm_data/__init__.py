"""slm_data — clean, label and pack the candidate/resume corpus for the SLM."""
import sys
from pathlib import Path

# Make this repository's `model/slm_model` (prompt format + tokenizer encoders, shared with the training
# repository so train-time and serve-time inputs are identical; see slm_model/format_version.py) importable.
# Only when no slm_model is importable yet: inside the training repository the full package must win.
import importlib.util as _ilu

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
if _MODEL_DIR.exists() and str(_MODEL_DIR) not in sys.path and _ilu.find_spec("slm_model") is None:
    sys.path.append(str(_MODEL_DIR))

__version__ = "0.1.0"
