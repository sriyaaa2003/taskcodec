"""P1: download Speech Commands v2 inside Kaggle, cache log-mel, train and freeze KWS + SID models."""
import glob, os, sys

hits = glob.glob("/kaggle/input/**/taskcodec/__init__.py", recursive=True)
assert hits, f"code dataset not attached: {os.listdir('/kaggle/input')}"
code_root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, code_root)

import torch  # noqa: E402
from taskcodec.config import load_config  # noqa: E402
from taskcodec.pipeline import run_prep  # noqa: E402

cfg = load_config(config_dir=os.path.join(code_root, "configs"))
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
run_prep(cfg, "cuda" if torch.cuda.is_available() else "cpu")
