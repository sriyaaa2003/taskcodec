"""Copy the package and configs into code_dataset/ (the folder pushed as the private Kaggle dataset)."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
dst = Path(__file__).resolve().parent / "code_dataset"
for name in ("taskcodec", "configs"):
    if (dst / name).exists():
        shutil.rmtree(dst / name)
    shutil.copytree(root / name, dst / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
print("synced", sorted(p.name for p in dst.iterdir()))
