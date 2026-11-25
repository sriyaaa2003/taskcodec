"""Speech Commands v2: download (inside Kaggle only), indexing, official splits, SID subset."""
from __future__ import annotations

import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def ensure_dataset(root: str | Path, urls: list[str], min_clips: int) -> Path:
    """Return the dataset directory, downloading it first if needed. Fails loudly."""
    root = Path(root)
    if (root / "testing_list.txt").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "speech_commands_v0.02.tar.gz"
    errors = []
    for url in urls:
        try:
            urllib.request.urlretrieve(url, archive)
            break
        except Exception as e:                       # noqa: BLE001 - report every URL tried
            errors.append(f"{url}: {e!r}")
    else:
        raise RuntimeError("dataset download failed for all URLs:\n" + "\n".join(errors))
    with tarfile.open(archive) as tf:
        tf.extractall(root)
    archive.unlink(missing_ok=True)
    n = sum(1 for _ in root.glob("*/*.wav"))
    if n < min_clips:
        raise RuntimeError(f"only {n} wav files under {root}, expected >= {min_clips}")
    return root


def index_clips(root: str | Path) -> list[dict]:
    """All clips outside _background_noise_, sorted; speaker id from the file name."""
    root = Path(root)
    out = []
    for wdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted(wdir.glob("*.wav")):
            out.append({"rel": f"{wdir.name}/{f.name}", "word": wdir.name,
                        "speaker": f.name.split("_nohash_")[0]})
    return out


def official_split(rels: list[str], val_list: set[str], test_list: set[str]) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, from the official list files."""
    s = np.zeros(len(rels), dtype=np.int8)
    for i, r in enumerate(rels):
        if r in test_list:
            s[i] = 2
        elif r in val_list:
            s[i] = 1
    return s


def read_list(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def build_sid_subset(speakers: list[str], k: int, fracs, seed: int):
    """Top-k speakers by clip count; their clips re-split per speaker into train/val/test.

    Returns (label, split): label = speaker index in 0..k-1 or -1; split = 0/1/2 or -1.
    """
    sp = np.asarray(speakers)
    uniq, counts = np.unique(sp, return_counts=True)
    order = np.lexsort((uniq, -counts))               # by count desc, name asc (deterministic)
    top = uniq[order[:k]]
    lab_of = {s: i for i, s in enumerate(top)}
    label = np.full(len(sp), -1, dtype=np.int64)
    split = np.full(len(sp), -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    cum = np.cumsum(fracs)
    for s, li in lab_of.items():
        idx = np.flatnonzero(sp == s)
        rng.shuffle(idx)
        n = len(idx)
        cut1, cut2 = int(round(cum[0] * n)), int(round(cum[1] * n))
        label[idx] = li
        split[idx[:cut1]] = 0
        split[idx[cut1:cut2]] = 1
        split[idx[cut2:]] = 2
    return label, split


def _read_one(path: Path, n: int) -> np.ndarray:
    from scipy.io import wavfile
    sr, w = wavfile.read(path)
    w = w.astype(np.float32) / 32768.0
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    return w[:n]


def load_wavs(root: Path, rels: list[str], n_samples: int, workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda r: _read_one(root / r, n_samples), rels)))
