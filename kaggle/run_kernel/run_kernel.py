"""P2-P4: E1-E4 and the ablation from the trained autoencoders; writes results/*.json."""
import glob, json, os, shutil, sys, time

hits = glob.glob("/kaggle/input/**/taskcodec/__init__.py", recursive=True)
assert hits, f"code dataset not attached: {os.listdir('/kaggle/input')}"
code_root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, code_root)
import zipfile
def _unzip_outputs():
    for z in glob.glob("/kaggle/input/notebooks/*/*/_output_.zip"):
        dst = "/kaggle/temp/" + z.split("/")[-2]
        if not os.path.isdir(dst):
            os.makedirs(dst, exist_ok=True)
            zipfile.ZipFile(z).extractall(dst)
_unzip_outputs()
for r, d, f in os.walk("/kaggle/temp"):
    if r.count("/") <= 4:
        print(r, d[:5], f[:8])
prep = glob.glob("/kaggle/temp/**/task_report.json", recursive=True)
aes = glob.glob("/kaggle/temp/**/ae_G8_s0.pt", recursive=True) + glob.glob("/kaggle/input/**/ae_G8_s0.pt", recursive=True)
assert prep and aes, "prep / AE kernel outputs not attached"
prep_dir, ae_dir = os.path.dirname(prep[0]), os.path.dirname(aes[0])
cfg_dir = os.path.join(code_root, "configs")
OUT = "/kaggle/working/results"
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(prep_dir, "task_report.json"), os.path.join(OUT, "tasks.json"))
if os.path.exists(os.path.join(ae_dir, "ae_training.json")):
    shutil.copy(os.path.join(ae_dir, "ae_training.json"), os.path.join(OUT, "ae_training.json"))

import subprocess
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pystoi"], capture_output=True, text=True).stdout[-300:])
import torch  # noqa: E402
from taskcodec import experiments as ex  # noqa: E402
from taskcodec.config import load_config  # noqa: E402
from taskcodec.pipeline import load_ae, load_bundle  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
ov = {"paths": {"work_dir": ae_dir}}
base = load_config(overrides=ov, config_dir=cfg_dir)
bundle = load_bundle(base, dev, work_dir=prep_dir)
G = base["latent"]["groups"]
seeds = base["ae_seeds"]
t0 = time.time()
codecs = [ex.Codec(load_ae(base, G, s, dev), bundle, base, G, s) for s in seeds]
print("codecs ready", time.time() - t0)
only = os.environ.get("ONLY")  # not used on Kaggle; for debugging


def cfg_for(name):
    return load_config(name, overrides=ov, config_dir=cfg_dir)


out = __import__("pathlib").Path(OUT)
for name, fn in [("e1", ex.run_e1), ("e2", ex.run_e2), ("e3", ex.run_e3), ("e4", ex.run_e4)]:
    ex.run_experiment(name, fn, codecs, cfg_for(name), out, seeds, bundle.counts)

abl = cfg_for("ablation")
by_g = {}
for g in abl["groups"]:
    if g == G:
        by_g[g] = codecs[:1]
    else:
        by_g[g] = [ex.Codec(load_ae(base, g, seeds[0], dev), bundle, base, g, seeds[0])]
ex.run_experiment("ablation", lambda cs, cf: ex.run_ablation(cs, cf), by_g, abl, out, seeds[:1], bundle.counts)
print("done", os.listdir(OUT))
