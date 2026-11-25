"""P2: train the autoencoders (G=8 x 3 seeds for the main runs, G=4 and G=16 x 1 seed for the ablation)."""
import glob, json, os, sys

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
    if r.count("/") <= 5:
        print(r, d[:5], f[:8])
prep = glob.glob("/kaggle/temp/**/task_report.json", recursive=True)
assert prep, "prep kernel output not attached"
prep_dir = os.path.dirname(prep[0])
print(open(prep[0]).read())

import torch  # noqa: E402
from taskcodec.config import load_config  # noqa: E402
from taskcodec.pipeline import load_bundle, run_train_ae  # noqa: E402

cfg = load_config(config_dir=os.path.join(code_root, "configs"))
abl = load_config("ablation", config_dir=os.path.join(code_root, "configs"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
bundle = load_bundle(cfg, dev, work_dir=prep_dir)
main_g = cfg["latent"]["groups"]
groups = [main_g] + [g for g in abl["groups"] if g != main_g]
info = run_train_ae(cfg, bundle, groups, lambda g: cfg["ae_seeds"] if g == main_g else cfg["ae_seeds"][:1])
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(info, open("/kaggle/working/ae_training.json", "w"), indent=1)
print(info)
