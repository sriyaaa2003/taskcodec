"""Autoencoder training with random per-group quantization depth and packet-loss augmentation."""
from __future__ import annotations

import time

import torch

from .models import Autoencoder
from .quant import quantize


def spectral_convergence(x_norm, x_hat_norm, normalizer) -> torch.Tensor:
    a = torch.exp(0.5 * normalizer.invert(x_norm).clamp(-20, 15))
    b = torch.exp(0.5 * normalizer.invert(x_hat_norm).clamp(-20, 15))
    return (torch.linalg.vector_norm(a - b, dim=(1, 2)) / torch.linalg.vector_norm(a, dim=(1, 2))).mean()


@torch.no_grad()
def encode_all(ae: Autoencoder, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    ae.eval()
    return torch.cat([ae.encode(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def latent_stats(z: torch.Tensor):
    """mean and std per (group, dim) over clips and time."""
    return z.mean(dim=(0, 1)), z.std(dim=(0, 1)).clamp_min(1e-6)


def train_autoencoder(x_train: torch.Tensor, normalizer, cfg: dict, groups: int, seed: int,
                      log=print, task_models=None) -> tuple[Autoencoder, dict]:
    """x_train: normalised log-mel (N, n_mels, frames) on device. Returns the AE and a log."""
    a, lat, ae_cfg = cfg["audio"], cfg["latent"], cfg["ae"]
    dev = x_train.device
    torch.manual_seed(seed)
    ae = Autoencoder(a["n_mels"], a["frames"], groups, lat["total_dim"], ae_cfg["channels"]).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=ae_cfg["lr"], weight_decay=1e-5)
    bs = ae_cfg["batch_size"]
    steps_per_ep = max(1, len(x_train) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ae_cfg["lr"],
                                                total_steps=ae_cfg["epochs"] * steps_per_ep)
    aug = ae_cfg["aug"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # latent statistics for the quantizer are refreshed each epoch from a train subset
    n_stats = min(len(x_train), ae_cfg["stats_clips"])
    history, t0 = [], time.time()
    mu = sigma = None
    for ep in range(ae_cfg["epochs"]):
        if mu is None or ep % 3 == 0:
            mu, sigma = latent_stats(encode_all(ae, x_train[:n_stats]))
        ae.train()
        perm = torch.randperm(len(x_train), generator=gen).to(dev)
        tot = 0.0
        for i in range(steps_per_ep):
            x = x_train[perm[i * bs:(i + 1) * bs]]
            B = x.shape[0]
            z = ae.encode(x)
            bits = torch.randint(aug["bits_min"], aug["bits_max"] + 1, (B, groups),
                                 generator=gen).to(dev).float()
            bits = torch.where(torch.rand(B, groups, generator=gen).to(dev) < aug["p_zero"],
                               torch.zeros_like(bits), bits)
            zq = quantize(z, bits, mu, sigma, lat["clip_k"], ste=True)
            T = zq.shape[1]
            use = (torch.rand(B, 1, 1, 1, generator=gen) < aug["p_loss_aug"]).to(dev)
            rate = torch.rand(B, 1, 1, 1, generator=gen).to(dev) * aug["loss_max"]
            drop = (torch.rand(B, T, groups, 1, generator=gen).to(dev) < rate) & use
            zq = torch.where(drop, mu.expand_as(zq), zq)
            nest = (torch.rand(B, generator=gen).to(dev) < aug.get("p_nested", 0.0))
            keep = torch.randint(1, groups + 1, (B,), generator=gen).to(dev)       # keep groups < keep
            cut = nest[:, None] & (torch.arange(groups, device=dev)[None, :] >= keep[:, None])
            zq = torch.where(cut[:, None, :, None], mu.expand_as(zq), zq)
            x_hat = ae.decode(zq)
            loss = (x_hat - x).abs().mean() + ae_cfg["sc_weight"] * spectral_convergence(x, x_hat, normalizer)
            if task_models and ae_cfg.get("task_weight", 0.0) > 0:
                tl = 0.0
                for m in task_models:
                    with torch.no_grad():
                        e = m.embed(x)
                    eh = m.embed(x_hat)
                    tl = tl + ((eh - e) ** 2).mean() / (e ** 2).mean().clamp_min(1e-6)
                loss = loss + ae_cfg["task_weight"] * tl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN loss in AE training (G={groups}, seed={seed}, epoch={ep})")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
        history.append(tot / steps_per_ep)
        log(f"  AE G={groups} seed={seed} epoch {ep + 1}/{ae_cfg['epochs']} loss={history[-1]:.4f}")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, {"loss_history": history, "train_seconds": time.time() - t0}
