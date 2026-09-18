"""
Weight averaging ("model soup") of R(2+1)D-34 video models, checked before use.

    python soup.py --phase A|B|C|D --data-root DIR --ckpt-root DIR --out-dir DIR [--smoke N]

The package holds one R(2+1)D-34 (a second one would break the 100 MB limit), yet
the two self-trained students P (r2p-pl, 128 px, in #15) and R (r2p-r2, 144 px, in
#16) score the same with 56 of 405 fused predictions apart, which is where an
ensemble gains. Both were fine-tuned from the same IG-65M/Kinetics weights with
the same recipe, so their average may keep much of that gain at the size of one
model. Averaging can also fail (a loss barrier between the two), so it is tested:

Phase A, held-out users: the fold-0 models a = r2p-a (128 px), b = r2p-b (144 px)
  and p = r2p-pl (128 px, self-trained). The soups a+b, p+b and a+p are scored on
  fold 0 at 128 and 144 px, with the averaged BatchNorm statistics and with
  statistics re-estimated on the training folds, next to the endpoints and their
  prediction ensembles.
Phase B, the full-data students: the soup P+R with both BatchNorm variants at 144
  and 128 px. Its fit on a training subset (a loss barrier shows up as lost fit),
  and its test probabilities with train.py's three-view protocol; P and R go
  through the same protocol to confirm it reproduces their saved predictions.
Phase C, held-out users, three-way: a, b and p averaged together, uniformly and
  with p counted twice, at 128 and 144 px (averaged BatchNorm statistics): does
  averaging weaker models into a strong one help? Next to the endpoints and the
  matching prediction ensembles.
Phase D, the full-data students after round 3: the four-student average of P, R,
  R3A (144 px) and R3B (128 px), and the same with the teacher-era full models
  A (r2p-a) and B (r2p-b) added at half weight, at 144 px with averaged BatchNorm
  statistics: fit on a training subset, and test probabilities with train.py's
  three-view protocol.

--smoke N runs N clips per split and a reduced set of models, for a local CPU check.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cuhkx import Cfg, build_folds, load_index, make_dataset, make_model

MOD = "DepthIR_PC"
BN = torch.nn.modules.batchnorm._BatchNorm
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


def find_ckpt(root, run, tag, mod=MOD):
    """<mod>_<tag>.pt of one training run: runs/<run>/ locally, the output of
    kernel cuhkx-<run> on Kaggle."""
    pat = re.compile(rf"(^|[\\/-]){re.escape(run)}([\\/]|$)")
    hits = sorted(p for p in glob.glob(os.path.join(root, "**", f"{mod}_{tag}.pt"), recursive=True)
                  if pat.search(os.path.dirname(p)))
    if not hits:
        raise SystemExit(f"no {mod}_{tag}.pt for {run} under {root}")
    if len(hits) > 1:
        log(f"several {mod}_{tag}.pt for {run}, using {hits[0]}: {hits}")
    return hits[0]


def model_cfg(data_root, size, workers):
    """The students' training config (runs/r2p-*/log_*.txt); only the size differs."""
    cfg = Cfg(data_root=data_root, modality=MOD, frames=16, size=size, arch="r2plus1d_34",
              batch_size=16, num_workers=workers, seed=42)
    cfg.extra.update(temporal_aug=True, amp_aug=False, center_z=False, norm="kinetics")
    return cfg


def soup(*sds, weights=None):
    """Weighted average (uniform by default), rounded to fp16 as the checkpoints are stored."""
    w = [1.0] * len(sds) if weights is None else [float(x) for x in weights]
    total = sum(w)
    return {k: (sum(wi * sd[k].float() for wi, sd in zip(w, sds)) / total).half()
            if sds[0][k].is_floating_point() else sds[0][k].clone() for k in sds[0]}


def build(cfg, sd, device):
    m = make_model(cfg, pretrained=False)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return m.to(device).eval()


def loader(df, mod_dir, cfg, train):
    # The loader train.py builds for its test views, so augmentation draws match.
    return DataLoader(make_dataset(df, mod_dir, cfg, train=train), batch_size=cfg.batch_size,
                      shuffle=False, num_workers=cfg.num_workers, pin_memory=True)


@torch.no_grad()
def reestimate_bn(models, dl, device):
    """BatchNorm running statistics recomputed as a plain average over augmented
    training batches (train-mode forward, no weight update), rounded to fp16."""
    if not models:
        return 0
    for m in models:
        for mod in m.modules():
            if isinstance(mod, BN):
                mod.reset_running_stats()
                mod.momentum = None
        m.train()
    n = 0
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            for m in models:
                m(x)
        n += len(x)
    for m in models:
        for mod in m.modules():
            if isinstance(mod, BN):
                mod.running_mean.copy_(mod.running_mean.half().float())
                mod.running_var.copy_(mod.running_var.half().float())
        m.eval()
    return n


@torch.no_grad()
def predict(models, dl, device):
    """Softmax of every model on the same batches."""
    out, ys = {k: [] for k in models}, []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            for k, m in models.items():
                out[k].append(m(x).float().softmax(1).cpu())
        ys.append(y)
    return {k: torch.cat(v).numpy() for k, v in out.items()}, torch.cat(ys).numpy()


def score(p, y):
    p = p.astype(np.float64)
    return dict(acc=round(float((p.argmax(1) == y).mean()), 4),
                nll=round(float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()), 4))


def test_views(models, test_df, mod_dir, cfg, device, out_dir, prefix):
    """train.py's three test views (centre, then two seeded augmented views),
    averaged and saved in its format."""
    total = {k: 0 for k in models}
    for v in range(3):
        torch.manual_seed(1000 + v)
        np.random.seed(1000 + v)
        pr, _ = predict(models, loader(test_df, mod_dir, cfg, v > 0), device)
        for k, p in pr.items():
            total[k] = total[k] + p
        log(f"{prefix} {cfg.size}px test view {v} done")
    for k, t in total.items():
        np.savez(os.path.join(out_dir, f"{MOD}_{k}_test.npz"), probs=t / 3,
                 clip_id=test_df.clip_id.values)


def fold0(a):
    mod_dir, idx = load_index(a.data_root, MOD)
    tr_all = idx[idx.split == "train"].copy()
    tr_all["fold"] = build_folds(tr_all, 5, 42)
    va, tr = tr_all[tr_all.fold == 0], tr_all[tr_all.fold != 0]
    if a.smoke:
        va = va.iloc[:a.smoke]
    tag = "full" if a.smoke else "f0"     # the fold-0 checkpoints live only on Kaggle
    sds = {k: torch.load(find_ckpt(a.ckpt_root, run, tag), map_location="cpu")
           for k, run in (("a", "r2p-a"), ("b", "r2p-b"), ("p", "r2p-pl"))}
    return mod_dir, va, tr, sds


def phase_a(a, device):
    mod_dir, va, tr, sds = fold0(a)
    bn_df = tr.sample(n=min(a.bn_clips, len(tr)), random_state=0)
    own = {"a": 128, "b": 144, "p": 128}
    pairs = {"a+b": ("a", "b")} if a.smoke else {"a+b": ("a", "b"), "p+b": ("p", "b"),
                                                  "a+p": ("a", "p")}
    soups = {k: soup(sds[x], sds[z]) for k, (x, z) in pairs.items()}
    log(f"A: fold 0 = {len(va)} clips / {va.user.nunique()} users; BatchNorm clips {len(bn_df)}")
    res, probs, y = {}, {}, None
    for size in (128, 144):
        cfg = model_cfg(a.data_root, size, a.workers)
        mine = [k for k in soups if not (k == "a+p" and size == 144)]
        specs = ([(f"{k} re{size}", soups[k]) for k in mine]
                 + [(k, sds[k]) for k in own if own[k] == size]
                 + [(f"{k} avg{size}", soups[k]) for k in mine])
        if a.smoke:
            specs = specs[:2]
        models = {k: build(cfg, sd, device) for k, sd in specs}
        re_models = [m for k, m in models.items() if " re" in k]
        torch.manual_seed(0)
        np.random.seed(0)
        n = reestimate_bn(re_models, loader(bn_df, mod_dir, cfg, True), device)
        log(f"A {size}px: BatchNorm re-estimated for {len(re_models)} soups on {n} clips")
        pr, y = predict(models, loader(va, mod_dir, cfg, False), device)
        for k, p in pr.items():
            res[k] = score(p, y)
            probs[k] = p.astype(np.float16)
            log(f"A {k:<12} fold 0: acc {res[k]['acc']:.4f}  nll {res[k]['nll']:.4f}")
        del models, re_models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for k, (x, z) in pairs.items():
        if x in probs and z in probs:
            ens = (probs[x].astype(np.float32) + probs[z].astype(np.float32)) / 2
            res[f"{k} prediction ensemble"] = score(ens, y)
            log(f"A {k} prediction ensemble: acc {res[f'{k} prediction ensemble']['acc']:.4f}")
    with open(os.path.join(a.out_dir, "soupA_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(a.out_dir, "soupA_probs.npz"), y=y,
                        clip_id=va.clip_id.astype(str).values,
                        **{re.sub(r"\W", "_", k): v for k, v in probs.items()})


def phase_b(a, device):
    mod_dir, idx = load_index(a.data_root, MOD)
    tr_all = idx[idx.split == "train"].copy()
    test_df = idx[idx.split == "test"].copy()
    bn_df = tr_all.sample(n=min(a.bn_clips, len(tr_all)), random_state=0)
    fit_df = tr_all.sample(n=min(a.fit_clips, len(tr_all)), random_state=1)
    if a.smoke:
        test_df, fit_df = test_df.iloc[:a.smoke], fit_df.iloc[:a.smoke]
    P = torch.load(find_ckpt(a.ckpt_root, "r2p-pl", "full"), map_location="cpu")
    R = torch.load(find_ckpt(a.ckpt_root, "r2p-r2", "full"), map_location="cpu")
    S = soup(P, R)
    log(f"B: test {len(test_df)} clips; training fit subset {len(fit_df)}; BatchNorm clips {len(bn_df)}")
    res = {}
    for size, name, end in ((144, "R", R), (128, "P", P)):
        cfg = model_cfg(a.data_root, size, a.workers)
        specs = [(f"soup_re{size}", S), (f"repro_{name}{size}", end), (f"soup_avg{size}", S)]
        if a.smoke:
            specs = specs[:2]
        models = {k: build(cfg, sd, device) for k, sd in specs}
        re_model = models[f"soup_re{size}"]
        torch.manual_seed(0)
        np.random.seed(0)
        n = reestimate_bn([re_model], loader(bn_df, mod_dir, cfg, True), device)
        torch.save({k: v.half() if v.is_floating_point() else v
                    for k, v in re_model.state_dict().items()
                    if ".running_" in k or k.endswith("num_batches_tracked")},
                   os.path.join(a.out_dir, f"soup_re{size}_bn.pt"))
        log(f"B {size}px: BatchNorm re-estimated on {n} training clips")
        pr, y = predict(models, loader(fit_df, mod_dir, cfg, False), device)
        for k, p in pr.items():
            res[f"{k} training fit"] = score(p, y)
            log(f"B {k:<14} training fit: acc {res[f'{k} training fit']['acc']:.4f}  "
                f"nll {res[f'{k} training fit']['nll']:.4f}")
        test_views(models, test_df, mod_dir, cfg, device, a.out_dir, "B")
        del models, re_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(os.path.join(a.out_dir, "soupB_results.json"), "w") as f:
        json.dump(res, f, indent=2)


def phase_c(a, device):
    mod_dir, va, _, sds = fold0(a)
    own = {"a": 128, "b": 144, "p": 128}
    mixes = {"a+b+p": {"a": 1, "b": 1, "p": 1}, "a+b+2p": {"a": 1, "b": 1, "p": 2}}
    if a.smoke:
        mixes = {"a+b+p": mixes["a+b+p"]}
    soups = {k: soup(*[sds[m] for m in w], weights=list(w.values())) for k, w in mixes.items()}
    log(f"C: fold 0 = {len(va)} clips / {va.user.nunique()} users")
    res, probs, y = {}, {}, None
    for size in (128, 144):
        cfg = model_cfg(a.data_root, size, a.workers)
        specs = ([(f"{k} avg{size}", s) for k, s in soups.items()]
                 + [(k, sds[k]) for k in own if own[k] == size])
        if a.smoke:
            specs = specs[:2]
        models = {k: build(cfg, sd, device) for k, sd in specs}
        pr, y = predict(models, loader(va, mod_dir, cfg, False), device)
        for k, p in pr.items():
            res[k] = score(p, y)
            probs[k] = p
            log(f"C {k:<14} fold 0: acc {res[k]['acc']:.4f}  nll {res[k]['nll']:.4f}")
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for k, w in mixes.items():
        if all(m in probs for m in w):
            ens = sum(wt * probs[m].astype(np.float64) for m, wt in w.items()) / sum(w.values())
            res[f"{k} prediction ensemble"] = score(ens, y)
            log(f"C {k} prediction ensemble: acc {res[f'{k} prediction ensemble']['acc']:.4f}")
    with open(os.path.join(a.out_dir, "soupC_results.json"), "w") as f:
        json.dump(res, f, indent=2)


def phase_d(a, device):
    mod_dir, idx = load_index(a.data_root, MOD)
    tr_all = idx[idx.split == "train"].copy()
    test_df = idx[idx.split == "test"].copy()
    fit_df = tr_all.sample(n=min(a.fit_clips, len(tr_all)), random_state=1)
    if a.smoke:
        test_df, fit_df = test_df.iloc[:a.smoke], fit_df.iloc[:a.smoke]
    src = {"P": ("r2p-pl", MOD), "R": ("r2p-r2", MOD),
           "R3A": ("r2p-r3", "DepthIR_R3A"), "R3B": ("r2p-r3", "DepthIR_R3B"),
           "A": ("r2p-a", MOD), "B": ("r2p-b", MOD)}
    mixes = {"S4": {"P": 1, "R": 1, "R3A": 1, "R3B": 1},
             "S6": {"P": 2, "R": 2, "R3A": 2, "R3B": 2, "A": 1, "B": 1}}
    if a.smoke:                  # the round-3 checkpoints live only on Kaggle
        src.update(R3A=("r2p-r2", MOD), R3B=("r2p-pl", MOD))
        mixes = {"S4": mixes["S4"]}
    sds = {k: torch.load(find_ckpt(a.ckpt_root, src[k][0], "full", src[k][1]), map_location="cpu")
           for k in sorted({m for w in mixes.values() for m in w})}
    cfg = model_cfg(a.data_root, 144, a.workers)
    models = {}
    for k, w in mixes.items():
        models[f"soup_{k}_avg144"] = build(cfg, soup(*[sds[m] for m in w],
                                                     weights=list(w.values())), device)
        log(f"D {k} = " + " + ".join(f"{wt}x{m}" for m, wt in w.items()))
    del sds
    log(f"D: test {len(test_df)} clips; training fit subset {len(fit_df)}")
    res = {}
    pr, y = predict(models, loader(fit_df, mod_dir, cfg, False), device)
    for k, p in pr.items():
        res[f"{k} training fit"] = score(p, y)
        log(f"D {k:<18} training fit: acc {res[f'{k} training fit']['acc']:.4f}  "
            f"nll {res[f'{k} training fit']['nll']:.4f}")
    test_views(models, test_df, mod_dir, cfg, device, a.out_dir, "D")
    with open(os.path.join(a.out_dir, "soupD_results.json"), "w") as f:
        json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bn-clips", type=int, default=1024)
    ap.add_argument("--fit-clips", type=int, default=640)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.bn_clips = a.smoke
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    log(f"phase {a.phase} on {device}")
    {"A": phase_a, "B": phase_b, "C": phase_c, "D": phase_d}[a.phase](a, device)
    log(f"phase {a.phase} finished")


if __name__ == "__main__":
    main()
