"""
Train the IMU stream on this laptop's CPU (a ~0.9 M-parameter 1-D ResNet).

    python scripts/imu_train.py --out runs/imu-a --epochs 80

Reads compact/IMU/{imu.npy, index.parquet} (scripts/imu_pack.py) and writes the
same files kaggle/train.py does, so fusion and packaging treat IMU like any
other stream: IMU_f{k}.pt / _oof.npz / _test.npz for the five user-grouped
folds (cuhkx.build_folds — the same folds as every other stream), IMU_full.pt
+ IMU_full_test.npz, log_IMU_f{k}.txt (config JSON first, read by
kaggle_ops.member_cfg) and results.txt. The final-epoch EMA weights are kept —
no epoch is picked on the held-out fold — so out-of-fold accuracy stays honest.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
from cuhkx import EMA, Cfg, build_folds, cosine_schedule, make_model, seed_everything  # noqa: E402
from prep import IMU_DEVICES  # noqa: E402

D = len(IMU_DEVICES)


def augment(x, rng):
    """
    x: [B, C, T] raw units (6 channels per device, then D presence channels).
    A random 70-100% temporal window resampled back to T, per-device amplitude
    scaling, sensor noise, and dropping a whole device (presence set to 0),
    which also teaches the model to cope with clips missing a sensor.
    """
    B, C, T = x.shape
    frac = rng.uniform(0.7, 1.0, B)
    start = rng.uniform(0, 1 - frac)
    pos = (start[:, None] + frac[:, None] * np.linspace(0, 1, T)[None]) * (T - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo)[:, None, :]
    bi = np.arange(B)[:, None]
    x = x[bi, :, lo].transpose(0, 2, 1) * (1 - w) + x[bi, :, hi].transpose(0, 2, 1) * w
    x = np.ascontiguousarray(x)
    sig = x[:, :6 * D].reshape(B, D, 6, T)
    sig *= rng.uniform(0.85, 1.15, (B, D, 1, 1))
    sig += rng.normal(0, 0.02, sig.shape) * sig.std(axis=(0, 3), keepdims=True)
    drop = rng.rand(B, D) < 0.10
    sig[drop] = 0.0
    pres = x[:, 6 * D:]
    pres[drop] = 0.0
    return x.astype(np.float32)


def predict(net, X, bs=256):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(torch.from_numpy(X[i:i + bs]))
                          for i in range(0, len(X), bs)]).numpy()


def fit(cfg, Xtr, ytr, mu, sd, log):
    seed_everything(cfg.seed + max(cfg.fold, 0))
    rng = np.random.RandomState(cfg.seed + max(cfg.fold, 0))
    net = make_model(cfg)
    net.mu.copy_(torch.from_numpy(mu).view(1, -1, 1))
    net.sd.copy_(torch.from_numpy(sd).view(1, -1, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    per_epoch = len(Xtr) // cfg.batch_size
    steps = per_epoch * cfg.epochs
    warm = int(steps * cfg.warmup_frac)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    ema, step, t0 = EMA(net, cfg.ema_decay), 0, time.time()
    for ep in range(cfg.epochs):
        net.train()
        perm, tot = rng.permutation(len(Xtr)), 0.0
        for b in range(per_epoch):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            xb = torch.from_numpy(augment(Xtr[idx].copy(), rng))
            yb = torch.from_numpy(ytr[idx])
            for g in opt.param_groups:
                g["lr"] = cfg.lr * cosine_schedule(step, steps, warm)
            loss = crit(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(net)
            tot += loss.item()
            step += 1
        if (ep + 1) % 20 == 0 or ep == cfg.epochs - 1:
            log(f"ep {ep + 1:3d}/{cfg.epochs}  loss {tot / max(per_epoch, 1):.4f}  "
                f"{time.time() - t0:.0f}s")
    final = make_model(cfg)
    final.load_state_dict(net.state_dict())
    ema.copy_to(final)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "compact", "IMU"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "imu-a"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--folds", default="0,1,2,3,4,-1")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    idx = pd.read_parquet(os.path.join(args.data, "index.parquet"))
    X = np.load(os.path.join(args.data, "imu.npy")).astype(np.float32)       # [N, C, T]
    tr_mask = (idx.split == "train").values
    train_df, test_df = idx[tr_mask].copy(), idx[~tr_mask]
    Xall, Xte = X[tr_mask], X[~tr_mask]
    yall = train_df.action_id.values.astype(np.int64)
    train_df["fold"] = build_folds(train_df, 5, 42)
    # Normalisation from training users only; presence channels stay 0/1.
    mu = Xall.mean(axis=(0, 2)).astype(np.float32)
    sd = (Xall.std(axis=(0, 2)) + 1e-3).astype(np.float32)
    mu[6 * D:], sd[6 * D:] = 0.0, 1.0

    for fold in [int(f) for f in args.folds.split(",")]:
        cfg = Cfg(modality="IMU", arch="imu1d", frames=int(X.shape[2]), fold=fold,
                  epochs=args.epochs, batch_size=64, lr=args.lr, weight_decay=0.05,
                  label_smoothing=0.1, dropout=0.3, ema_decay=0.995, warmup_frac=0.05,
                  grad_clip=5.0, out_dir=args.out)
        cfg.extra.update(imu_ch=int(X.shape[1]), imu_width=args.width)
        tag = f"IMU_f{fold}" if fold >= 0 else "IMU_full"
        lf = open(os.path.join(args.out, f"log_IMU_f{fold}.txt"), "w")

        def log(msg, tag=tag, lf=lf):
            print(f"[{tag}] {msg}", flush=True)
            lf.write(msg + "\n")
            lf.flush()

        lf.write(cfg.to_json() + "\n")
        sel = (train_df.fold != fold).values if fold >= 0 else np.ones(len(train_df), bool)
        net = fit(cfg, Xall[sel], yall[sel], mu, sd, log)
        torch.save({k: v.detach().clone() for k, v in net.state_dict().items()},
                   os.path.join(args.out, f"{tag}.pt"))
        n_par = sum(p.numel() for p in net.parameters())
        if fold >= 0:
            va = ~sel
            logits = predict(net, Xall[va])
            acc = float((logits.argmax(1) == yall[va]).mean())
            np.savez(os.path.join(args.out, f"{tag}_oof.npz"), logits=logits, y=yall[va],
                     clip_id=train_df.clip_id.values[va],
                     user=train_df.user.astype(str).values[va])
            log(f"held-out users {sorted(train_df.user[va].astype(str).unique())}: acc {acc:.4f}")
            line = f"{tag}\t{acc:.4f}\t{n_par / 1e6:.2f}M\n"
        else:
            line = f"{tag}\tfull\t{n_par / 1e6:.2f}M\n"
        te = torch.from_numpy(predict(net, Xte)).softmax(1).numpy()
        np.savez(os.path.join(args.out, f"{tag}_test.npz"), probs=te,
                 clip_id=test_df.clip_id.values)
        with open(os.path.join(args.out, "results.txt"), "a") as fh:
            fh.write(line)
        lf.close()


if __name__ == "__main__":
    main()
