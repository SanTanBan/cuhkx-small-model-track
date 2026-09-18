"""
Train one modality x one fold. Run once per (modality, fold) on Kaggle.

    python train.py --modality Thermal --fold 0 --epochs 20
    python train.py --modality Thermal --fold -1 --epochs 20   # all users

Writes to out_dir:
    {modality}_f{fold}.pt          best EMA weights (fp32, packed later)
    {modality}_f{fold}_oof.npz     OOF logits + labels + clip_ids
    {modality}_f{fold}_test.npz    test probabilities (TTA-averaged)

With --fold -1 the tag is {modality}_full: every training user, a fixed
schedule and the final EMA weights, and no OOF file because there is no
held-out fold. These are the members that go into the <100 MB package; the
folds exist to measure cross-subject accuracy and fit the fusion weights.

The OOF files are what fusion weights are fitted on — never fit them on the
leaderboard, that is how teams end up with a 0.97 public score and a Selection
Stage failure.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cuhkx import (Cfg, EMA, build_folds, cosine_schedule, load_index,
                   make_dataset, make_model, mixup, predict_logits,
                   seed_everything)


def evaluate(model, loader, device, amp):
    logits, y = predict_logits(model, loader, device, amp)
    acc = (logits.argmax(1) == y).float().mean().item()
    return acc, logits, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/kaggle/input/cuhkx-compact")
    ap.add_argument("--modality", default="Thermal")
    ap.add_argument("--fold", type=int, default=0,
                    help="held-out fold, or -1 to train on every training user")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=144)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--temporal-aug", action="store_true",
                    help="random temporal window / speed augmentation")
    ap.add_argument("--amp-aug", action="store_true",
                    help="skeleton motion-amplitude augmentation")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop training after this long and still write the "
                         "best checkpoint + OOF + test outputs (0 = no limit)")
    args = ap.parse_args()

    cfg = Cfg(data_root=args.data_root, modality=args.modality, fold=args.fold,
              n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size,
              frames=args.frames, size=args.size, arch=args.arch, lr=args.lr,
              seed=args.seed, num_workers=args.workers, out_dir=args.out_dir)
    cfg.extra.update(temporal_aug=args.temporal_aug, amp_aug=args.amp_aug)
    seed_everything(cfg.seed + max(cfg.fold, 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(cfg.to_json())

    mod_dir, idx = load_index(cfg.data_root, cfg.modality)
    train_all = idx[idx.split == "train"].copy()
    test_df = idx[idx.split == "test"].copy()
    train_all["fold"] = build_folds(train_all, cfg.n_folds, cfg.seed)

    if cfg.fold >= 0:
        tr = train_all[train_all.fold != cfg.fold]
        va = train_all[train_all.fold == cfg.fold]
    else:
        tr, va = train_all, train_all.iloc[:0]
    has_val = len(va) > 0
    print(f"\n{cfg.modality} fold {cfg.fold}: "
          f"train {len(tr)} clips / {tr.user.nunique()} users | "
          f"val {len(va)} clips / {va.user.nunique()} users")
    if has_val:
        print(f"val users: {sorted(va.user.astype(str).unique())}")
    print(f"test clips: {len(test_df)}\n")

    dl = lambda ds, sh: DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=sh, persistent_workers=cfg.num_workers > 0)

    train_dl = dl(make_dataset(tr, mod_dir, cfg, train=True), True)
    val_dl = dl(make_dataset(va, mod_dir, cfg, train=False), False) if has_val else None

    model = make_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par/1e6:.2f} M  ({n_par*2/1e6:.1f} MB fp16)\n")

    head = [p for n, p in model.named_parameters()
            if n.startswith(("fc.", "attn."))]
    trunk = [p for n, p in model.named_parameters()
             if not n.startswith(("fc.", "attn."))]
    # Only an ImageNet-pretrained trunk needs a reduced LR; the graph net is
    # trained from scratch, so slowing it down would just under-fit it.
    trunk_mult = cfg.backbone_lr_mult if not cfg.modality.lower().startswith("skel") else 1.0
    opt = torch.optim.AdamW(
        [{"params": trunk, "lr": cfg.lr * trunk_mult},
         {"params": head, "lr": cfg.lr}],
        weight_decay=cfg.weight_decay)

    steps = max(1, len(train_dl)) * cfg.epochs
    warm = int(steps * cfg.warmup_frac)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    ema = EMA(model, cfg.ema_decay)

    best, best_sd, step = -1.0, None, 0
    t_start = time.time()
    limit_s = args.max_minutes * 60 if args.max_minutes > 0 else None
    out_of_time = False
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            if limit_s and time.time() - t_start > limit_s:
                out_of_time = True
                break
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * cosine_schedule(step, steps, warm)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            use_mix = np.random.rand() < cfg.mixup_prob
            if use_mix:
                x, ya, yb, lam = mixup(x, y, cfg.mixup_alpha)

            with torch.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                out = model(x)
                loss = (lam * crit(out, ya) + (1 - lam) * crit(out, yb)
                        if use_mix else crit(out, y))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            ema.update(model)

            tot += loss.item() * y.size(0); seen += y.size(0); step += 1

        eval_model = make_model(cfg, pretrained=False).to(device)
        eval_model.load_state_dict(model.state_dict())
        ema.copy_to(eval_model)

        flag = ""
        if has_val:
            acc, _, _ = evaluate(eval_model, val_dl, device, cfg.amp)
            if acc > best:
                best = acc
                best_sd = {k: v.detach().cpu().clone()
                           for k, v in eval_model.state_dict().items()}
                flag = "  <-- best"
            score = f"val_acc {acc:.4f}  best {best:.4f}"
        else:
            # No held-out users: keep the latest EMA weights.
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in eval_model.state_dict().items()}
            score = "val n/a (full data)"
        print(f"ep {ep+1:2d}/{cfg.epochs}  loss {tot/max(seen,1):.4f}  "
              f"{score}  {time.time()-t0:.0f}s{flag}", flush=True)
        del eval_model
        if out_of_time:
            print(f"time limit of {args.max_minutes:.0f} min reached during epoch "
                  f"{ep+1} — keeping the best checkpoint so far", flush=True)
            break

    tag = f"{cfg.modality}_f{cfg.fold}" if cfg.fold >= 0 else f"{cfg.modality}_full"
    torch.save(best_sd, os.path.join(cfg.out_dir, f"{tag}.pt"))

    # Reload the kept weights, then dump OOF + test predictions for fusion.
    model = make_model(cfg, pretrained=False).to(device)
    model.load_state_dict(best_sd)

    if has_val:
        _, oof_logits, oof_y = evaluate(model, val_dl, device, cfg.amp)
        np.savez(os.path.join(cfg.out_dir, f"{tag}_oof.npz"),
                 logits=oof_logits.numpy(), y=oof_y.numpy(),
                 clip_id=va.clip_id.values, user=va.user.astype(str).values)

    n_views = 1 if args.no_tta else 3
    test_logits = 0
    for v in range(n_views):
        torch.manual_seed(1000 + v)
        np.random.seed(1000 + v)
        # View 0 is the deterministic centre view; extra views re-sample time
        # and jitter the crop. The dataset must be rebuilt per view — mutating
        # a flag on it would never reach forked dataloader workers.
        ds_v = make_dataset(test_df, mod_dir, cfg, train=v > 0)
        dl_v = DataLoader(ds_v, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)
        lg, _ = predict_logits(model, dl_v, device, cfg.amp)
        test_logits = test_logits + lg.softmax(1)
    test_logits = (test_logits / n_views).numpy()

    np.savez(os.path.join(cfg.out_dir, f"{tag}_test.npz"),
             probs=test_logits, clip_id=test_df.clip_id.values)

    verdict = (f"best cross-subject val acc = {best:.4f}" if has_val
               else "full-data model (no held-out users)")
    print(f"\nDONE {tag}: {verdict}")
    with open(os.path.join(cfg.out_dir, "results.txt"), "a") as f:
        f.write(f"{tag}\t{best:.4f}\t{n_par/1e6:.2f}M\n" if has_val
                else f"{tag}\tfull\t{n_par/1e6:.2f}M\n")


if __name__ == "__main__":
    main()
