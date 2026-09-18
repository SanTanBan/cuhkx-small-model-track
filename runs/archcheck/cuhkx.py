"""
CUHK-X Small Model Track — shared training/inference library.

Design notes (why it is built this way):

* The competition is scored cross-subject: test users never appear in training.
  So validation MUST be grouped by user, otherwise CV is meaningless and you
  tune yourself off a cliff. Everything here uses GroupKFold on `user`.

* Backbone is ImageNet-pretrained ResNet18 with Temporal Shift Modules inserted
  into the residual branches. TSM gives 3D-conv-like temporal modelling at 2D
  cost and adds *zero* parameters, which matters under the 100 MB budget.
  Host confirmed ImageNet-pretrained small CNNs are allowed.

* Per-clip intensity normalisation is applied to thermal/IR. Absolute pixel
  level encodes body temperature and ambient conditions — i.e. subject and
  session identity — which is exactly the nuisance variable we must discard to
  generalise across people.

* Weights are exported fp16; a full modality x fold ensemble packs into one
  checkpoint well under 100 MB.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

cv2.setNumThreads(0)

NUM_CLASSES = 40

# A composite input stacks pixel-aligned modalities channel-wise into one clip.
# Depth_Color (3 channels, a colour-mapped depth image) and IR (1 channel) come
# from one sensor, so their person crops align pixel for pixel.
COMPOSITES = {"DepthIR_PC": (("Depth_Color_PC", 3), ("IR_PC", 1))}


def in_channels(modality):
    return sum(c for _, c in COMPOSITES[modality]) if modality in COMPOSITES else 3


# ------------------------------------------------------------------- config

@dataclass
class Cfg:
    data_root: str = "/kaggle/input/cuhkx-compact"
    modality: str = "Thermal"
    frames: int = 16          # frames fed to the model
    size: int = 144           # train crop; stored frames are larger
    arch: str = "resnet18"
    n_folds: int = 5
    fold: int = 0
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    backbone_lr_mult: float = 0.3   # pretrained trunk moves slower than the head
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5
    shift_div: int = 8        # TSM: fraction of channels shifted
    dropout: float = 0.3
    ema_decay: float = 0.999
    warmup_frac: float = 0.1
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 2
    seed: int = 42
    per_clip_norm: bool = True
    out_dir: str = "/kaggle/working"
    extra: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=str)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------- dataset

class BlobReader:
    """Random access into the packed JPEG blobs written by 04_preprocess.py."""

    def __init__(self, mod_dir):
        self.mod_dir = mod_dir
        self._files = {}

    def read(self, blob_id, offset, length):
        f = self._files.get(blob_id)
        if f is None:
            f = open(os.path.join(self.mod_dir, f"blob_{blob_id:03d}.bin"), "rb")
            self._files[blob_id] = f
        f.seek(offset)
        return f.read(length)

    def __getstate__(self):
        # File handles must not cross the fork into dataloader workers.
        return {"mod_dir": self.mod_dir, "_files": {}}

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._files = {}


def load_index(data_root, modality):
    """
    (dir, index) of one packed modality. For a composite, (list of part dirs,
    one index): the first part defines the clips and labels, and every part's
    blob columns are suffixed with its position (blob0, offsets1, t1, ...). A
    part missing for a clip leaves NaN there, which the dataset renders black.
    """
    if modality in COMPOSITES:
        dirs, merged = [], None
        for p, (part, _) in enumerate(COMPOSITES[modality]):
            d = os.path.join(data_root, part)
            df = pd.read_parquet(os.path.join(d, "index.parquet"))
            cols = {c: f"{c}{p}" for c in ("blob", "offsets", "lengths", "t")}
            df = df.rename(columns=cols)
            dirs.append(d)
            merged = df if merged is None else merged.merge(
                df[["clip_id"] + list(cols.values())], on="clip_id", how="left")
        merged["t"] = merged["t0"]
        return dirs, merged
    mod_dir = os.path.join(data_root, modality)
    idx = pd.read_parquet(os.path.join(mod_dir, "index.parquet"))
    return mod_dir, idx


def temporal_window(n, T, min_frac=0.6):
    """
    Training-time temporal crop and speed change: T indices spread uniformly
    over a random contiguous window covering 60-100% of an n-frame clip, with
    sub-step jitter. Indices are non-decreasing and repeat when the window is
    shorter than T.

    Test subjects perform in shorter clips than the training subjects (median
    20 vs 24 source frames, p90 41 vs 56), so their stored samples carry more
    repeated frames and less motion per step. Without this the model sees one
    fixed tempo per clip every epoch, and learns it.
    """
    if n <= 1:
        return [0] * T
    frac = np.random.uniform(min_frac, 1.0)
    span = frac * (n - 1)
    start = np.random.uniform(0.0, (n - 1) - span)
    step = span / max(T - 1, 1)
    pos = start + np.arange(T) * step + np.random.uniform(-0.5, 0.5, T) * step
    return np.clip(np.round(pos), 0, n - 1).astype(int).tolist()


class ClipDataset(Dataset):
    """
    Yields (clip_tensor[T,C,H,W], label).

    Temporal sampling: the preprocessor already stored T_stored uniformly-spaced
    frames. At train time we jitter *within* that grid so the model sees
    different phases of the action across epochs.
    """

    def __init__(self, df, mod_dir, cfg: Cfg, train: bool):
        self.df = df.reset_index(drop=True)
        self.parts = COMPOSITES.get(cfg.modality)
        # a composite reads one blob set per part (load_index returned their dirs)
        self.readers = [BlobReader(d) for d in (mod_dir if self.parts else [mod_dir])]
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.df)

    # -- frame decode -------------------------------------------------------
    def _decode(self, row, take, part=None):
        sfx = "" if part is None else str(part)
        reader = self.readers[part or 0]
        offs, lens = row["offsets" + sfx], row["lengths" + sfx]
        out = []
        for i in take:
            buf = reader.read(int(row["blob" + sfx]), int(offs[i]), int(lens[i]))
            a = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(a, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((self.cfg.size, self.cfg.size, 3), np.uint8)
            out.append(img)
        return out

    def _decode_composite(self, row, take):
        """
        Every part of a frame stacked channel-wise: RGB depth first, then IR
        (stored as three identical channels, so one is kept). A part missing
        for this clip (4 test clips have zero-filled IR files) stays black.
        """
        n0 = int(row["t0"])
        layers = []
        for p, (_, ch) in enumerate(self.parts):
            t = row.get(f"t{p}")
            if t is None or pd.isna(t):
                layers.append(None)
                continue
            t = int(t)
            idx = take if p == 0 else [int(round(i * (t - 1) / max(n0 - 1, 1))) for i in take]
            layers.append([f[:, :, :1] if ch == 1 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                           for f in self._decode(row, idx, part=p)])
        H, W = next(l for l in layers if l is not None)[0].shape[:2]
        out = []
        for k in range(len(take)):
            chans = []
            for p, (_, ch) in enumerate(self.parts):
                f = None if layers[p] is None else layers[p][k]
                if f is None:
                    f = np.zeros((H, W, ch), np.uint8)
                elif f.shape[:2] != (H, W):
                    f = cv2.resize(f, (W, H)).reshape(H, W, ch)
                chans.append(f)
            out.append(np.concatenate(chans, axis=2))
        return out

    def _pick(self, n_stored):
        T = self.cfg.frames
        if self.train and self.cfg.extra.get("temporal_aug"):
            return temporal_window(n_stored, T)
        if n_stored <= T:
            base = list(range(n_stored)) + [n_stored - 1] * (T - n_stored)
            return base
        if self.train:
            # segment-based random sampling (TSN style): one random frame per segment
            edges = np.linspace(0, n_stored, T + 1)
            return [int(np.random.randint(edges[i], max(edges[i] + 1, edges[i + 1])))
                    for i in range(T)]
        edges = np.linspace(0, n_stored, T + 1)
        return [int((edges[i] + edges[i + 1]) / 2) for i in range(T)]

    # -- augmentation -------------------------------------------------------
    def _augment(self, imgs):
        cfg = self.cfg
        H, W = imgs[0].shape[:2]
        if self.train:
            scale = np.random.uniform(0.65, 1.0)
            ar = np.random.uniform(0.85, 1.18)
            ch = int(min(H, H * scale * ar))
            cw = int(min(W, W * scale / ar))
            y0 = np.random.randint(0, H - ch + 1)
            x0 = np.random.randint(0, W - cw + 1)
            flip = np.random.rand() < 0.5
            # brightness/contrast jitter — models sensor gain drift, not identity
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-12, 12)
        else:
            side = int(min(H, W) * 0.90)
            y0 = (H - side) // 2
            x0 = (W - side) // 2
            ch = cw = side
            flip, alpha, beta = False, 1.0, 0.0

        out = []
        for im in imgs:
            im = im[y0:y0 + ch, x0:x0 + cw]
            im = cv2.resize(im, (cfg.size, cfg.size), interpolation=cv2.INTER_LINEAR)
            if flip:
                im = im[:, ::-1]
            if self.train and (alpha != 1.0 or beta != 0.0):
                im = np.clip(im.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            out.append(im)
        return out

    def __getitem__(self, i):
        row = self.df.iloc[i]
        take = self._pick(int(row["t"]))
        imgs = self._decode_composite(row, take) if self.parts else self._decode(row, take)
        imgs = self._augment(imgs)

        x = np.stack(imgs).astype(np.float32) / 255.0     # [T,H,W,C]
        x = torch.from_numpy(x).permute(0, 3, 1, 2)        # [T,C,H,W]

        norm = self.cfg.extra.get("norm") or ("clip" if self.cfg.per_clip_norm else "imagenet")
        if norm == "clip":
            # Standardise each clip independently: removes the absolute thermal /
            # IR offset that identifies the subject and the session.
            m, s = x.mean(), x.std().clamp_min(1e-4)
            x = (x - m) / s
        else:
            mean, std = norm_stats(norm, x.shape[1])
            x = (x - mean) / std

        if self.train and np.random.rand() < 0.25:
            x = random_erase(x)

        return x, int(row["action_id"])


def random_erase(x, max_frac=0.20):
    T, C, H, W = x.shape
    h = int(H * np.random.uniform(0.06, max_frac))
    w = int(W * np.random.uniform(0.06, max_frac))
    y = np.random.randint(0, H - h + 1)
    xx = np.random.randint(0, W - w + 1)
    x[:, :, y:y + h, xx:xx + w] = torch.randn(T, C, h, w) * 0.1
    return x


# --------------------------------------------------------------------- model

class TemporalShift(nn.Module):
    """
    Shift a fraction of channels forward/backward along time before `block`.

    Cost: a memory copy. Params added: zero. This is what buys temporal
    reasoning without paying for 3D convolutions.
    """

    def __init__(self, block, n_segment, shift_div=8):
        super().__init__()
        self.block = block
        self.n_segment = n_segment
        self.shift_div = shift_div

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.n_segment
        n = nt // t
        x = x.view(n, t, c, h, w)
        fold = max(1, c // self.shift_div)
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]              # shift left  (future)
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right (past)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]         # keep the rest
        return self.block(out.view(nt, c, h, w))


def make_tsm_resnet(arch="resnet18", n_segment=16, shift_div=8, pretrained=True):
    import torchvision
    fn = getattr(torchvision.models, arch)
    try:
        net = fn(weights="IMAGENET1K_V1" if pretrained else None)
    except TypeError:
        net = fn(pretrained=pretrained)

    # Wrap the first conv of every BasicBlock so the shift happens on the
    # residual branch only — identity path stays clean (as in the TSM paper).
    for layer in [net.layer1, net.layer2, net.layer3, net.layer4]:
        for blk in layer:
            blk.conv1 = TemporalShift(blk.conv1, n_segment, shift_div)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


class VideoNet(nn.Module):
    """TSM backbone + attention-weighted temporal pooling + linear classifier."""

    def __init__(self, cfg: Cfg, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.backbone, d = make_tsm_resnet(
            cfg.arch, cfg.frames, cfg.shift_div, pretrained)
        self.attn = nn.Sequential(nn.Linear(d, d // 4), nn.Tanh(), nn.Linear(d // 4, 1))
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(d, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):                 # x: [B,T,3,H,W]
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))         # [B*T, d]
        f = f.view(B, T, -1)
        a = self.attn(f).softmax(dim=1)            # [B,T,1]
        pooled = (f * a).sum(1)                    # [B,d]
        return self.fc(self.drop(pooled))


# ------------------------------------------------- video backbones (3D)

KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VIDEO_ARCHS = {"s3d", "mc3_18", "r2plus1d_18", "r3d_18", "r2plus1d_34"}
# R(2+1)D-34 pretrained on IG-65M (65 M Instagram videos), then fine-tuned on
# Kinetics-400 (Ghadiyaram et al., CVPR 2019): weights from Facebook's VMZ,
# converted to PyTorch by moabitcoin/ig65m-pytorch (MIT).
IG65M_KINETICS_URL = ("https://github.com/moabitcoin/ig65m-pytorch/releases/download/"
                      "v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth")


def norm_stats(norm, channels=3):
    """[1,C,1,1] mean/std for 'kinetics' or 'imagenet'. Channels after the first
    three (IR stacked behind RGB depth) take the RGB average."""
    mean, std = (KINETICS_MEAN, KINETICS_STD) if norm == "kinetics" else (IMAGENET_MEAN, IMAGENET_STD)
    extra = channels - 3
    mean = list(mean) + [sum(mean) / 3] * extra
    std = list(std) + [sum(std) / 3] * extra
    return torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1)


def r2plus1d_34(num_classes=400):
    """
    R(2+1)D-34 exactly as the IG-65M release defines it. torchvision's
    VideoResNet sizes the (2+1)D mid-plane of each stage's first block from its
    *input* channels where Caffe2 used the output channels, so those three
    convolutions are rebuilt (288/576/1152 mid-planes) or the pretrained
    weights will not load; BatchNorm takes Caffe2's eps.
    """
    from torchvision.models.video.resnet import (BasicBlock, Conv2Plus1D,
                                                 R2Plus1dStem, VideoResNet)
    net = VideoResNet(block=BasicBlock, conv_makers=[Conv2Plus1D] * 4,
                      layers=[3, 4, 6, 3], stem=R2Plus1dStem, num_classes=num_classes)
    net.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    net.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    net.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
    return net


def adapt_in_channels(conv, channels):
    """First conv for `channels` inputs: the pretrained RGB kernels are kept and
    each extra channel starts as their mean."""
    new = nn.Conv3d(channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(1, keepdim=True).expand(
            -1, channels - 3, -1, -1, -1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def prefetch(arch):
    """Fetch non-torchvision pretrained weights into the hub cache once, before
    parallel jobs race to download the same file."""
    if arch == "r2plus1d_34":
        torch.hub.load_state_dict_from_url(IG65M_KINETICS_URL, map_location="cpu",
                                           progress=False)


class VideoNet3D(nn.Module):
    """
    Kinetics-400 pretrained 3D CNN from torchvision (S3D 8.3 M, MC3-18 11.7 M,
    R(2+1)D-18 31.5 M parameters). Pretraining on human-action video is what
    the ImageNet 2D trunk lacks: its features already separate motions and
    hand-object interactions, where a cross-subject appearance model otherwise
    falls back on memorising the people it was trained on.
    """

    def __init__(self, cfg: Cfg, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        import torchvision.models.video as tvv
        if cfg.arch == "r2plus1d_34":
            net = r2plus1d_34()
            if pretrained:
                net.load_state_dict(torch.hub.load_state_dict_from_url(
                    IG65M_KINETICS_URL, map_location="cpu", progress=False))
        else:
            fn = getattr(tvv, cfg.arch)
            try:
                net = fn(weights="KINETICS400_V1" if pretrained else None)
            except TypeError:
                net = fn(pretrained=pretrained)
        if cfg.arch == "s3d":
            self.backbone, d = net.features, 1024
        else:
            net.fc = nn.Identity()
            self.backbone, d = net, 512
        channels = in_channels(cfg.modality)
        if channels != 3:
            if cfg.arch == "s3d":
                raise ValueError("a multi-channel composite input needs a VideoResNet arch")
            net.stem[0] = adapt_in_channels(net.stem[0], channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(d, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):                     # x: [B,T,3,H,W]
        f = self.backbone(x.permute(0, 2, 1, 3, 4))
        if f.dim() == 5:
            f = self.pool(f).flatten(1)
        return self.fc(self.drop(f))


# ------------------------------------------------------- skeleton (ST-GCN)

# Human3.6M 17-joint bone list; parent[j] is the joint j hangs off.
H36M_EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
              (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
N_JOINTS = 17
PARENT = np.zeros(N_JOINTS, dtype=np.int64)
for _p, _c in H36M_EDGES:
    PARENT[_c] = _p


def build_adjacency():
    """
    Three-partition spatial graph (ST-GCN): self, centripetal (towards root),
    centrifugal (away). Splitting by distance-to-root lets the convolution treat
    "limb moving inward" and "limb moving outward" differently, which matters
    for reach/retract actions like *Take medicine* vs *Put on clothes*.
    """
    A = np.zeros((N_JOINTS, N_JOINTS), np.float32)
    for i, j in H36M_EDGES:
        A[i, j] = A[j, i] = 1.0

    # hop distance from the root joint
    dist = np.full(N_JOINTS, 1e9)
    dist[0] = 0
    for _ in range(N_JOINTS):
        for i, j in H36M_EDGES:
            dist[j] = min(dist[j], dist[i] + 1)
            dist[i] = min(dist[i], dist[j] + 1)

    parts = np.zeros((3, N_JOINTS, N_JOINTS), np.float32)
    parts[0] = np.eye(N_JOINTS, dtype=np.float32)
    for i in range(N_JOINTS):
        for j in range(N_JOINTS):
            if A[i, j] == 0:
                continue
            if dist[j] < dist[i]:
                parts[1, i, j] = 1.0      # neighbour closer to root
            else:
                parts[2, i, j] = 1.0      # neighbour further from root

    # symmetric normalisation, per partition
    for k in range(3):
        d = parts[k].sum(1, keepdims=True)
        parts[k] = parts[k] / np.maximum(d, 1e-6)
    return torch.from_numpy(parts)


def pose_features(x):
    """
    [B,T,V,3] -> [B,9,T,V]: joint position, bone vector, and velocity.

    Bones encode limb orientation independently of where the joint sits, and
    velocity supplies the short-term dynamics that separate the exercise classes
    (jog / squat / jumping jack) from each other.
    """
    B, T, V, C = x.shape
    parent = torch.as_tensor(PARENT, device=x.device)
    bone = x - x[:, :, parent, :]
    vel = torch.zeros_like(x)
    vel[:, 1:] = x[:, 1:] - x[:, :-1]
    f = torch.cat([x, bone, vel], dim=-1)          # [B,T,V,9]
    return f.permute(0, 3, 1, 2).contiguous()      # [B,9,T,V]


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.1, residual=True):
        super().__init__()
        self.register_buffer("A", A)
        K = A.size(0)
        self.gcn = nn.Conv2d(cin, cout * K, 1)
        self.K, self.cout = K, cout
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        # learnable edge weighting — lets the graph adapt beyond the skeleton
        self.edge = nn.Parameter(torch.ones(K, A.size(1), A.size(2)))
        if not residual:
            self.res = None
        elif cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)),
                                     nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = 0 if self.res is None else self.res(x)
        y = self.gcn(x)
        N, _, T, V = y.shape
        y = y.view(N, self.K, self.cout, T, V)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A * self.edge)
        return self.relu(self.tcn(y) + res)


class SkeletonNet(nn.Module):
    """
    Compact ST-GCN over 3D pose. ~1 M parameters, so it costs almost nothing
    against the 100 MB budget while adding a modality that is present for 100%
    of clips and is far more subject-invariant than appearance.
    """

    def __init__(self, num_classes=NUM_CLASSES, width=64, dropout=0.3):
        super().__init__()
        A = build_adjacency()
        w = width
        self.bn = nn.BatchNorm1d(9 * N_JOINTS)
        self.blocks = nn.ModuleList([
            STGCNBlock(9, w, A, residual=False),
            STGCNBlock(w, w, A),
            STGCNBlock(w, w, A),
            STGCNBlock(w, 2 * w, A, stride=2),
            STGCNBlock(2 * w, 2 * w, A),
            STGCNBlock(2 * w, 4 * w, A, stride=2),
            STGCNBlock(4 * w, 4 * w, A),
        ])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * w, num_classes)

    def forward(self, x):                       # x: [B,T,V,3]
        f = pose_features(x)                    # [B,9,T,V]
        B, C, T, V = f.shape
        f = self.bn(f.permute(0, 1, 3, 2).reshape(B, C * V, T))
        f = f.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        for b in self.blocks:
            f = b(f)
        f = f.mean(dim=(2, 3))                  # global average over time+joints
        return self.fc(self.drop(f))


class SkeletonDataset(Dataset):
    """Yields ([T,17,3] pose, label) from the packed poses.npy."""

    def __init__(self, df, mod_dir, cfg: Cfg, train: bool):
        self.df = df.reset_index(drop=True)
        self.poses = np.load(os.path.join(mod_dir, "poses.npy"), mmap_mode="r")
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        seq = np.asarray(self.poses[int(row["row"])], dtype=np.float32)  # [T,V,3]
        T = self.cfg.frames

        if self.cfg.extra.get("center_z"):
            # Also remove the root's height every frame (the original
            # normalisation). On fold 4 it beat keeping height, 0.564 vs
            # 0.535: absolute hip height from the pose estimator varies by
            # subject and recording setup, so it acts as identity, not signal.
            seq[:, :, 2] -= seq[:, 0:1, 2]

        if self.train and self.cfg.extra.get("temporal_aug"):
            seq = seq[temporal_window(seq.shape[0], T)]
        elif seq.shape[0] != T:                     # resample along time
            src = np.linspace(0, seq.shape[0] - 1, T)
            if self.train:
                src = np.clip(src + np.random.uniform(-0.5, 0.5, T), 0,
                              seq.shape[0] - 1)
            seq = seq[np.round(src).astype(int)]

        if self.train and self.cfg.extra.get("amp_aug"):
            # Test subjects move less (lower motion per step, smaller
            # extent), so scale each clip's motion about its mean posture.
            mean = seq.mean(axis=0, keepdims=True)
            seq = mean + np.random.uniform(0.6, 1.15) * (seq - mean)

        if self.train:
            # Rotation about the vertical axis: the camera yaw relative to the
            # subject is arbitrary, so the label must be invariant to it.
            th = np.random.uniform(-0.35, 0.35)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            seq = seq @ R.T
            seq = seq * np.random.uniform(0.9, 1.1)
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            if np.random.rand() < 0.5:              # mirror left/right
                seq[..., 0] *= -1
                swap = np.arange(N_JOINTS)
                for a, b in [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]:
                    swap[a], swap[b] = b, a
                seq = seq[:, swap]

        return torch.from_numpy(np.ascontiguousarray(seq)), int(row["action_id"])


def make_model(cfg: Cfg, pretrained=True):
    """Pose -> graph net; images -> a Kinetics 3D net if cfg.arch names one, else TSM 2D."""
    if cfg.modality.lower().startswith("skel"):
        return SkeletonNet(dropout=cfg.dropout)
    if cfg.arch in VIDEO_ARCHS:
        return VideoNet3D(cfg, pretrained=pretrained)
    return VideoNet(cfg, pretrained=pretrained)


def make_dataset(df, mod_dir, cfg: Cfg, train: bool):
    if cfg.modality.lower().startswith("skel"):
        return SkeletonDataset(df, mod_dir, cfg, train)
    return ClipDataset(df, mod_dir, cfg, train)


# ------------------------------------------------------------------ training

class EMA:
    """
    Exponential moving average of weights, with a warmup ramp on the decay.

    The shadow starts as a copy of the *untrained* weights, so a flat 0.999
    decay leaves it pinned near initialisation for the first ~1000 steps — on a
    short run you would evaluate and checkpoint an essentially untrained model.
    Ramping the decay as (1+t)/(10+t) makes the average track closely at first
    and tighten as training proceeds.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def cosine_schedule(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def build_folds(df, n_folds, seed=42):
    """
    GroupKFold on user. Deterministic, and balanced by user count rather than
    row count so each fold holds out a comparable number of *people*.
    """
    users = sorted(df["user"].astype(str).unique())
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(users))
    assign = {users[u]: i % n_folds for i, u in enumerate(order)}
    return df["user"].astype(str).map(assign).values


@torch.no_grad()
def predict_logits(model, loader, device, amp=True):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            outs.append(model(x).float().cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def pack_fp16(state_dicts, path, meta=None):
    """Bundle every ensemble member into ONE fp16 checkpoint (rule requirement)."""
    packed = {
        name: {k: v.half() if v.dtype.is_floating_point else v
               for k, v in sd.items()}
        for name, sd in state_dicts.items()
    }
    torch.save({"models": packed, "meta": meta or {}}, path)
    mb = os.path.getsize(path) / 1e6
    print(f"checkpoint: {path}  {mb:.1f} MB  ({len(packed)} models)")
    if mb > 100:
        print("!! OVER the 100 MB limit — drop members or shrink the backbone")
    return mb
