# herat.py

# ==============================
# Cell 1: imports and global config
# ==============================

import os
import random
from typing import List, Tuple
import soundfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import torchaudio
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_ROOT = "../heart_2016/"

TRAIN_FOLDERS = ["training-a", "training-b", "training-c",
                 "training-d", "training-e", "training-f"]
VAL_FOLDER = "validation"

# ---- 心音采样率 ----
TARGET_SR = 2000

# ---- Mel 频谱参数 ----
# 50 ms 窗长、25 ms hop、128 Mel 滤波，25–1000 Hz 频段
N_MELS = 128
WIN_LENGTH = int(0.05 * TARGET_SR)   # 50 ms -> 100 samples
HOP_LENGTH = int(0.025 * TARGET_SR)  # 25 ms -> 50 samples
N_FFT = 256                          # >= WIN_LENGTH, 2 的幂
F_MIN = 25.0
F_MAX = TARGET_SR / 2.0              # 1000 Hz

# ---- 训练超参数 ----
BATCH_SIZE = 16
NUM_WORKERS = 0   # 避免 dataloader 多进程崩，先设 0
NUM_EPOCHS = 30   # “一阶段”epoch 数：CE-30 / SCA-long 用到
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BEST_MODEL_PATH = "pcg_resnet18_melspec_best.pt"  # CE-30
SEED = 2025

# 三条主实验权重路径
CE_LONG_BEST_PATH      = "pcg_resnet18_melspec_ce_long_best.pt"          # CE-60
SCA_LONG_BEST_PATH     = "pcg_resnet18_melspec_sca_toy_long_best.pt"     # CE-30 -> SCA-30
SCA_ALL_LONG_BEST_PATH = "pcg_resnet18_melspec_sca_all_toy_long_best.pt" # SCA-60 from scratch

# λ 网格（SCA-all 用）
LAMBDA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]

# ---- 随机种子 ----
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==============================
# Cell 2: utilities to parse REFERENCE.csv and raw waveform dataset
# ==============================

def parse_reference_file(ref_path: str) -> List[Tuple[str, int]]:
    """
    解析 REFERENCE.csv
    返回: list of (record_id, label),
          label: 0=normal, 1=abnormal
    带 0 (unsure) 的条目直接丢弃。
    """
    pairs: List[Tuple[str, int]] = []
    with open(ref_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 兼容 "a0001 -1" / "a0002 1" / "a0003,1" / 可能的 "a0001-1"
            line_clean = line.replace(",", " ")
            parts = line_clean.split()
            if len(parts) == 2:
                rec_id, lab_str = parts
            else:
                # fallback：最后两位是标签，其余是 ID
                rec_id = line[:-2].strip()
                lab_str = line[-2:].strip()

            if lab_str == "-1":
                label = 0   # normal
            elif lab_str == "1":
                label = 1   # abnormal
            elif lab_str == "0":
                # unsure -> 丢弃
                continue
            else:
                # 其他奇怪标签直接跳过
                continue

            pairs.append((rec_id, label))
    return pairs


class PhysioNetPCGRaw(Dataset):
    """
    返回原始心音波形的 Dataset。

    每个样本: (waveform: Tensor [T], label: int)
    """

    def __init__(self, root: str, folders: List[str]):
        super().__init__()
        self.root = root
        self.items: List[Tuple[str, int]] = []

        for folder in folders:
            ref_path = os.path.join(root, folder, "REFERENCE.csv")
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"REFERENCE.csv not found in {folder}")

            rec_label_pairs = parse_reference_file(ref_path)
            for rec_id, label in rec_label_pairs:
                wav_path = os.path.join(root, folder, rec_id + ".wav")
                if not os.path.exists(wav_path):
                    # 有些系统可能是 .WAV 大写
                    wav_path_alt = os.path.join(root, folder, rec_id + ".WAV")
                    if os.path.exists(wav_path_alt):
                        wav_path = wav_path_alt
                    else:
                        # 找不到就跳过
                        continue
                self.items.append((wav_path, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, label = self.items[idx]
        waveform, sr = torchaudio.load(wav_path)  # [C, T]

        # 只用单通道
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        waveform = waveform.squeeze(0)  # [T]

        # 如有必要，重采样到 2 kHz
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        # 幅度归一化到 [-1,1]
        max_abs = waveform.abs().max()
        if max_abs > 0:
            waveform = waveform / max_abs

        return waveform, label


# ==============================
# Cell 3: Mel-spectrogram wrapper dataset and collate_fn
# ==============================

class PCGMelDataset(Dataset):
    """
    包一层，将原始 waveform -> log-Mel 频谱图

    输出: (mel: Tensor [1, T_frames, N_MELS], label)
    """

    def __init__(self, base_dataset: Dataset,
                 sample_rate: int = TARGET_SR,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 win_length: int = WIN_LENGTH,
                 hop_length: int = HOP_LENGTH,
                 f_min: float = F_MIN,
                 f_max: float = F_MAX):
        super().__init__()
        self.base = base_dataset
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        waveform, label = self.base[idx]    # waveform: [T]
        waveform = waveform.unsqueeze(0)    # [1, T]

        # [1, n_mels, time]
        melspec = self.melspec(waveform)
        melspec_db = self.amp_to_db(melspec)

        # 转成 [1, time, n_mels]，和 WAM-1D 的 compute_melspec 对齐
        melspec_db = melspec_db.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T_frames, N_MELS]

        return melspec_db, label


def collate_mel_batch(batch):
    """
    collate_fn：对时间维做 padding，使 batch 里所有样本时间长度一致。

    输入: list of (mel [1, T_i, F], label)
    输出:
        mel_padded: [B, 1, T_max, F]
        labels:     [B]
        lengths:    [B]  （原始 T_i）
    """
    mels, labels = zip(*batch)
    lengths = [m.shape[1] for m in mels]
    max_len = max(lengths)
    feat_dim = mels[0].shape[2]

    mel_padded = torch.zeros(len(mels), 1, max_len, feat_dim, dtype=mels[0].dtype)
    for i, m in enumerate(mels):
        T = m.shape[1]
        mel_padded[i, 0, :T, :] = m[0]

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return mel_padded, labels, lengths


# ==============================
# Cell 4: 2D CNN model over Mel-spectrograms
# ==============================

class PCGResNet18(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        base = models.resnet18(weights=None)  # 不用 ImageNet 预训练
        # 第一层改成 1 通道
        base.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # 最后一层改成 2 类
        base.fc = nn.Linear(base.fc.in_features, n_classes)
        self.backbone = base

    def forward(self, x):
        # x: [B, 1, T, F]
        return self.backbone(x)


# ==============================
# Cell 5: build datasets and dataloaders (9:1 split on training-a...f)
# ==============================

# 1) 先把 training-a...f 全部读进来
full_raw = PhysioNetPCGRaw(
    root=DATA_ROOT,
    folders=TRAIN_FOLDERS,
)
num_total = len(full_raw)
print(f"Total recordings in training-a...f: {num_total}")

# 2) 按 9:1 划分成 train_raw / val_raw，使用固定随机种子保证可复现
indices = np.arange(num_total)
rng = np.random.RandomState(SEED)
rng.shuffle(indices)

split = int(0.9 * num_total)
train_indices = indices[:split]
val_indices = indices[split:]

train_raw = Subset(full_raw, train_indices)
val_raw = Subset(full_raw, val_indices)

print(f"Number of training recordings (split 9:1): {len(train_raw)}")
print(f"Number of validation recordings (split 9:1): {len(val_raw)}")

# 3) 包一层 Mel 频谱 dataset
train_ds = PCGMelDataset(train_raw)
val_ds = PCGMelDataset(val_raw)

# 随便看一个 mel 形状
mel_example, label_example = train_ds[0]
print(f"Example mel shape: {mel_example.shape}, label={label_example}")
# 期待形状: [1, T_frames, N_MELS]

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_mel_batch,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_mel_batch,
    pin_memory=True,
)


# ==============================
# Cell 6: training & evaluation helpers
# ==============================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for mel_batch, labels, lengths in loader:
        mel_batch = mel_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(mel_batch)  # [B, 2]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel_batch.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_batch, labels, lengths in loader:
            mel_batch = mel_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(mel_batch)
            loss = criterion(logits, labels)

            running_loss += loss.item() * mel_batch.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["normal(0)", "abnormal(1)"]
    )
    return epoch_loss, epoch_acc, cm, report


def compute_sensitivity_specificity(cm: np.ndarray):
    """
    由 2x2 混淆矩阵计算：
      sensitivity = TP / (TP + FN)  (针对 abnormal=1)
      specificity = TN / (TN + FP)  (针对 normal=0)
    """
    if cm.shape != (2, 2):
        return float("nan"), float("nan")
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec


# ==============================
# Cell 9: 频率先验 SCA (Mel bands) + KL helpers
# ==============================

# 把 128 个 Mel bin 粗分成 4 个“频段 band”
N_BANDS = 4
band_edges = np.linspace(0, N_MELS, N_BANDS + 1, dtype=int)
band_slices = [(int(band_edges[i]), int(band_edges[i + 1])) for i in range(N_BANDS)]
print("Band slices (Mel index):", band_slices)

# 频率知情的 band-level 先验：
#   band1: 低频（约 25–150 Hz）       → S1/S2 主能量
#   band2: 中低频（约 150–300 Hz）   → 早期收缩/舒张杂音
#   band3: 中高频（约 300–600 Hz）   → 杂音高频成分
#   band4: 高频尾部（约 >600 Hz）    → 多为噪声/伪影
PRIOR_NORMAL = [0.7, 0.2, 0.1, 0.0]
PRIOR_ABNORM = [0.15, 0.35, 0.5, 0.0]

print("Freq-informed band prior (normal):  ", PRIOR_NORMAL)
print("Freq-informed band prior (abnormal):", PRIOR_ABNORM)


def compute_gradxinput_attr(mel_batch, logits, labels, create_graph: bool = True):
    """
    mel_batch: [B, 1, T, F], requires_grad=True
    logits:    [B, C]
    labels:    [B]

    返回: attr: [B, 1, T, F]  (Grad×Input 的绝对值)
    """
    B = mel_batch.size(0)

    selected_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
    scalar = selected_logits.sum()

    grads = torch.autograd.grad(
        outputs=scalar,
        inputs=mel_batch,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    attr = grads * mel_batch
    attr = attr.abs()
    return attr


def aggregate_attr_over_bands(attr):
    """
    attr: [B, 1, T, F]

    按照 band_slices 把 attribution 聚合成 band 级别的质量:
      P_attr: [B, N_BANDS]
    """
    # 先对 time 做和
    attr_sum = attr.sum(dim=2)  # [B, 1, F]

    band_masses = []
    for (start, end) in band_slices:
        band_mass = attr_sum[..., start:end].sum(dim=-1)  # [B, 1]
        band_masses.append(band_mass)

    band_masses = torch.cat(band_masses, dim=-1)  # [B, N_BANDS]

    eps = 1e-8
    denom = band_masses.sum(dim=-1, keepdim=True) + eps
    P_attr = band_masses / denom
    return P_attr


def build_freq_prior(labels, device):
    """
    基于 PCG 频谱知识的 band-level 先验.

    labels: [B], 0=normal, 1=abnormal
    返回 prior: [B, N_BANDS]
    """
    B = labels.size(0)
    prior = torch.zeros(B, N_BANDS, device=device)

    p_normal = torch.tensor(PRIOR_NORMAL, device=device, dtype=torch.float32)
    p_abnorm = torch.tensor(PRIOR_ABNORM, device=device, dtype=torch.float32)

    mask_normal = (labels == 0)
    mask_abnorm = (labels == 1)

    if mask_normal.any():
        prior[mask_normal] = p_normal
    if mask_abnorm.any():
        prior[mask_abnorm] = p_abnorm

    prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
    return prior


def build_toy_prior(labels, device):
    """
    为了兼容旧脚本而保留的函数名。
    实际上已经不再是 toy，而是调用频率知情的 build_freq_prior。
    """
    return build_freq_prior(labels, device)


def kl_divergence(P_attr, P_prior):
    """
    逐样本 KL(P_attr || P_prior)，再对 batch 求平均
    """
    eps = 1e-8
    P_attr = P_attr.clamp(min=eps)
    P_prior = P_prior.clamp(min=eps)

    log_ratio = (P_attr.log() - P_prior.log())
    kl = (P_attr * log_ratio).sum(dim=-1)   # [B]
    return kl.mean()


# 默认 λ（给其他脚本用）
lambda_sca = 0.05
print("Default lambda_sca for SCA:", lambda_sca)


def train_one_epoch_sca(model, loader, optimizer, criterion, device, lambda_sca: float = lambda_sca):
    """
    单 epoch 训练：CE + λ · SCA，其中 SCA 是基于 Grad×Input 的 band-level KL 正则。
    """
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_sca = 0.0
    all_preds = []
    all_labels = []

    for mel_batch, labels, lengths in loader:
        mel_batch = mel_batch.to(device, non_blocking=True)   # [B,1,T,F]
        labels = labels.to(device, non_blocking=True)

        mel_batch.requires_grad_(True)

        optimizer.zero_grad()

        logits = model(mel_batch)                  # [B, 2]
        ce_loss = criterion(logits, labels)

        # 频率先验 SCA: Grad×Input + band 聚合 + KL
        attr = compute_gradxinput_attr(mel_batch, logits, labels)   # [B,1,T,F]
        P_attr = aggregate_attr_over_bands(attr)                    # [B, N_BANDS]
        P_prior = build_freq_prior(labels, device)                  # [B, N_BANDS]
        sca_loss = kl_divergence(P_attr, P_prior)

        total_loss = ce_loss + lambda_sca * sca_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * mel_batch.size(0)
        running_ce += ce_loss.item() * mel_batch.size(0)
        running_sca += sca_loss.item() * mel_batch.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_ce = running_ce / len(loader.dataset)
    epoch_sca = running_sca / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_ce, epoch_sca, epoch_acc


# ==============================
# 训练封装 (CE-30, CE-60, SCA-long, SCA-all grid search)
# ==============================

def run_ce_baseline_training(num_epochs=NUM_EPOCHS, save_path=BEST_MODEL_PATH):
    """
    从随机初始化开始，纯 CE 训练 num_epochs（用于 CE-30 baseline）。
    """
    print(f"\n[CE-BASE] Training CE baseline on device: {device}")
    model = PCGResNet18(n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_sens = float("nan")
    best_spec = float("nan")
    best_cm = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [CE-BASE] Epoch {epoch}/{num_epochs} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_cm, val_report = eval_one_epoch(
            model, val_loader, criterion, device
        )
        sens, spec = compute_sensitivity_specificity(val_cm)

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(
            f"Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.4f}, "
            f"sens (abnormal): {sens:.4f}, spec (normal): {spec:.4f}"
        )
        print("Val confusion matrix:\n", val_cm)
        print("Val classification report:\n", val_report)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sens = sens
            best_spec = spec
            best_cm = val_cm.copy()
            torch.save(model.state_dict(), save_path)
            print(f"*** New best CE-BASE model saved to {save_path} (val_acc={best_val_acc:.4f})")

    print("[CE-BASE] Training finished.")
    print(
        f"[CE-BASE] Best val acc: {best_val_acc:.4f}, "
        f"sensitivity: {best_sens:.4f}, specificity: {best_spec:.4f}"
    )
    if best_cm is not None:
        print("Best-val confusion matrix:\n", best_cm)
    return model, best_val_acc, best_sens, best_spec


def run_ce_long_from_best_ce(
    init_ckpt_path=BEST_MODEL_PATH,
    num_epochs: int = 30,
    save_path: str = CE_LONG_BEST_PATH,
):
    """
    CE-60: 以 CE-30 checkpoint 为起点，再跑 num_epochs 纯 CE（默认 30）.
    """
    print(f"\n[CE-60] Loading CE-30 init model from: {init_ckpt_path}")
    model = PCGResNet18(n_classes=2).to(device)
    model.load_state_dict(torch.load(init_ckpt_path, map_location=device))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_sens = float("nan")
    best_spec = float("nan")
    best_cm = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [CE-60] Epoch {epoch}/{num_epochs} (CE-long) =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_cm, val_report = eval_one_epoch(
            model, val_loader, criterion, device
        )
        sens, spec = compute_sensitivity_specificity(val_cm)

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(
            f"Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.4f}, "
            f"sens (abnormal): {sens:.4f}, spec (normal): {spec:.4f}"
        )
        print("Val confusion matrix:\n", val_cm)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sens = sens
            best_spec = spec
            best_cm = val_cm.copy()
            torch.save(model.state_dict(), save_path)
            print(f"*** New best CE-60 model saved to {save_path} (val_acc={best_val_acc:.4f})")

    print("[CE-60] Training finished.")
    print(
        f"[CE-60] Best val acc: {best_val_acc:.4f}, "
        f"sensitivity: {best_sens:.4f}, specificity: {best_spec:.4f}"
    )
    if best_cm is not None:
        print("Best-val confusion matrix:\n", best_cm)
    return model, best_val_acc, best_sens, best_spec


def run_sca_training_from_best_ce(
    init_ckpt_path=BEST_MODEL_PATH,
    num_epochs=NUM_EPOCHS,
    save_path=SCA_LONG_BEST_PATH,
    lambda_sca_val: float = lambda_sca,
):
    """
    SCA-long: CE-30 -> SCA-30.
    从 CE-30 checkpoint 出发，训练 num_epochs 个 epoch 的 SCA 模型。
    """
    print(f"\n[SCA-LONG] Loading CE-30 init model from: {init_ckpt_path}")
    print(f"[SCA-LONG] Using freq-informed band prior:")
    print(f"  normal   = {PRIOR_NORMAL}")
    print(f"  abnormal = {PRIOR_ABNORM}")
    print(f"[SCA-LONG] Using lambda_sca = {lambda_sca_val}")

    model_sca = PCGResNet18(n_classes=2).to(device)
    model_sca.load_state_dict(torch.load(init_ckpt_path, map_location=device))
    model_sca.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model_sca.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc_sca = 0.0
    best_sens = float("nan")
    best_spec = float("nan")
    best_cm = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [SCA-LONG] Epoch {epoch}/{num_epochs} (λ={lambda_sca_val}) =====")

        train_loss, train_ce, train_sca_val, train_acc = train_one_epoch_sca(
            model_sca, train_loader, optimizer, criterion, device,
            lambda_sca=lambda_sca_val
        )
        val_loss, val_acc, val_cm, val_report = eval_one_epoch(
            model_sca, val_loader, criterion, device
        )
        sens, spec = compute_sensitivity_specificity(val_cm)

        print(
            f"Train: total={train_loss:.4f}, CE={train_ce:.4f}, "
            f"SCA={train_sca_val:.4f}, acc={train_acc:.4f}"
        )
        print(
            f"Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, "
            f"sens (abnormal): {sens:.4f}, spec (normal): {spec:.4f}"
        )
        print("Val confusion matrix:\n", val_cm)

        if val_acc > best_val_acc_sca:
            best_val_acc_sca = val_acc
            best_sens = sens
            best_spec = spec
            best_cm = val_cm.copy()
            torch.save(model_sca.state_dict(), save_path)
            print(
                f"*** New best SCA-long model saved to {save_path}, "
                f"val_acc={best_val_acc_sca:.4f}"
            )

    print("[SCA-LONG] Training finished.")
    print(
        f"[SCA-LONG] Best val acc: {best_val_acc_sca:.4f}, "
        f"sensitivity: {best_sens:.4f}, specificity: {best_spec:.4f}"
    )
    if best_cm is not None:
        print("Best-val confusion matrix:\n", best_cm)
    return model_sca, best_val_acc_sca, best_sens, best_spec


def run_sca_all_grid_search(
    num_epochs: int = 60,
    save_path: str = SCA_ALL_LONG_BEST_PATH,
    lambda_candidates=None,
):
    """
    SCA-all: 从随机初始化开始，直接训练 num_epochs 个 epoch 的 SCA 模型，
    并在 lambda_candidates 上做超参搜索。

    返回:
      best_model, global_best_acc, global_best_sens, global_best_spec, global_best_lambda, per_lambda_stats
        其中 per_lambda_stats 是一个列表:
        [(λ, best_acc_λ, best_sens_λ, best_spec_λ, best_epoch_λ), ...]
    """
    if lambda_candidates is None:
        lambda_candidates = LAMBDA_GRID

    print(f"\n[SCA-ALL] Grid search over λ: {lambda_candidates}")
    print(f"[SCA-ALL] Using freq-informed band prior.")
    print(f"[SCA-ALL] Checkpoint path (global best): {save_path}")

    global_best_acc = 0.0
    global_best_lambda = None
    global_best_sens = float("nan")
    global_best_spec = float("nan")
    global_best_state_dict = None
    per_lambda_stats = []

    for lam in lambda_candidates:
        print(f"\n================ [SCA-ALL] λ={lam} =================")
        model_sca = PCGResNet18(n_classes=2).to(device)
        model_sca.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model_sca.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        best_val_acc_this = 0.0
        best_sens_this = float("nan")
        best_spec_this = float("nan")
        best_epoch_this = 0
        best_state_this = None

        for epoch in range(1, num_epochs + 1):
            print(f"\n[SCA-ALL λ={lam}] Epoch {epoch}/{num_epochs}")

            train_loss, train_ce, train_sca_val, train_acc = train_one_epoch_sca(
                model_sca, train_loader, optimizer, criterion, device,
                lambda_sca=lam
            )
            val_loss, val_acc, val_cm, val_report = eval_one_epoch(
                model_sca, val_loader, criterion, device
            )
            sens, spec = compute_sensitivity_specificity(val_cm)

            print(
                f"Train: total={train_loss:.4f}, CE={train_ce:.4f}, "
                f"SCA={train_sca_val:.4f}, acc={train_acc:.4f}"
            )
            print(
                f"Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, "
                f"sens (abnormal): {sens:.4f}, spec (normal): {spec:.4f}"
            )
            print("Val confusion matrix:\n", val_cm)

            if val_acc > best_val_acc_this:
                best_val_acc_this = val_acc
                best_sens_this = sens
                best_spec_this = spec
                best_epoch_this = epoch
                best_state_this = {k: v.detach().cpu() for k, v in model_sca.state_dict().items()}
                print(
                    f"*** New best for λ={lam}: acc={best_val_acc_this:.4f} "
                    f"(epoch {best_epoch_this}), "
                    f"sens={best_sens_this:.4f}, spec={best_spec_this:.4f}"
                )

        per_lambda_stats.append(
            (lam, best_val_acc_this, best_sens_this, best_spec_this, best_epoch_this)
        )

        # 更新 global best
        if best_val_acc_this > global_best_acc:
            global_best_acc = best_val_acc_this
            global_best_lambda = lam
            global_best_sens = best_sens_this
            global_best_spec = best_spec_this
            global_best_state_dict = best_state_this

        del model_sca
        torch.cuda.empty_cache()

    # 打印 per-lambda 结果
    print("\n[SCA-ALL] Grid search finished.")
    print("Per-lambda best results (SCA-ALL, 9:1 split):")
    for lam, acc, sens, spec, epoch in per_lambda_stats:
        print(
            f"  λ={lam}: best val acc={acc:.4f} at epoch {epoch}, "
            f"sens={sens:.4f}, spec={spec:.4f}"
        )

    # 保存 global best 模型
    if global_best_state_dict is not None:
        best_model = PCGResNet18(n_classes=2).to(device)
        best_model.load_state_dict(global_best_state_dict)
        torch.save(best_model.state_dict(), save_path)
        print(
            f"\nGLOBAL best SCA-all val acc: {global_best_acc:.4f} "
            f"(λ={global_best_lambda})"
        )
        print(f"SCA-all checkpoint: {save_path}")
    else:
        best_model = None
        print("[SCA-ALL] WARNING: No best state dict recorded!")

    return best_model, global_best_acc, global_best_sens, global_best_spec, global_best_lambda, per_lambda_stats


# ==============================
# WAM 可视化（仍然基于当前 9:1 val_raw）
# ==============================

from lib.wam_1D import WaveletAttribution1D
import matplotlib.pyplot as plt
import pywt

def visualize_one_abnormal_wam(ckpt_path=BEST_MODEL_PATH):
    # ---- 1) 重新加载 best 模型，并设为 eval ----
    wam_model = PCGResNet18(n_classes=2).to(device)
    wam_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    wam_model.eval()

    print("Loaded model for WAM from:", ckpt_path)

    # ---- 2) Mel 变换（与训练一致）----
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="htk",
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    # ---- 3) 构建 WAM-1D explainer ----
    wavelet_name = "haar"
    levels = 2
    method = "smooth"

    explainer = WaveletAttribution1D(
        wam_model,
        wavelet=wavelet_name,
        J=levels,
        method=method,
        mode="reflect",
        device=device,
        approx_coeffs=False,
        n_mels=N_MELS,
        n_fft=N_FFT,
        sample_rate=TARGET_SR,
        n_samples=25,
        stdev_spread=0.001,
        random_seed=42,
    )

    # ---- 4) 在 val_raw 里找一条 abnormal 样本 ----
    target_idx = None
    print("Searching for an abnormal sample (true=1, pred=1 preferred)...")

    for i in range(len(val_raw)):
        wf_i, lab_i = val_raw[i]
        if lab_i != 1:
            continue

        wf_1ch = wf_i.unsqueeze(0)             # [1, T]
        mel_i = mel_transform(wf_1ch)          # [1, n_mels, time]
        mel_i_db = amp_to_db(mel_i)            # [1, n_mels, time]
        mel_for_model = mel_i_db.permute(0, 2, 1).unsqueeze(0)  # [1,1,time,n_mels]

        with torch.no_grad():
            logits_i = wam_model(mel_for_model.to(device))
            pred_i = logits_i.argmax(dim=1).item()

        if pred_i == 1:
            target_idx = i
            print(f"Found matched abnormal sample at idx={i}, pred={pred_i}")
            break

    if target_idx is None:
        print("No abnormal sample with pred=1 found. Falling back to first true_label=1 sample.")
        for i in range(len(val_raw)):
            wf_i, lab_i = val_raw[i]
            if lab_i == 1:
                target_idx = i
                break

    if target_idx is None:
        raise RuntimeError("Could not find any abnormal sample in val_raw!")

    waveform, true_label = val_raw[target_idx]
    waveform_1ch = waveform.unsqueeze(0)

    print(f"\nSelected sample idx={target_idx}, len={len(waveform)}, true_label={true_label}")

    melspec = mel_transform(waveform_1ch)
    melspec_db = amp_to_db(melspec)
    mel_orig_img = melspec_db.squeeze(0).cpu().numpy()

    melspec_db_for_model = melspec_db.permute(0, 2, 1).unsqueeze(0)
    with torch.no_grad():
        logits = wam_model(melspec_db_for_model.to(device))
        y_pred = logits.argmax(dim=1).item()

    print(f"Model prediction on this sample: pred={y_pred} (0=normal, 1=abnormal)")

    x_wam = waveform.unsqueeze(0)
    melspec_grad, grad_coeffs = explainer(x_wam, y_pred)
    print("WAM mel-grad shape:", melspec_grad.shape)

    waveform_np = waveform.cpu().numpy()
    coeffs_orig = pywt.wavedec(
        waveform_np,
        wavelet=wavelet_name,
        level=levels,
        mode="symmetric",
    )

    EPS = 0.3
    filtered_coeffs = []
    for c, g in zip(coeffs_orig, grad_coeffs):
        g_1d = np.squeeze(g)
        if g_1d.size == 0 or np.allclose(g_1d, 0):
            mask = np.zeros_like(c)
        else:
            L = min(len(c), len(g_1d))
            mask = np.zeros_like(c, dtype=np.float32)
            norm_grad = np.abs(g_1d[:L])
            norm_grad = norm_grad / (norm_grad.max() + 1e-12)
            mask[:L] = (norm_grad > EPS).astype(np.float32)
        filtered_coeffs.append(c * mask)

    expl_waveform_np = pywt.waverec(filtered_coeffs, wavelet=wavelet_name)
    L0 = len(waveform_np)
    if len(expl_waveform_np) > L0:
        expl_waveform_np = expl_waveform_np[:L0]
    elif len(expl_waveform_np) < L0:
        pad = np.zeros(L0, dtype=expl_waveform_np.dtype)
        pad[:len(expl_waveform_np)] = expl_waveform_np
        expl_waveform_np = pad

    max_abs = np.max(np.abs(expl_waveform_np))
    if max_abs > 0:
        expl_waveform_np = expl_waveform_np / max_abs

    expl_waveform = torch.tensor(expl_waveform_np, dtype=torch.float32).unsqueeze(0)
    melspec_expl = mel_transform(expl_waveform)
    melspec_expl_db = amp_to_db(melspec_expl)
    mel_expl_img = melspec_expl_db.squeeze(0).cpu().numpy()

    T_orig = mel_orig_img.shape[1]
    T_expl = mel_expl_img.shape[1]
    T_min = min(T_orig, T_expl)
    mel_orig_img = mel_orig_img[:, :T_min]
    mel_expl_img = mel_expl_img[:, :T_min]

    all_vals = np.concatenate([mel_orig_img.flatten(), mel_expl_img.flatten()])
    vmin = np.percentile(all_vals, 5)
    vmax = np.percentile(all_vals, 99)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    im1 = ax1.imshow(
        mel_orig_img,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(f"Original PCG log-Mel (abnormal)\n(true={true_label}, pred={y_pred})")
    ax1.set_xlabel("Time frames")
    ax1.set_ylabel("Mel bins")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(
        mel_expl_img,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(f"Explanation audio log-Mel\n(wavelet={wavelet_name}, levels={levels}, EPS={EPS})")
    ax2.set_xlabel("Time frames")
    ax2.set_ylabel("Mel bins")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# ==============================
# main: 一次性跑 CE-30 / CE-60 / SCA-long / SCA-all grid search
# ==============================

if __name__ == "__main__":
    # 1) CE-30 baseline
    ce30_model, ce30_acc, ce30_sens, ce30_spec = run_ce_baseline_training(
        num_epochs=NUM_EPOCHS, save_path=BEST_MODEL_PATH
    )

    # 2) CE-60: 在 CE-30 基础上再跑 30 epoch 纯 CE
    ce60_model, ce60_acc, ce60_sens, ce60_spec = run_ce_long_from_best_ce(
        init_ckpt_path=BEST_MODEL_PATH,
        num_epochs=30,
        save_path=CE_LONG_BEST_PATH,
    )

    # 3) SCA-long: CE-30 -> SCA-30 （这里默认 λ=0.05，但你可以自己改）
    sca_long_model, sca_long_acc, sca_long_sens, sca_long_spec = run_sca_training_from_best_ce(
        init_ckpt_path=BEST_MODEL_PATH,
        num_epochs=30,
        save_path=SCA_LONG_BEST_PATH,
        lambda_sca_val=lambda_sca,
    )

    # 4) SCA-all: SCA-60 from scratch + λ grid search
    (
        sca_all_model,
        sca_all_acc,
        sca_all_sens,
        sca_all_spec,
        sca_all_best_lambda,
        sca_all_stats,
    ) = run_sca_all_grid_search(
        num_epochs=60,
        save_path=SCA_ALL_LONG_BEST_PATH,
        lambda_candidates=LAMBDA_GRID,
    )

    print("\n============== SUMMARY (herat.py, 9:1 split on training-a...f) ==============")
    print(
        f"CE-30   (baseline)                  "
        f"best val acc: {ce30_acc:.4f}, sens: {ce30_sens:.4f}, spec: {ce30_spec:.4f}, "
        f"ckpt: {BEST_MODEL_PATH}"
    )
    print(
        f"CE-60   (CE-long, 30+30 CE)         "
        f"best val acc: {ce60_acc:.4f}, sens: {ce60_sens:.4f}, spec: {ce60_spec:.4f}, "
        f"ckpt: {CE_LONG_BEST_PATH}"
    )
    print(
        f"SCA-long(CE-30 -> SCA-30, λ={lambda_sca:.2f})   "
        f"best val acc: {sca_long_acc:.4f}, sens: {sca_long_sens:.4f}, spec: {sca_long_spec:.4f}, "
        f"ckpt: {SCA_LONG_BEST_PATH}"
    )
    print(
        f"SCA-all (SCA-60 from scratch, λ*={sca_all_best_lambda:.2f})   "
        f"best val acc: {sca_all_acc:.4f}, sens: {sca_all_sens:.4f}, spec: {sca_all_spec:.4f}, "
        f"ckpt: {SCA_ALL_LONG_BEST_PATH}"
    )
    print("================================================================")
