#
#
# import os
# import random
# from typing import List, Tuple, Dict
#
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
#
# import torchaudio
# import torchvision.models as models
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#
# # ---- 数据路径（按你本地情况改）----
# # 假设结构类似：
# #   /path/to/circor-heart-sound-1.0.3/
# #       training_data.csv
# #       training_data/
# #           2530_AV.wav, 2530_PV.wav, ...
# CIRCOR_ROOT = "../heart_2022/training_data"
# TRAIN_CSV_PATH = os.path.join(os.path.dirname(CIRCOR_ROOT), "training_data.csv")
#
# # ---- 心音采样率 ----
# TARGET_SR = 2000   # CirCor 原始 4k，这里和 2016 一样统一重采到 2k
#
# # ---- Mel 频谱参数 ----
# N_MELS = 128
# WIN_LENGTH = int(0.05 * TARGET_SR)   # 50 ms
# HOP_LENGTH = int(0.025 * TARGET_SR)  # 25 ms
# N_FFT = 256                          # >= WIN_LENGTH, power of 2
# F_MIN = 25.0
# F_MAX = TARGET_SR / 2.0
#
# # ---- 训练超参数 ----
# BATCH_SIZE = 16
# NUM_WORKERS = 0
# NUM_EPOCHS = 30  # CE-30 / SCA-long 的 epoch 数；SCA-all 用 60
# LEARNING_RATE = 1e-3
# WEIGHT_DECAY = 1e-4
# SEED = 2025
#
# # ---- SCA λ（固定 0.3）----
# lambda_sca = 0.3
#
# # ---- 权重文件名（全部带 2）----
# BEST_MODEL_PATH2       = "pcg_resnet18_melspec_best2.pt"            # CE-30
# CE_LONG_BEST_PATH2     = "pcg_resnet18_melspec_ce_long_best2.pt"    # CE-60
# SCA_LONG_BEST_PATH2    = "pcg_resnet18_melspec_sca_long_best2.pt"   # CE-30 -> SCA-30
# SCA_ALL_BEST_PATH2     = "pcg_resnet18_melspec_sca_all_best2.pt"    # SCA-60 from scratch
#
# # ---- 随机种子 ----
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
#
#
# # ==============================
# # Cell 2: 解析 CirCor training_data.csv + 构建 subject-level item 列表
# # ==============================
#
# def load_subject_labels_from_csv(csv_path: str) -> Dict[str, int]:
#     """
#     从 training_data.csv 里读取每个 Patient ID 的 Outcome：
#       - Normal   -> 0
#       - Abnormal -> 1
#     返回: {subject_id(str): label(int)}
#     """
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"training_data.csv not found at {csv_path}")
#
#     subj2label: Dict[str, int] = {}
#     with open(csv_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             # 跳过表头行
#             if line.startswith("Patient ID"):
#                 continue
#
#             # 同时兼容逗号分隔 & 空格分隔
#             if "," in line:
#                 parts = [p.strip() for p in line.split(",")]
#             else:
#                 parts = line.split()
#
#             if len(parts) < 3:
#                 continue
#
#             subj_id = parts[0]  # 第一列是 Patient ID
#             outcome = parts[-3] # 倒数第 3 列通常是 Outcome (Normal / Abnormal)
#
#             if outcome not in ("Normal", "Abnormal"):
#                 # 极少数行可能是别的值，直接跳过
#                 continue
#
#             label = 0 if outcome == "Normal" else 1
#             if subj_id in subj2label and subj2label[subj_id] != label:
#                 print(f"Warning: conflicting labels for subject {subj_id}: {subj2label[subj_id]} vs {label}")
#             subj2label[subj_id] = label
#
#     print(f"Loaded subject labels from csv: {len(subj2label)} subjects.")
#     return subj2label
#
#
# def build_circor_items_by_subject(
#     wav_root: str,
#     csv_path: str,
# ) -> Tuple[List[Tuple[str, int, str]], List[str]]:
#     """
#     扫描 wav_root 下的 .wav 文件，结合 training_data.csv 的 subject label，
#     构建 item 列表（recording-level），同时保证是 subject-level 标签。
#
#     返回:
#       items: list of (wav_path, label(int), subject_id(str))
#       subject_ids: list of unique subject_id（按字典序排序）
#     """
#     subj2label = load_subject_labels_from_csv(csv_path)
#
#     # 列出所有 wav 文件
#     wav_files = [
#         f for f in os.listdir(wav_root)
#         if f.lower().endswith(".wav")
#     ]
#     if not wav_files:
#         print(f"[WARN] No .wav files found under {wav_root}")
#
#     items: List[Tuple[str, int, str]] = []
#     used_subjects = set()
#
#     for fname in wav_files:
#         base = os.path.splitext(fname)[0]  # e.g., "2530_AV" or "50032_TV_2"
#         subj_id = base.split("_")[0]       # "2530"
#
#         if subj_id not in subj2label:
#             # 这条 subject 不在 training_data.csv（理论上很少发生）
#             continue
#
#         label = subj2label[subj_id]
#         wav_path = os.path.join(wav_root, fname)
#         items.append((wav_path, label, subj_id))
#         used_subjects.add(subj_id)
#
#     subject_ids = sorted(list(used_subjects))
#     print(f"Total recordings found in {wav_root}: {len(items)}")
#     print(f"Total subjects with at least one recording: {len(subject_ids)}")
#
#     return items, subject_ids
#
#
# # ==============================
# # Cell 3: raw waveform Dataset（subject-level划分）
# # ==============================
#
# class CirCorPCGRaw(Dataset):
#     """
#     CirCor 数据集上，记录级的 Dataset。
#     items: list of (wav_path, label, subject_id)
#     每个样本: (waveform[T], label)
#     """
#
#     def __init__(self, items: List[Tuple[str, int, str]]):
#         super().__init__()
#         self.items = items
#
#     def __len__(self) -> int:
#         return len(self.items)
#
#     def __getitem__(self, idx: int):
#         wav_path, label, subj_id = self.items[idx]
#         waveform, sr = torchaudio.load(wav_path)  # [C, T]
#
#         # 单通道
#         if waveform.shape[0] > 1:
#             waveform = waveform[0:1, :]
#         waveform = waveform.squeeze(0)  # [T]
#
#         # 重采样到 2k
#         if sr != TARGET_SR:
#             resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
#             waveform = resampler(waveform)
#
#         # 归一化
#         max_abs = waveform.abs().max()
#         if max_abs > 0:
#             waveform = waveform / max_abs
#
#         return waveform, label
#
#
# # ==============================
# # Cell 4: Mel-spectrogram Dataset & collate_fn
# # ==============================
#
# class PCGMelDataset(Dataset):
#     """
#     waveform -> log-Mel 频谱图
#     输出: (mel: [1, T_frames, N_MELS], label)
#     """
#
#     def __init__(self, base_dataset: CirCorPCGRaw,
#                  sample_rate: int = TARGET_SR,
#                  n_mels: int = N_MELS,
#                  n_fft: int = N_FFT,
#                  win_length: int = WIN_LENGTH,
#                  hop_length: int = HOP_LENGTH,
#                  f_min: float = F_MIN,
#                  f_max: float = F_MAX):
#         super().__init__()
#         self.base = base_dataset
#         self.melspec = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=n_fft,
#             win_length=win_length,
#             hop_length=hop_length,
#             n_mels=n_mels,
#             f_min=f_min,
#             f_max=f_max,
#             power=2.0,
#             center=True,
#             pad_mode="reflect",
#             norm="slaney",
#             mel_scale="htk",
#         )
#         self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
#
#     def __len__(self) -> int:
#         return len(self.base)
#
#     def __getitem__(self, idx: int):
#         waveform, label = self.base[idx]
#         waveform = waveform.unsqueeze(0)  # [1,T]
#
#         melspec = self.melspec(waveform)       # [1, n_mels, time]
#         melspec_db = self.amp_to_db(melspec)   # [1, n_mels, time]
#         melspec_db = melspec_db.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T_frames, N_MELS]
#
#         return melspec_db, label
#
#
# def collate_mel_batch(batch):
#     """
#     输入: list of (mel [1,T_i,F], label)
#     输出:
#       mel_padded: [B,1,T_max,F]
#       labels:     [B]
#       lengths:    [B]
#     """
#     mels, labels = zip(*batch)
#     lengths = [m.shape[1] for m in mels]
#     max_len = max(lengths)
#     feat_dim = mels[0].shape[2]
#
#     mel_padded = torch.zeros(len(mels), 1, max_len, feat_dim, dtype=mels[0].dtype)
#     for i, m in enumerate(mels):
#         T = m.shape[1]
#         mel_padded[i, 0, :T, :] = m[0]
#
#     labels = torch.tensor(labels, dtype=torch.long)
#     lengths = torch.tensor(lengths, dtype=torch.long)
#     return mel_padded, labels, lengths
#
#
# # ==============================
# # Cell 5: 2D CNN model (ResNet18)
# # ==============================
#
# class PCGResNet18(nn.Module):
#     def __init__(self, n_classes: int = 2):
#         super().__init__()
#         base = models.resnet18(weights=None)
#         base.conv1 = nn.Conv2d(
#             in_channels=1,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False,
#         )
#         base.fc = nn.Linear(base.fc.in_features, n_classes)
#         self.backbone = base
#
#     def forward(self, x):
#         # x: [B,1,T,F]
#         return self.backbone(x)
#
#
# # ==============================
# # Cell 6: build subject-level train/val datasets and loaders
# # ==============================
#
# # 1) 记录级 item 列表 & subject 列表
# all_items, all_subject_ids = build_circor_items_by_subject(
#     wav_root=CIRCOR_ROOT,
#     csv_path=TRAIN_CSV_PATH,
# )
#
# if len(all_items) == 0:
#     raise RuntimeError(
#         "No recordings found. "
#         "Please check CIRCOR_ROOT and TRAIN_CSV_PATH paths in herat2.py."
#     )
#
# # 2) subject-level 9:1 划分
# rng = np.random.RandomState(SEED)
# subject_ids_shuffled = all_subject_ids.copy()
# rng.shuffle(subject_ids_shuffled)
#
# n_subj = len(subject_ids_shuffled)
# n_train_subj = int(0.9 * n_subj)
# train_subj_ids = set(subject_ids_shuffled[:n_train_subj])
# val_subj_ids = set(subject_ids_shuffled[n_train_subj:])
#
# train_items = []
# val_items = []
# for wav_path, label, subj_id in all_items:
#     if subj_id in train_subj_ids:
#         train_items.append((wav_path, label, subj_id))
#     elif subj_id in val_subj_ids:
#         val_items.append((wav_path, label, subj_id))
#     # 理论上所有 subj_id 都在 train 或 val 两端，此处不 else
#
# print(f"Total recordings: {len(all_items)}")
# print(f"Number of training recordings (subject-level 9:1): {len(train_items)}")
# print(f"Number of validation recordings (subject-level 9:1): {len(val_items)}")
#
# # 3) Dataset + DataLoader（subject-level 分好以后，再做 record-level 训练）
# train_raw = CirCorPCGRaw(train_items)
# val_raw = CirCorPCGRaw(val_items)
#
# train_ds = PCGMelDataset(train_raw)
# val_ds = PCGMelDataset(val_raw)
#
# mel_example, label_example = train_ds[0]
# print(f"Example mel shape: {mel_example.shape}, label={label_example}")
#
# train_loader = DataLoader(
#     train_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     collate_fn=collate_mel_batch,
#     pin_memory=True,
# )
#
# val_loader = DataLoader(
#     val_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     collate_fn=collate_mel_batch,
#     pin_memory=True,
# )
#
#
# # ==============================
# # Cell 7: training & evaluation helpers (含 sens/spec)
# # ==============================
#
# def compute_sens_spec(cm: np.ndarray):
#     """
#     cm: 2x2 confusion matrix, order [[TN, FP],[FN, TP]]
#     """
#     if cm.shape != (2, 2):
#         return float("nan"), float("nan")
#     TN, FP, FN, TP = cm.ravel()
#     sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#     spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
#     return sens, spec
#
#
# def train_one_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []
#
#     for mel_batch, labels, lengths in loader:
#         mel_batch = mel_batch.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#
#         optimizer.zero_grad()
#         logits = model(mel_batch)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * mel_batch.size(0)
#         preds = logits.argmax(dim=1).detach().cpu().numpy()
#         all_preds.append(preds)
#         all_labels.append(labels.cpu().numpy())
#
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#     epoch_loss = running_loss / len(loader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     return epoch_loss, epoch_acc
#
#
# def eval_one_epoch(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for mel_batch, labels, lengths in loader:
#             mel_batch = mel_batch.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)
#
#             logits = model(mel_batch)
#             loss = criterion(logits, labels)
#
#             running_loss += loss.item() * mel_batch.size(0)
#             preds = logits.argmax(dim=1).detach().cpu().numpy()
#             all_preds.append(preds)
#             all_labels.append(labels.cpu().numpy())
#
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#     epoch_loss = running_loss / len(loader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
#     sens, spec = compute_sens_spec(cm)
#     report = classification_report(
#         all_labels, all_preds,
#         target_names=["Normal(0)", "Abnormal(1)"]
#     )
#     return epoch_loss, epoch_acc, cm, report, sens, spec
#
#
# # ==============================
# # Cell 8: 频率先验 SCA + KL
# # ==============================
#
# # 4 个 Mel band
# N_BANDS = 4
# band_edges = np.linspace(0, N_MELS, N_BANDS + 1, dtype=int)
# band_slices = [(int(band_edges[i]), int(band_edges[i + 1])) for i in range(N_BANDS)]
# print("Band slices (Mel index):", band_slices)
#
# # 和 2016 一致的频率先验
# PRIOR_NORMAL = [0.7, 0.2, 0.1, 0.0]
# PRIOR_ABNORM = [0.15, 0.35, 0.5, 0.0]
# print("Freq-informed band prior (normal):  ", PRIOR_NORMAL)
# print("Freq-informed band prior (abnormal):", PRIOR_ABNORM)
# print("Default lambda_sca for SCA:", lambda_sca)
#
#
# def compute_gradxinput_attr(mel_batch, logits, labels, create_graph: bool = True):
#     """
#     Grad×Input attribution: [B,1,T,F]
#     """
#     selected_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
#     scalar = selected_logits.sum()
#     grads = torch.autograd.grad(
#         outputs=scalar,
#         inputs=mel_batch,
#         create_graph=create_graph,
#         retain_graph=create_graph,
#     )[0]
#     attr = grads * mel_batch
#     return attr.abs()
#
#
# def aggregate_attr_over_bands(attr):
#     """
#     attr: [B,1,T,F]  ->  P_attr: [B, N_BANDS]
#     """
#     attr_sum = attr.sum(dim=2)  # [B,1,F]
#     band_masses = []
#     for (start, end) in band_slices:
#         band_mass = attr_sum[..., start:end].sum(dim=-1)  # [B,1]
#         band_masses.append(band_mass)
#
#     band_masses = torch.cat(band_masses, dim=-1)  # [B,N_BANDS]
#     eps = 1e-8
#     denom = band_masses.sum(dim=-1, keepdim=True) + eps
#     P_attr = band_masses / denom
#     return P_attr
#
#
# def build_freq_prior(labels, device):
#     B = labels.size(0)
#     prior = torch.zeros(B, N_BANDS, device=device)
#     p_normal = torch.tensor(PRIOR_NORMAL, device=device, dtype=torch.float32)
#     p_abnorm = torch.tensor(PRIOR_ABNORM, device=device, dtype=torch.float32)
#
#     mask_normal = (labels == 0)
#     mask_abnorm = (labels == 1)
#     if mask_normal.any():
#         prior[mask_normal] = p_normal
#     if mask_abnorm.any():
#         prior[mask_abnorm] = p_abnorm
#
#     prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
#     return prior
#
#
# def build_toy_prior(labels, device):
#     # 兼容旧脚本名字 —— 实际调用 freq-informed prior
#     return build_freq_prior(labels, device)
#
#
# def kl_divergence(P_attr, P_prior):
#     eps = 1e-8
#     P_attr = P_attr.clamp(min=eps)
#     P_prior = P_prior.clamp(min=eps)
#     log_ratio = (P_attr.log() - P_prior.log())
#     kl = (P_attr * log_ratio).sum(dim=-1)
#     return kl.mean()
#
#
# def train_one_epoch_sca(
#     model, loader, optimizer, criterion, device,
#     lambda_sca_val: float = lambda_sca,
# ):
#     model.train()
#     running_loss = 0.0
#     running_ce = 0.0
#     running_sca = 0.0
#     all_preds = []
#     all_labels = []
#
#     for mel_batch, labels, lengths in loader:
#         mel_batch = mel_batch.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#
#         mel_batch.requires_grad_(True)
#         optimizer.zero_grad()
#
#         logits = model(mel_batch)
#         ce_loss = criterion(logits, labels)
#
#         attr = compute_gradxinput_attr(mel_batch, logits, labels, create_graph=True)
#         P_attr = aggregate_attr_over_bands(attr)
#         P_prior = build_freq_prior(labels, device)
#         sca_loss = kl_divergence(P_attr, P_prior)
#
#         total_loss = ce_loss + lambda_sca_val * sca_loss
#         total_loss.backward()
#         optimizer.step()
#
#         running_loss += total_loss.item() * mel_batch.size(0)
#         running_ce += ce_loss.item() * mel_batch.size(0)
#         running_sca += sca_loss.item() * mel_batch.size(0)
#
#         preds = logits.argmax(dim=1).detach().cpu().numpy()
#         all_preds.append(preds)
#         all_labels.append(labels.detach().cpu().numpy())
#
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#
#     epoch_loss = running_loss / len(loader.dataset)
#     epoch_ce = running_ce / len(loader.dataset)
#     epoch_sca = running_sca / len(loader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#
#     return epoch_loss, epoch_ce, epoch_sca, epoch_acc
#
#
# # ==============================
# # Cell 9: 训练封装 (CE-30 / CE-60 / SCA-long / SCA-all, λ=0.3)
# # ==============================
#
# def run_ce_baseline_training2(num_epochs=NUM_EPOCHS, save_path=BEST_MODEL_PATH2):
#     print(f"\n[CE-30] Training CE baseline on device: {device}")
#     model = PCGResNet18(n_classes=2).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )
#
#     best_val_acc = 0.0
#     best_sens = 0.0
#     best_spec = 0.0
#
#     for epoch in range(1, num_epochs + 1):
#         print(f"\n===== [CE-30] Epoch {epoch}/{num_epochs} =====")
#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, optimizer, criterion, device
#         )
#         val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
#             model, val_loader, criterion, device
#         )
#
#         print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
#         print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}, sens: {sens:.4f}, spec: {spec:.4f}")
#         print("Val confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_sens = sens
#             best_spec = spec
#             torch.save(model.state_dict(), save_path)
#             print(f"*** New best CE-30 model saved to {save_path} "
#                   f"(val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f})")
#
#     print("[CE-30] Training finished.")
#     print(f"[CE-30] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
#     return model, best_val_acc, best_sens, best_spec
#
#
# def run_ce_long_from_best_ce2(
#     init_ckpt_path=BEST_MODEL_PATH2,
#     num_epochs=30,
#     save_path=CE_LONG_BEST_PATH2,
# ):
#     print(f"\n[CE-60] Loading CE-30 init model from: {init_ckpt_path}")
#     model = PCGResNet18(n_classes=2).to(device)
#     model.load_state_dict(torch.load(init_ckpt_path, map_location=device))
#     model.train()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )
#
#     best_val_acc = 0.0
#     best_sens = 0.0
#     best_spec = 0.0
#
#     for epoch in range(1, num_epochs + 1):
#         print(f"\n===== [CE-60] Epoch {epoch}/{num_epochs} =====")
#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, optimizer, criterion, device
#         )
#         val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
#             model, val_loader, criterion, device
#         )
#
#         print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
#         print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}, sens: {sens:.4f}, spec: {spec:.4f}")
#         print("Val confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_sens = sens
#             best_spec = spec
#             torch.save(model.state_dict(), save_path)
#             print(f"*** New best CE-60 model saved to {save_path} "
#                   f"(val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f})")
#
#     print("[CE-60] Training finished.")
#     print(f"[CE-60] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
#     return model, best_val_acc, best_sens, best_spec
#
#
# def run_sca_training_from_best_ce2(
#     init_ckpt_path=BEST_MODEL_PATH2,
#     num_epochs=NUM_EPOCHS,
#     save_path=SCA_LONG_BEST_PATH2,
#     lambda_sca_val: float = lambda_sca,
# ):
#     print(f"\n[SCA-LONG] Loading CE-30 init model from: {init_ckpt_path}")
#     print(f"[SCA-LONG] Using freq-informed band prior:")
#     print(f"  normal   = {PRIOR_NORMAL}")
#     print(f"  abnormal = {PRIOR_ABNORM}")
#     print(f"[SCA-LONG] Using lambda_sca = {lambda_sca_val}")
#
#     model_sca = PCGResNet18(n_classes=2).to(device)
#     model_sca.load_state_dict(torch.load(init_ckpt_path, map_location=device))
#     model_sca.train()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         model_sca.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )
#
#     best_val_acc = 0.0
#     best_sens = 0.0
#     best_spec = 0.0
#
#     for epoch in range(1, num_epochs + 1):
#         print(f"\n===== [SCA-LONG] Epoch {epoch}/{num_epochs} (λ={lambda_sca_val}) =====")
#         train_loss, train_ce, train_sca_val, train_acc = train_one_epoch_sca(
#             model_sca, train_loader, optimizer, criterion, device,
#             lambda_sca_val=lambda_sca_val,
#         )
#         val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
#             model_sca, val_loader, criterion, device
#         )
#
#         print(
#             f"Train: total={train_loss:.4f}, CE={train_ce:.4f}, "
#             f"SCA={train_sca_val:.4f}, acc={train_acc:.4f}"
#         )
#         print(f"Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, sens={sens:.4f}, spec={spec:.4f}")
#         print("Val confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_sens = sens
#             best_spec = spec
#             torch.save(model_sca.state_dict(), save_path)
#             print(f"*** New best SCA-long model saved to {save_path}, "
#                   f"val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f}")
#
#     print("[SCA-LONG] Training finished.")
#     print(f"[SCA-LONG] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
#     return model_sca, best_val_acc, best_sens, best_spec
#
#
# def run_sca_all_from_scratch2(
#     num_epochs: int = 60,
#     save_path: str = SCA_ALL_BEST_PATH2,
#     lambda_sca_val: float = lambda_sca,
# ):
#     print(f"\n[SCA-ALL] Training from scratch with freq-informed SCA.")
#     print(f"[SCA-ALL] Using lambda_sca = {lambda_sca_val}")
#     print(f"[SCA-ALL] Checkpoint path: {save_path}")
#
#     model_sca = PCGResNet18(n_classes=2).to(device)
#     model_sca.train()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         model_sca.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )
#
#     best_val_acc = 0.0
#     best_sens = 0.0
#     best_spec = 0.0
#
#     for epoch in range(1, num_epochs + 1):
#         print(f"\n===== [SCA-ALL] Epoch {epoch}/{num_epochs} (λ={lambda_sca_val}) =====")
#         train_loss, train_ce, train_sca_val, train_acc = train_one_epoch_sca(
#             model_sca, train_loader, optimizer, criterion, device,
#             lambda_sca_val=lambda_sca_val,
#         )
#         val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
#             model_sca, val_loader, criterion, device
#         )
#
#         print(
#             f"Train: total={train_loss:.4f}, CE={train_ce:.4f}, "
#             f"SCA={train_sca_val:.4f}, acc={train_acc:.4f}"
#         )
#         print(f"Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, sens={sens:.4f}, spec={spec:.4f}")
#         print("Val confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_sens = sens
#             best_spec = spec
#             torch.save(model_sca.state_dict(), save_path)
#             print(f"*** New best SCA-all model saved to {save_path}, "
#                   f"val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f}")
#
#     print("[SCA-ALL] Training finished.")
#     print(f"[SCA-ALL] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
#     return model_sca, best_val_acc, best_sens, best_spec
#
#
# # ==============================
# # main: 一次性跑 CE-30 / CE-60 / SCA-long / SCA-all（subject-level split）
# # ==============================
#
# if __name__ == "__main__":
#     ce30_model, ce30_acc, ce30_sens, ce30_spec = run_ce_baseline_training2(
#         num_epochs=NUM_EPOCHS, save_path=BEST_MODEL_PATH2
#     )
#
#     ce60_model, ce60_acc, ce60_sens, ce60_spec = run_ce_long_from_best_ce2(
#         init_ckpt_path=BEST_MODEL_PATH2,
#         num_epochs=30,
#         save_path=CE_LONG_BEST_PATH2,
#     )
#
#     sca_long_model, sca_long_acc, sca_long_sens, sca_long_spec = run_sca_training_from_best_ce2(
#         init_ckpt_path=BEST_MODEL_PATH2,
#         num_epochs=30,
#         save_path=SCA_LONG_BEST_PATH2,
#         lambda_sca_val=lambda_sca,
#     )
#
#     sca_all_model, sca_all_acc, sca_all_sens, sca_all_spec = run_sca_all_from_scratch2(
#         num_epochs=60,
#         save_path=SCA_ALL_BEST_PATH2,
#         lambda_sca_val=lambda_sca,
#     )
#
#     print("\n============== SUMMARY (CirCor herat2.py, subject-level 9:1) ==============")
#     print(f"CE-30   (baseline)                      best val acc: {ce30_acc:.4f}, "
#           f"sens: {ce30_sens:.4f}, spec: {ce30_spec:.4f}, ckpt: {BEST_MODEL_PATH2}")
#     print(f"CE-60   (CE-long, 30+30 CE)             best val acc: {ce60_acc:.4f}, "
#           f"sens: {ce60_sens:.4f}, spec: {ce60_spec:.4f}, ckpt: {CE_LONG_BEST_PATH2}")
#     print(f"SCA-long(CE-30 -> SCA-30, λ={lambda_sca:.2f})   best val acc: {sca_long_acc:.4f}, "
#           f"sens: {sca_long_sens:.4f}, spec: {sca_long_spec:.4f}, ckpt: {SCA_LONG_BEST_PATH2}")
#     print(f"SCA-all (SCA-60 from scratch, λ={lambda_sca:.2f})   best val acc: {sca_all_acc:.4f}, "
#           f"sens: {sca_all_sens:.4f}, spec: {sca_all_spec:.4f}, ckpt: {SCA_ALL_BEST_PATH2}")
#     print("===========================================================================")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
herat2.py — CirCor heart sound (PhysioNet Challenge 2022) 心音分类 + SCA 频率先验对齐

使用官方三分数据结构：
  circor-heart-sound/
    ├── training_data/
    ├── validation_data/
    ├── test_data/
    ├── training_data.csv
    ├── validation_data.csv
    ├── test_data.csv

任务：Outcome 二分类（Normal / Abnormal）= 0 / 1

训练实验：
  - CE-30   (baseline)
  - CE-60   (CE-30 -> 再训 30 epoch 纯 CE)
  - SCA-long (CE-30 -> SCA-30, λ = λ*)
  - SCA-all  (SCA-60 from scratch, λ ∈ {0.05,0.1,0.3,0.5} 网格搜索)

指标：
  - accuracy / sensitivity / specificity
  - 训练与 early stopping 全部基于 validation_data，
    最终四个模型在 test_data 上做一次测试评估。
"""

import os
import csv
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import shutil

# ============================================================
# 全局配置
# ============================================================

# ---- 根目录：----
CIRCOR_ROOT = "../heart_2022"

TRAIN_WAV_ROOT = os.path.join(CIRCOR_ROOT, "training_data")
VAL_WAV_ROOT   = os.path.join(CIRCOR_ROOT, "validation_data")
TEST_WAV_ROOT  = os.path.join(CIRCOR_ROOT, "test_data")

TRAIN_CSV_PATH = os.path.join(CIRCOR_ROOT, "training_data.csv")
VAL_CSV_PATH   = os.path.join(CIRCOR_ROOT, "validation_data.csv")
TEST_CSV_PATH  = os.path.join(CIRCOR_ROOT, "test_data.csv")

# ---- 心音采样率 ----
TARGET_SR = 2000   # 原始 4000 Hz，这里统一重采样到 2k（与 2016 保持一致）

# ---- Mel 频谱参数 ----
N_MELS = 128
WIN_LENGTH = int(0.05 * TARGET_SR)   # 50 ms
HOP_LENGTH = int(0.025 * TARGET_SR)  # 25 ms
N_FFT = 256                          # >= WIN_LENGTH, power of 2
F_MIN = 25.0
F_MAX = TARGET_SR / 2.0

# ---- 训练超参数 ----
BATCH_SIZE   = 16
NUM_WORKERS  = 0
NUM_EPOCHS_CE = 30      # CE-30 / SCA-long 的 epoch 数
NUM_EPOCHS_SCA_ALL = 60 # SCA-all 的 epoch 数
LEARNING_RATE = 1e-2
WEIGHT_DECAY  = 1e-4
SEED          = 2025

# ---- SCA λ 网格 ----
SCA_LAMBDA_GRID = [0.05, 0.1,0.2, 0.3, 0.5]

# ---- 权重文件名（全部带 2）----
BEST_MODEL_PATH2       = "pcg_resnet18_melspec_best2.pt"              # CE-30
CE_LONG_BEST_PATH2     = "pcg_resnet18_melspec_ce_long_best2.pt"      # CE-60
SCA_LONG_BEST_PATH2    = "pcg_resnet18_melspec_sca_long_best2.pt"     # SCA-long (λ = λ*)
SCA_ALL_BEST_PATH2     = "pcg_resnet18_melspec_sca_all_best2.pt"      # SCA-all (λ = λ*)

# ---- 随机种子 ----
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 解析 CSV：Patient ID + Outcome
# ============================================================

def load_subject_labels_from_csv(csv_path: str) -> Dict[str, int]:
    """
    从 *split*_data.csv 里读取每个 Patient ID 的 Outcome：
      - Outcome == "Normal"   -> 0
      - Outcome == "Abnormal" -> 1

    返回: {subject_id(str): label(int)}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    subj2label: Dict[str, int] = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        # 尝试兼容几种可能的列名
        possible_id_keys = ["Patient ID", "patient_id", "PatientID", "ID"]
        possible_outcome_keys = ["Outcome", "outcome"]

        for row in reader:
            # 找 Patient ID
            subj_id = None
            for k in possible_id_keys:
                if k in row and row[k].strip():
                    subj_id = row[k].strip()
                    break
            if subj_id is None:
                continue

            # 找 Outcome
            outcome_str = None
            for k in possible_outcome_keys:
                if k in row and row[k].strip():
                    outcome_str = row[k].strip()
                    break
            if outcome_str is None:
                continue

            if outcome_str == "Normal":
                label = 0
            elif outcome_str == "Abnormal":
                label = 1
            else:
                # 其他标签（如 Unknown）直接跳过
                continue

            if subj_id in subj2label and subj2label[subj_id] != label:
                print(f"[WARN] conflicting labels for subject {subj_id}: "
                      f"{subj2label[subj_id]} vs {label}")
            subj2label[subj_id] = label

    print(f"Loaded subject labels from {os.path.basename(csv_path)}: {len(subj2label)} subjects.")
    return subj2label


def build_circor_items_for_split(
    wav_root: str,
    csv_path: str,
    split_name: str,
) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """
    对于一个 split（train / val / test）：
      - 从 csv 里读 subject-level Outcome
      - 扫描对应 wav_root 下所有 .wav
      - 按 (wav_path, label, subject_id) 构建 item 列表

    返回:
      items: list[(wav_path, label, subject_id)]
      subject_ids: 去重后的 subject 列表（排序）
    """
    subj2label = load_subject_labels_from_csv(csv_path)

    if not os.path.isdir(wav_root):
        raise FileNotFoundError(f"Wave root dir not found: {wav_root}")

    wav_files = [f for f in os.listdir(wav_root) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"[WARN] No .wav files found under {wav_root}")

    items: List[Tuple[str, int, str]] = []
    used_subjects = set()

    for fname in wav_files:
        base = os.path.splitext(fname)[0]    # e.g., "2530_AV" or "50032_TV_2"
        subj_id = base.split("_")[0]         # "2530"

        if subj_id not in subj2label:
            # 这条记录在 csv 中没有 Outcome（极少），直接跳过
            continue

        label = subj2label[subj_id]
        wav_path = os.path.join(wav_root, fname)
        items.append((wav_path, label, subj_id))
        used_subjects.add(subj_id)

    subject_ids = sorted(list(used_subjects))
    print(f"[{split_name}] Total recordings: {len(items)}, "
          f"Total subjects with at least one recording: {len(subject_ids)}")
    return items, subject_ids


# ============================================================
# Dataset：raw waveform + log-Mel
# ============================================================

class CirCorPCGRaw(Dataset):
    """
    CirCor 数据集上，记录级的 Dataset。
    items: list of (wav_path, label, subject_id)
    每个样本: (waveform[T], label)
    """

    def __init__(self, items: List[Tuple[str, int, str]]):
        super().__init__()
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, label, subj_id = self.items[idx]
        waveform, sr = torchaudio.load(wav_path)  # [C, T]

        # 单通道
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        waveform = waveform.squeeze(0)  # [T]

        # 重采样到 TARGET_SR
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        # 幅度归一化
        max_abs = waveform.abs().max()
        if max_abs > 0:
            waveform = waveform / max_abs

        return waveform, label


class PCGMelDataset(Dataset):
    """
    waveform -> log-Mel 频谱图
    输出: (mel: [1, T_frames, N_MELS], label)
    """

    def __init__(self, base_dataset: CirCorPCGRaw,
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
        waveform, label = self.base[idx]
        waveform = waveform.unsqueeze(0)  # [1, T]

        melspec = self.melspec(waveform)       # [1, n_mels, time]
        melspec_db = self.amp_to_db(melspec)   # [1, n_mels, time]

        # 转成 [1, T_frames, N_MELS]
        melspec_db = melspec_db.squeeze(0).transpose(0, 1).unsqueeze(0)

        return melspec_db, label


def collate_mel_batch(batch):
    """
    输入: list[(mel[1, T_i, F], label)]
    输出:
      mel_padded: [B, 1, T_max, F]
      labels    : [B]
      lengths   : [B] (原始 T_i)
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


# ============================================================
# 模型：ResNet18
# ============================================================

class PCGResNet18(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        base.fc = nn.Linear(base.fc.in_features, n_classes)
        self.backbone = base

    def forward(self, x):
        # x: [B,1,T,F]
        return self.backbone(x)


# ============================================================
# 构建三个 split 的数据集 & DataLoader
# ============================================================

train_items, train_subjects = build_circor_items_for_split(
    wav_root=TRAIN_WAV_ROOT,
    csv_path=TRAIN_CSV_PATH,
    split_name="TRAIN",
)

val_items, val_subjects = build_circor_items_for_split(
    wav_root=VAL_WAV_ROOT,
    csv_path=VAL_CSV_PATH,
    split_name="VAL",
)

test_items, test_subjects = build_circor_items_for_split(
    wav_root=TEST_WAV_ROOT,
    csv_path=TEST_CSV_PATH,
    split_name="TEST",
)

if len(train_items) == 0 or len(val_items) == 0:
    raise RuntimeError("Train/Val items empty. Please check paths & csv parsing.")

train_raw = CirCorPCGRaw(train_items)
val_raw   = CirCorPCGRaw(val_items)
test_raw  = CirCorPCGRaw(test_items)

train_ds = PCGMelDataset(train_raw)
val_ds   = PCGMelDataset(val_raw)
test_ds  = PCGMelDataset(test_raw)

mel_example, label_example = train_ds[0]
print(f"Example mel shape: {mel_example.shape}, label={label_example}")

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

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_mel_batch,
    pin_memory=True,
)


# ============================================================
# 训练 & 评估工具（含 sens/spec）
# ============================================================

def compute_sens_spec(cm: np.ndarray):
    """
    cm: 2x2 confusion matrix, order [[TN, FP],[FN, TP]]
    """
    if cm.shape != (2, 2):
        return float("nan"), float("nan")
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return sens, spec


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for mel_batch, labels, lengths in loader:
        mel_batch = mel_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(mel_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel_batch.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device, split_name="VAL"):
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
            all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    sens, spec = compute_sens_spec(cm)
    report = classification_report(
        all_labels, all_preds,
        target_names=["Normal(0)", "Abnormal(1)"]
    )
    print(f"[{split_name}] loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, "
          f"sens={sens:.4f}, spec={spec:.4f}")
    print(f"[{split_name}] confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")
    # 不打印 report，避免太长，必要时你可以打印
    return epoch_loss, epoch_acc, cm, report, sens, spec


# ============================================================
# SCA：频率先验 + KL
# ============================================================

# 4 个 Mel band
N_BANDS = 4
band_edges = np.linspace(0, N_MELS, N_BANDS + 1, dtype=int)
band_slices = [(int(band_edges[i]), int(band_edges[i + 1])) for i in range(N_BANDS)]
print("Band slices (Mel index):", band_slices)

# 与 2016 相同的频率先验
PRIOR_NORMAL = [0.7, 0.2, 0.1, 0.0]
PRIOR_ABNORM = [0.15, 0.35, 0.5, 0.0]
print("Freq-informed band prior (normal):  ", PRIOR_NORMAL)
print("Freq-informed band prior (abnormal):", PRIOR_ABNORM)

def build_toy_prior(labels, device):
    """
    为了兼容旧脚本而保留的函数名。
    实际上已经不再是 toy，而是调用频率知情的 build_freq_prior。
    """
    return build_freq_prior(labels, device)


def compute_gradxinput_attr(mel_batch, logits, labels, create_graph: bool = True):
    """
    Grad×Input attribution: [B,1,T,F]
    """
    selected_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
    scalar = selected_logits.sum()
    grads = torch.autograd.grad(
        outputs=scalar,
        inputs=mel_batch,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    attr = grads * mel_batch
    return attr.abs()


def aggregate_attr_over_bands(attr):
    """
    attr: [B,1,T,F]  ->  P_attr: [B, N_BANDS]
    """
    attr_sum = attr.sum(dim=2)  # [B,1,F]
    band_masses = []
    for (start, end) in band_slices:
        band_mass = attr_sum[..., start:end].sum(dim=-1)  # [B,1]
        band_masses.append(band_mass)

    band_masses = torch.cat(band_masses, dim=-1)  # [B,N_BANDS]
    eps = 1e-8
    denom = band_masses.sum(dim=-1, keepdim=True) + eps
    P_attr = band_masses / denom
    return P_attr


def build_freq_prior(labels, device):
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


def kl_divergence(P_attr, P_prior):
    eps = 1e-8
    P_attr = P_attr.clamp(min=eps)
    P_prior = P_prior.clamp(min=eps)
    log_ratio = (P_attr.log() - P_prior.log())
    kl = (P_attr * log_ratio).sum(dim=-1)
    return kl.mean()


def train_one_epoch_sca(
    model,
    loader,
    optimizer,
    criterion,
    device,
    lambda_sca_val: float,
):
    """
    单 epoch：CE + λ * KL(P_attr || P_prior)
    """
    model.train()
    running_loss = 0.0
    running_ce   = 0.0
    running_sca  = 0.0
    all_preds = []
    all_labels = []

    for mel_batch, labels, lengths in loader:
        mel_batch = mel_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mel_batch.requires_grad_(True)
        optimizer.zero_grad()

        logits = model(mel_batch)
        ce_loss = criterion(logits, labels)

        attr   = compute_gradxinput_attr(mel_batch, logits, labels, create_graph=True)
        P_attr = aggregate_attr_over_bands(attr)
        P_prior = build_freq_prior(labels, device)
        sca_loss = kl_divergence(P_attr, P_prior)

        total_loss = ce_loss + lambda_sca_val * sca_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * mel_batch.size(0)
        running_ce   += ce_loss.item() * mel_batch.size(0)
        running_sca  += sca_loss.item() * mel_batch.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_ce   = running_ce   / len(loader.dataset)
    epoch_sca  = running_sca  / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_ce, epoch_sca, epoch_acc


# ============================================================
# 训练封装：CE-30 / CE-60 / SCA-long / SCA-all (grid λ)
# ============================================================

def run_ce_baseline_training2(num_epochs=NUM_EPOCHS_CE, save_path=BEST_MODEL_PATH2):
    print(f"\n[CE-30] Training CE baseline on device: {device}")
    model = PCGResNet18(n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_sens = 0.0
    best_spec = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [CE-30] Epoch {epoch}/{num_epochs} =====")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"[TRAIN] loss={train_loss:.4f}, acc={train_acc:.4f}")

        val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
            model, val_loader, criterion, device, split_name="VAL"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sens = sens
            best_spec = spec
            torch.save(model.state_dict(), save_path)
            print(f"*** New best CE-30 model saved to {save_path} "
                  f"(val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f})")

    print("[CE-30] Training finished.")
    print(f"[CE-30] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
    return model, best_val_acc, best_sens, best_spec


def run_ce_long_from_best_ce2(
    init_ckpt_path=BEST_MODEL_PATH2,
    num_epochs=NUM_EPOCHS_CE,
    save_path=CE_LONG_BEST_PATH2,
):
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
    best_sens = 0.0
    best_spec = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [CE-60] Epoch {epoch}/{num_epochs} =====")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"[TRAIN] loss={train_loss:.4f}, acc={train_acc:.4f}")

        val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
            model, val_loader, criterion, device, split_name="VAL"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sens = sens
            best_spec = spec
            torch.save(model.state_dict(), save_path)
            print(f"*** New best CE-60 model saved to {save_path} "
                  f"(val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f})")

    print("[CE-60] Training finished.")
    print(f"[CE-60] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
    return model, best_val_acc, best_sens, best_spec


def run_sca_training_from_best_ce2(
    init_ckpt_path=BEST_MODEL_PATH2,
    num_epochs=NUM_EPOCHS_CE,
    save_path=SCA_LONG_BEST_PATH2,
    lambda_sca_val: float = 0.3,
):
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

    best_val_acc = 0.0
    best_sens = 0.0
    best_spec = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [SCA-LONG] Epoch {epoch}/{num_epochs} (λ={lambda_sca_val:.2f}) =====")
        train_loss, train_ce, train_sca_val, train_acc = train_one_epoch_sca(
            model_sca, train_loader, optimizer, criterion, device,
            lambda_sca_val=lambda_sca_val,
        )
        print(
            f"[TRAIN] total={train_loss:.4f}, CE={train_ce:.4f}, "
            f"SCA={train_sca_val:.4f}, acc={train_acc:.4f}"
        )

        val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
            model_sca, val_loader, criterion, device, split_name="VAL"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sens = sens
            best_spec = spec
            torch.save(model_sca.state_dict(), save_path)
            print(f"*** New best SCA-long model saved to {save_path}, "
                  f"val_acc={best_val_acc:.4f}, sens={best_sens:.4f}, spec={best_spec:.4f}")

    print("[SCA-LONG] Training finished.")
    print(f"[SCA-LONG] Best val acc: {best_val_acc:.4f}, sens: {best_sens:.4f}, spec: {best_spec:.4f}")
    return model_sca, best_val_acc, best_sens, best_spec


def run_sca_all_grid_from_scratch2(
    num_epochs: int = NUM_EPOCHS_SCA_ALL,
    lambda_grid = None,
):
    """
    SCA-all: 从随机初始化开始，直接训练 num_epochs 个 epoch 的 SCA 模型，
    在 λ_grid 上做网格搜索，只保留全局最优 λ* 对应的模型到 SCA_ALL_BEST_PATH2。
    """
    if lambda_grid is None:
        lambda_grid = SCA_LAMBDA_GRID

    print("\n================ SCA-ALL GRID SEARCH (from scratch) ================")
    print(f"Lambda grid: {lambda_grid}")
    print(f"Global best checkpoint path: {SCA_ALL_BEST_PATH2}")
    print(f"Freq-informed band prior: normal={PRIOR_NORMAL}, abnormal={PRIOR_ABNORM}")

    criterion = nn.CrossEntropyLoss()

    per_lambda_results = {}  # λ -> (best_val_acc, best_sens, best_spec, best_epoch)

    global_best_acc = 0.0
    global_best_lambda = None
    global_best_sens = 0.0
    global_best_spec = 0.0

    for lambda_sca_val in lambda_grid:
        print("\n------------------------------------------------------")
        print(f"[SCA-ALL] Training from scratch with λ={lambda_sca_val:.2f}")
        print("------------------------------------------------------")

        # 为每个 λ 重新设定随机种子，保证对比公平
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model_sca = PCGResNet18(n_classes=2).to(device)
        optimizer = torch.optim.Adam(
            model_sca.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        best_val_acc_this_lambda = 0.0
        best_sens_this_lambda = 0.0
        best_spec_this_lambda = 0.0
        best_epoch_this_lambda = 0

        ckpt_path_lambda = f"pcg_resnet18_melspec_sca_all_lam{str(lambda_sca_val).replace('.', 'p')}_best2.pt"

        for epoch in range(1, num_epochs + 1):
            print(f"\n[SCA-ALL | λ={lambda_sca_val:.2f}] Epoch {epoch}/{num_epochs}")
            train_loss, train_ce, train_sca_val, train_acc = train_one_epoch_sca(
                model_sca, train_loader, optimizer, criterion, device,
                lambda_sca_val=lambda_sca_val,
            )
            print(
                f"[TRAIN] total={train_loss:.4f}, CE={train_ce:.4f}, "
                f"SCA={train_sca_val:.4f}, acc={train_acc:.4f}"
            )

            val_loss, val_acc, cm, report, sens, spec = eval_one_epoch(
                model_sca, val_loader, criterion, device, split_name="VAL"
            )

            # per-λ 最优
            if val_acc > best_val_acc_this_lambda:
                best_val_acc_this_lambda = val_acc
                best_sens_this_lambda = sens
                best_spec_this_lambda = spec
                best_epoch_this_lambda = epoch
                torch.save(model_sca.state_dict(), ckpt_path_lambda)
                print(
                    f"[SCA-ALL | λ={lambda_sca_val:.2f}] "
                    f"*** New best for this λ saved to {ckpt_path_lambda} "
                    f"(val_acc={best_val_acc_this_lambda:.4f}, sens={best_sens_this_lambda:.4f}, "
                    f"spec={best_spec_this_lambda:.4f}, epoch={best_epoch_this_lambda})"
                )

            # 全局最优（跨 λ）
            if val_acc > global_best_acc:
                global_best_acc = val_acc
                global_best_lambda = lambda_sca_val
                global_best_sens = sens
                global_best_spec = spec
                # 拷贝当前 λ 的 best ckpt 到全局路径
                if os.path.exists(ckpt_path_lambda):
                    shutil.copy(ckpt_path_lambda, SCA_ALL_BEST_PATH2)
                else:
                    torch.save(model_sca.state_dict(), SCA_ALL_BEST_PATH2)
                print(
                    f"  >>> New GLOBAL best SCA-all model (λ={global_best_lambda:.2f}) "
                    f"copied to {SCA_ALL_BEST_PATH2} "
                    f"(val_acc={global_best_acc:.4f}, sens={global_best_sens:.4f}, "
                    f"spec={global_best_spec:.4f}, epoch={epoch})"
                )

        per_lambda_results[lambda_sca_val] = (
            best_val_acc_this_lambda,
            best_sens_this_lambda,
            best_spec_this_lambda,
            best_epoch_this_lambda,
        )
        print(
            f"[SCA-ALL | λ={lambda_sca_val:.2f}] finished. "
            f"Best val acc={best_val_acc_this_lambda:.4f}, "
            f"sens={best_sens_this_lambda:.4f}, spec={best_spec_this_lambda:.4f}, "
            f"epoch={best_epoch_this_lambda}"
        )

    # 总结
    print("\n[SCA-ALL] Grid search finished.")
    print("Per-λ best results (SCA-all from scratch):")
    for lambda_sca_val in lambda_grid:
        acc, sens, spec, ep = per_lambda_results[lambda_sca_val]
        print(
            f"  λ={lambda_sca_val:.2f}: best val acc={acc:.4f}, "
            f"sens={sens:.4f}, spec={spec:.4f}, epoch={ep}"
        )

    print(
        f"\nGLOBAL best SCA-all val acc={global_best_acc:.4f}, "
        f"sens={global_best_sens:.4f}, spec={global_best_spec:.4f}, "
        f"λ*={global_best_lambda:.2f}"
    )
    print(f"SCA-all global best checkpoint copied to: {SCA_ALL_BEST_PATH2}")

    return global_best_lambda, global_best_acc, global_best_sens, global_best_spec, per_lambda_results


# ============================================================
# main：训练 + λ 搜索 + test 评估
# ============================================================

if __name__ == "__main__":
    # 1) CE-30 baseline
    ce30_model, ce30_acc, ce30_sens, ce30_spec = run_ce_baseline_training2(
        num_epochs=NUM_EPOCHS_CE,
        save_path=BEST_MODEL_PATH2,
    )

    # 2) CE-60: 在 CE-30 基础上再跑 30 epoch 纯 CE
    ce60_model, ce60_acc, ce60_sens, ce60_spec = run_ce_long_from_best_ce2(
        init_ckpt_path=BEST_MODEL_PATH2,
        num_epochs=NUM_EPOCHS_CE,
        save_path=CE_LONG_BEST_PATH2,
    )

    # 3) SCA-all: λ ∈ {0.05,0.1,0.3,0.5} 网格搜索（from scratch）
    best_lambda_sca_all, sca_all_val_acc, sca_all_val_sens, sca_all_val_spec, sca_all_per_lambda = (
        run_sca_all_grid_from_scratch2(
            num_epochs=NUM_EPOCHS_SCA_ALL,
            lambda_grid=SCA_LAMBDA_GRID,
        )
    )

    # 4) SCA-long: 从 CE-30 出发，用 λ* 训练 30 epoch
    sca_long_model, sca_long_acc, sca_long_sens, sca_long_spec = run_sca_training_from_best_ce2(
        init_ckpt_path=BEST_MODEL_PATH2,
        num_epochs=NUM_EPOCHS_CE,
        save_path=SCA_LONG_BEST_PATH2,
        lambda_sca_val=best_lambda_sca_all,
    )

    print("\n============== SUMMARY (CirCor herat2.py, official TRAIN/VAL split) ==============")
    print(f"CE-30   (baseline)                      best val acc: {ce30_acc:.4f}, "
          f"sens: {ce30_sens:.4f}, spec: {ce30_spec:.4f}, ckpt: {BEST_MODEL_PATH2}")
    print(f"CE-60   (CE-long, 30+30 CE)             best val acc: {ce60_acc:.4f}, "
          f"sens: {ce60_sens:.4f}, spec: {ce60_spec:.4f}, ckpt: {CE_LONG_BEST_PATH2}")
    print(f"SCA-long(CE-30 -> SCA-30, λ*={best_lambda_sca_all:.2f})   best val acc: {sca_long_acc:.4f}, "
          f"sens: {sca_long_sens:.4f}, spec: {sca_long_spec:.4f}, ckpt: {SCA_LONG_BEST_PATH2}")
    print(f"SCA-all (SCA-60 from scratch, λ*={best_lambda_sca_all:.2f})   best val acc: {sca_all_val_acc:.4f}, "
          f"sens: {sca_all_val_sens:.4f}, spec: {sca_all_val_spec:.4f}, ckpt: {SCA_ALL_BEST_PATH2}")
    print("===============================================================================")

    # 5) 最终在 TEST split 上评估四个 best 模型
    print("\n====================== TEST SET EVALUATION ======================")
    criterion = nn.CrossEntropyLoss()

    def eval_on_test(ckpt_path: str, tag: str):
        model = PCGResNet18(n_classes=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"\n[TEST] Evaluating {tag} from {ckpt_path}")
        test_loss, test_acc, cm, report, sens, spec = eval_one_epoch(
            model, test_loader, criterion, device, split_name="TEST"
        )
        print(f"[TEST {tag}] acc={test_acc:.4f}, sens={sens:.4f}, spec={spec:.4f}")
        return test_acc, sens, spec

    eval_on_test(BEST_MODEL_PATH2,    "CE-30")
    eval_on_test(CE_LONG_BEST_PATH2,  "CE-60")
    eval_on_test(SCA_LONG_BEST_PATH2, "SCA-long (λ*)")
    eval_on_test(SCA_ALL_BEST_PATH2,  "SCA-all (λ*)")
    print("================================================================")
