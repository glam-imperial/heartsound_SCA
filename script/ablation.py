#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_mask_and_l2.py

两个消融实验：
1) 频带硬 mask：只保留低频 (band1+2) / 高频 (band3+4)，评估 CE-60 / SCA-long / SCA-all。
2) 去掉 SCA，仅对 penultimate feature 做 L2 正则 (CE + λ‖h‖²)，从头训练 60 epoch
   并在 λ ∈ {0.05, 0.1, 0.2, 0.3, 0.5} 上做网格搜索。
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

import herat as heart  # 确保是 9:1 split 的版本


# ==============================
# 通用工具：metrics & 评估
# ==============================

def compute_metrics_from_cm(cm: np.ndarray):
    """
    cm: 2x2 confusion matrix with label order [0, 1]
    返回: (acc, sensitivity, specificity)
    """
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")

    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, sens, spec


def eval_model(
    model: nn.Module,
    loader,
    device: torch.device,
    mask_mode: str | None = None,
    desc: str = "",
):
    """
    通用评估函数，可选对输入 Mel 做频带硬 mask。

    mask_mode:
      - None     : 不做 mask，原始输入
      - "low"    : 仅保留低频 band1+2
      - "high"   : 仅保留高频 band3+4
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for mel_batch, labels, lengths in loader:
            mel_batch = mel_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mask_mode is not None:
                mel_batch = apply_freq_mask(mel_batch, mode=mask_mode)

            logits = model(mel_batch)
            loss = criterion(logits, labels)

            running_loss += loss.item() * mel_batch.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    avg_loss = running_loss / len(loader.dataset)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    acc, sens, spec = compute_metrics_from_cm(cm)

    if desc:
        print(f"{desc}")
    print(
        f"  loss={avg_loss:.4f}, acc={acc:.4f}, sens={sens:.4f}, spec={spec:.4f}"
    )
    print("  Confusion matrix [[TN, FP],[FN, TP]]:\n", cm)

    return avg_loss, acc, sens, spec, cm


# ==============================
# 消融 1: 低频 / 高频硬 mask
# ==============================

# 直接复用 herat.py 里的 band 切分
band_slices = heart.band_slices  # [(0,32),(32,64),(64,96),(96,128)]
# 低频：band1+2 => [0, 64)
LOW_START = band_slices[0][0]
LOW_END = band_slices[1][1]
# 高频：band3+4 => [64, 128)
HIGH_START = band_slices[2][0]
HIGH_END = band_slices[3][1]


def apply_freq_mask(mel_batch: torch.Tensor, mode: str):
    """
    mel_batch: [B, 1, T, F]
    mode: "low" or "high"
    返回: masked_mel_batch (新的 Tensor，不 in-place)
    """
    assert mode in ("low", "high")
    mask = torch.zeros_like(mel_batch)

    if mode == "low":
        mask[..., LOW_START:LOW_END] = 1.0
    else:  # "high"
        mask[..., HIGH_START:HIGH_END] = 1.0

    return mel_batch * mask


def run_mask_ablation():
    device = heart.device
    val_loader = heart.val_loader

    print("\n================ MASK ABLATION (low / high bands) ================")
    print("Band slices (Mel index):", band_slices)
    print(f"Low-band  mask keeps Mel[{LOW_START}:{LOW_END})")
    print(f"High-band mask keeps Mel[{HIGH_START}:{HIGH_END})")

    # 加载三种模型
    PCGResNet18 = heart.PCGResNet18

    ce_model = PCGResNet18(n_classes=2).to(device)
    ce_model.load_state_dict(torch.load(heart.CE_LONG_BEST_PATH, map_location=device))

    sca_long_model = PCGResNet18(n_classes=2).to(device)
    sca_long_model.load_state_dict(torch.load(heart.SCA_LONG_BEST_PATH, map_location=device))

    sca_all_model = PCGResNet18(n_classes=2).to(device)
    sca_all_model.load_state_dict(torch.load(heart.SCA_ALL_LONG_BEST_PATH, map_location=device))

    # 1) 无 mask baseline
    print("\n--- No mask (full Mel) ---")
    eval_model(ce_model, val_loader, device, None,  desc="CE-60   (no mask)")
    eval_model(sca_long_model, val_loader, device, None, desc="SCA-long(no mask)")
    eval_model(sca_all_model, val_loader, device, None, desc="SCA-all (no mask)")

    # 2) 低频 only
    print("\n--- Low-band only (bands 1+2) ---")
    eval_model(ce_model, val_loader, device, "low",  desc="CE-60   (low-band only)")
    eval_model(sca_long_model, val_loader, device, "low", desc="SCA-long(low-band only)")
    eval_model(sca_all_model, val_loader, device, "low", desc="SCA-all (low-band only)")

    # 3) 高频 only
    print("\n--- High-band only (bands 3+4) ---")
    eval_model(ce_model, val_loader, device, "high",  desc="CE-60   (high-band only)")
    eval_model(sca_long_model, val_loader, device, "high", desc="SCA-long(high-band only)")
    eval_model(sca_all_model, val_loader, device, "high", desc="SCA-all (high-band only)")

    print("==================================================================")


# ==============================
# 消融 2: Feature L2 正则 (替换 SCA)
# ==============================

L2_ALL_BEST_PATH = "pcg_resnet18_melspec_l2feat_all_best.pt"
L2_LAMBDA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]


def forward_with_feature(model: nn.Module, x: torch.Tensor):
    """
    手动展开 ResNet18，拿到 penultimate feature (avgpool后 flatten) 和 logits.

    返回:
      logits: [B, 2]
      feat  : [B, D]  (通常 D=512)
    """
    b = model.backbone
    x = b.conv1(x)
    x = b.bn1(x)
    x = b.relu(x)
    x = b.maxpool(x)

    x = b.layer1(x)
    x = b.layer2(x)
    x = b.layer3(x)
    x = b.layer4(x)

    x = b.avgpool(x)
    feat = torch.flatten(x, 1)
    logits = b.fc(feat)
    return logits, feat


def train_one_epoch_l2feat(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_l2: float,
):
    """
    单 epoch 训练：CE + λ‖h‖²，其中 h 是 penultimate feature。

    这里的 ‖h‖² 用 feat.pow(2).mean()，保证量纲在 O(1) 附近，便于和 λ 直接对比。
    """
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_l2 = 0.0
    all_labels = []
    all_preds = []

    for mel_batch, labels, lengths in loader:
        mel_batch = mel_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, feat = forward_with_feature(model, mel_batch)
        ce_loss = criterion(logits, labels)
        l2_loss = feat.pow(2).mean()  # feature L2

        total_loss = ce_loss + lambda_l2 * l2_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * mel_batch.size(0)
        running_ce += ce_loss.item() * mel_batch.size(0)
        running_l2 += l2_loss.item() * mel_batch.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_ce = running_ce / len(loader.dataset)
    epoch_l2 = running_l2 / len(loader.dataset)
    epoch_acc = (all_labels == all_preds).mean()

    return epoch_loss, epoch_ce, epoch_l2, epoch_acc


def run_feature_l2_all(
    num_epochs: int = 60,
    lambda_grid = None,
):
    """
    从随机初始化开始，用 CE + λ‖h‖² 训练 60 epoch，
    在多个 λ 上做 grid search，并记录全局 best (按 val acc)。
    """
    if lambda_grid is None:
        lambda_grid = L2_LAMBDA_GRID

    device = heart.device
    train_loader = heart.train_loader
    val_loader = heart.val_loader
    PCGResNet18 = heart.PCGResNet18

    lr = heart.LEARNING_RATE
    wd = heart.WEIGHT_DECAY

    # 为了可复现，沿用 herat 的 SEED
    import random
    random.seed(heart.SEED)
    np.random.seed(heart.SEED)
    torch.manual_seed(heart.SEED)
    torch.cuda.manual_seed_all(heart.SEED)

    print("\n================ FEATURE L2 ABLATION (CE + λ‖h‖²) ================")
    print(f"Using num_epochs={num_epochs}")
    print(f"Lambda grid: {lambda_grid}")
    print(f"Checkpoint for global best L2-all: {L2_ALL_BEST_PATH}")

    # 记录 per-λ 最优结果 & 全局最优
    per_lambda_results = {}  # λ -> (best_acc, best_sens, best_spec, best_epoch)

    best_val_acc_overall = 0.0
    best_lambda = None
    best_epoch_overall = 0
    best_sens_overall = 0.0
    best_spec_overall = 0.0

    for lambda_l2 in lambda_grid:
        print("\n------------------------------------------------------")
        print(f"[L2-ALL] Start training from scratch with λ={lambda_l2:.2f}")
        print("------------------------------------------------------")

        model = PCGResNet18(n_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        best_val_acc_this_lambda = 0.0
        best_sens_this_lambda = 0.0
        best_spec_this_lambda = 0.0
        best_epoch_this_lambda = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n[L2-ALL | λ={lambda_l2:.2f}] Epoch {epoch}/{num_epochs}")

            train_loss, train_ce, train_l2, train_acc = train_one_epoch_l2feat(
                model, train_loader, optimizer, criterion, device, lambda_l2=lambda_l2
            )
            print(
                f"  Train: total={train_loss:.4f}, CE={train_ce:.4f}, "
                f"L2={train_l2:.4f}, acc={train_acc:.4f}"
            )

            val_loss, val_acc, val_sens, val_spec, cm = eval_model(
                model,
                val_loader,
                device,
                mask_mode=None,
                desc=f"  [Val | λ={lambda_l2:.2f}]",
            )

            # 记录该 λ 下的最优 epoch（按 acc）
            if val_acc > best_val_acc_this_lambda:
                best_val_acc_this_lambda = val_acc
                best_sens_this_lambda = val_sens
                best_spec_this_lambda = val_spec
                best_epoch_this_lambda = epoch

            # 全局最优（跨 λ 和 epoch）
            if val_acc > best_val_acc_overall:
                best_val_acc_overall = val_acc
                best_lambda = lambda_l2
                best_epoch_overall = epoch
                best_sens_overall = val_sens
                best_spec_overall = val_spec

                torch.save(model.state_dict(), L2_ALL_BEST_PATH)
                print(
                    f"  *** New GLOBAL best L2-all model saved to {L2_ALL_BEST_PATH} "
                    f"(val_acc={best_val_acc_overall:.4f}, sens={best_sens_overall:.4f}, "
                    f"spec={best_spec_overall:.4f}, λ={best_lambda:.2f}, epoch={best_epoch_overall})"
                )

        per_lambda_results[lambda_l2] = (
            best_val_acc_this_lambda,
            best_sens_this_lambda,
            best_spec_this_lambda,
            best_epoch_this_lambda,
        )
        print(
            f"[L2-ALL | λ={lambda_l2:.2f}] finished. "
            f"Best val acc={best_val_acc_this_lambda:.4f}, sens={best_sens_this_lambda:.4f}, "
            f"spec={best_spec_this_lambda:.4f} at epoch {best_epoch_this_lambda}"
        )

    # 总结
    print("\n[L2-ALL] Grid search finished.")
    print("Per-λ best results (L2-all from scratch):")
    for lambda_l2 in lambda_grid:
        acc, sens, spec, ep = per_lambda_results[lambda_l2]
        print(
            f"  λ={lambda_l2:.2f}: best val acc={acc:.4f}, "
            f"sens={sens:.4f}, spec={spec:.4f}, epoch={ep}"
        )

    print(
        f"\nGLOBAL best L2-all val acc={best_val_acc_overall:.4f}, "
        f"sens={best_sens_overall:.4f}, spec={best_spec_overall:.4f}, "
        f"λ={best_lambda:.2f}, epoch={best_epoch_overall}"
    )
    print(f"L2-all checkpoint: {L2_ALL_BEST_PATH}")
    print("=================================================================")


# ==============================
# main
# ==============================

if __name__ == "__main__":
    # 1) Mask ablation: 低频 / 高频
    run_mask_ablation()

    # 2) Feature L2 ablation: CE + λ‖h‖² from scratch, 60 epochs
    run_feature_l2_all(num_epochs=60, lambda_grid=L2_LAMBDA_GRID)
