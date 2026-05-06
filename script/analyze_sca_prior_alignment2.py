"""
analyze_sca_prior_alignment2.py

在 CirCor 数据集上，比较 CE-60, SCA-long, SCA-all
在“频率启发的 Mel-band prior”下的解释对齐情况：

Step 0: 在同一个 validation 集合上收集三种模型的 band-level Grad×Input 分布
Step 1: normal / abnormal 的 mean P_attr & mean KL（freq-informed prior）
Step 2: ΔKL & cosine(P_attr, prior)
Step 3: uniform prior sanity check
Step 4: wrong prior sanity check
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

# 注意：这里专门用 CirCor 版 herat2
import herat2 as heart   # 确保文件名是 herat2.py


# ------------------------
# 小工具：sample-wise KL & cosine
# ------------------------

def samplewise_kl_torch(P_attr: torch.Tensor,
                        P_prior: torch.Tensor) -> torch.Tensor:
    """
    P_attr, P_prior: [B, N_BANDS]
    返回: [B] 每个样本的 KL(P_attr || P_prior)
    """
    eps = 1e-8
    P_attr = P_attr.clamp(min=eps)
    P_prior = P_prior.clamp(min=eps)
    log_ratio = P_attr.log() - P_prior.log()
    kl = (P_attr * log_ratio).sum(dim=-1)   # [B]
    return kl


def samplewise_kl_numpy(P_attr: np.ndarray,
                        P_prior: np.ndarray) -> np.ndarray:
    """
    Numpy 版本 KL(P_attr || P_prior)，方便后面 uniform / wrong prior 分析

    P_attr, P_prior: [N, N_BANDS]
    返回: [N] 每个样本的 KL
    """
    eps = 1e-8
    P_attr = np.clip(P_attr, eps, 1.0)
    P_prior = np.clip(P_prior, eps, 1.0)
    log_ratio = np.log(P_attr) - np.log(P_prior)
    kl = np.sum(P_attr * log_ratio, axis=-1)
    return kl


def cosine_np(P_attr: np.ndarray,
              P_prior: np.ndarray) -> np.ndarray:
    """
    逐样本 cosine similarity，Numpy 版本

    P_attr, P_prior: [N, N_BANDS]
    返回: [N]
    """
    eps = 1e-8
    num = np.sum(P_attr * P_prior, axis=-1)
    den = np.linalg.norm(P_attr, axis=-1) * np.linalg.norm(P_prior, axis=-1) + eps
    return num / den


# ------------------------
# Step 0: 收集 CE-60 / SCA-long / SCA-all 的 attribution 数据
# ------------------------

def collect_attr_data(batch_size: int = 4):
    """
    在 CirCor 的 validation 集合上，分别用 CE-60, SCA-long, SCA-all 模型
    计算 Grad×Input attribution，并按 mel-bands 聚合成 P_attr。

    返回 data 字典字段：
      - labels:      [N] int (0/1)
      - ce_P:        [N, N_BANDS]
      - sca_long_P:  [N, N_BANDS]
      - sca_all_P:   [N, N_BANDS]
      - prior:       [N, N_BANDS]  (freq-informed prior: normal / abnormal)
      - kl_ce:       [N]
      - kl_sca_long: [N]
      - kl_sca_all:  [N]
    """
    device = heart.device

    # 使用 herat2 里的 val_ds + collate_mel_batch 构建一个“小 batch”的 val_loader
    val_loader = DataLoader(
        heart.val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=heart.collate_mel_batch,
        pin_memory=True,
    )

    PCGResNet18 = heart.PCGResNet18
    compute_gradxinput_attr = heart.compute_gradxinput_attr
    aggregate_attr_over_bands = heart.aggregate_attr_over_bands
    # 现在 build_toy_prior 内部其实就是频率启发的 prior
    build_prior = getattr(heart, "build_toy_prior", heart.build_freq_prior)

    # 从 herat2 里读 checkpoint 路径，避免写死
    # 这里假定常量名与原 herat.py 保持一致，只是文件名后缀加了 2
    CE_LONG_BEST_PATH      = heart.CE_LONG_BEST_PATH2
    SCA_LONG_BEST_PATH     = heart.SCA_LONG_BEST_PATH2
    SCA_ALL_LONG_BEST_PATH = heart.SCA_ALL_BEST_PATH2

    # ---------- 1) 加载三个模型 ----------
    ce_model = PCGResNet18(n_classes=2).to(device)
    ce_model.load_state_dict(torch.load(CE_LONG_BEST_PATH, map_location=device))
    ce_model.eval()

    sca_long_model = PCGResNet18(n_classes=2).to(device)
    sca_long_model.load_state_dict(torch.load(SCA_LONG_BEST_PATH, map_location=device))
    sca_long_model.eval()

    sca_all_model = PCGResNet18(n_classes=2).to(device)
    sca_all_model.load_state_dict(torch.load(SCA_ALL_LONG_BEST_PATH, map_location=device))
    sca_all_model.eval()

    print(f"Loaded CE-60      from {CE_LONG_BEST_PATH}")
    print(f"Loaded SCA-long   from {SCA_LONG_BEST_PATH}")
    print(f"Loaded SCA-all    from {SCA_ALL_LONG_BEST_PATH}")

    # ---------- 2) 在 val 上遍历，收集 P_attr 和 KL ----------
    labels_list = []
    ce_P_list, sca_long_P_list, sca_all_P_list = [], [], []
    prior_list = []
    kl_ce_list, kl_sca_long_list, kl_sca_all_list = [], [], []

    for mel_batch, labels, lengths in val_loader:
        mel_batch = mel_batch.to(device)  # [B,1,T,F]
        labels = labels.to(device)

        # ---- CE-60：Grad×Input ----
        mel_ce = mel_batch.clone().detach().requires_grad_(True)
        logits_ce = ce_model(mel_ce)

        attr_ce = compute_gradxinput_attr(
            mel_ce, logits_ce, labels,
            create_graph=False    # 一阶导，不建高阶图，省显存
        )  # [B,1,T,F]

        P_ce = aggregate_attr_over_bands(attr_ce)   # [B, N_BANDS]
        prior = build_prior(labels, device)         # [B, N_BANDS]
        kl_ce = samplewise_kl_torch(P_ce, prior)    # [B]

        del mel_ce, logits_ce, attr_ce

        # ---- SCA-long：Grad×Input ----
        mel_sca_long = mel_batch.clone().detach().requires_grad_(True)
        logits_sca_long = sca_long_model(mel_sca_long)

        attr_sca_long = compute_gradxinput_attr(
            mel_sca_long, logits_sca_long, labels,
            create_graph=False
        )
        P_sca_long = aggregate_attr_over_bands(attr_sca_long)
        kl_sca_long = samplewise_kl_torch(P_sca_long, prior)

        del mel_sca_long, logits_sca_long, attr_sca_long

        # ---- SCA-all：Grad×Input ----
        mel_sca_all = mel_batch.clone().detach().requires_grad_(True)
        logits_sca_all = sca_all_model(mel_sca_all)

        attr_sca_all = compute_gradxinput_attr(
            mel_sca_all, logits_sca_all, labels,
            create_graph=False
        )
        P_sca_all = aggregate_attr_over_bands(attr_sca_all)
        kl_sca_all = samplewise_kl_torch(P_sca_all, prior)

        del mel_sca_all, logits_sca_all, attr_sca_all

        # ---- 收集到 CPU / numpy ----
        labels_list.append(labels.detach().cpu().numpy())
        ce_P_list.append(P_ce.detach().cpu().numpy())
        sca_long_P_list.append(P_sca_long.detach().cpu().numpy())
        sca_all_P_list.append(P_sca_all.detach().cpu().numpy())
        prior_list.append(prior.detach().cpu().numpy())
        kl_ce_list.append(kl_ce.detach().cpu().numpy())
        kl_sca_long_list.append(kl_sca_long.detach().cpu().numpy())
        kl_sca_all_list.append(kl_sca_all.detach().cpu().numpy())

        torch.cuda.empty_cache()

    labels_all = np.concatenate(labels_list, axis=0)
    ce_P_all = np.concatenate(ce_P_list, axis=0)
    sca_long_P_all = np.concatenate(sca_long_P_list, axis=0)
    sca_all_P_all = np.concatenate(sca_all_P_list, axis=0)
    prior_all = np.concatenate(prior_list, axis=0)
    kl_ce_all = np.concatenate(kl_ce_list, axis=0)
    kl_sca_long_all = np.concatenate(kl_sca_long_list, axis=0)
    kl_sca_all_all = np.concatenate(kl_sca_all_list, axis=0)

    print(f"\nCollected attribution data for {len(labels_all)} validation samples.")

    data = {
        "labels": labels_all,
        "ce_P": ce_P_all,
        "sca_long_P": sca_long_P_all,
        "sca_all_P": sca_all_P_all,
        "prior": prior_all,
        "kl_ce": kl_ce_all,
        "kl_sca_long": kl_sca_long_all,
        "kl_sca_all": kl_sca_all_all,
    }
    return data


# ------------------------
# Step 1: mean P_attr + mean KL（freq-informed prior）
# ------------------------

def summarize_prior_alignment(data):
    labels = data["labels"]
    ce_P = data["ce_P"]
    sca_long_P = data["sca_long_P"]
    sca_all_P = data["sca_all_P"]
    kl_ce = data["kl_ce"]
    kl_sca_long = data["kl_sca_long"]
    kl_sca_all = data["kl_sca_all"]

    normal_mask = (labels == 0)
    abnorm_mask = (labels == 1)

    def safe_mean(x, mask, axis=None):
        if mask.sum() == 0:
            return None
        return x[mask].mean(axis=axis)

    ce_normal_mean       = safe_mean(ce_P, normal_mask, axis=0)
    sca_long_normal_mean = safe_mean(sca_long_P, normal_mask, axis=0)
    sca_all_normal_mean  = safe_mean(sca_all_P, normal_mask, axis=0)

    ce_abnorm_mean       = safe_mean(ce_P, abnorm_mask, axis=0)
    sca_long_abnorm_mean = safe_mean(sca_long_P, abnorm_mask, axis=0)
    sca_all_abnorm_mean  = safe_mean(sca_all_P, abnorm_mask, axis=0)

    ce_normal_kl_mean       = safe_mean(kl_ce, normal_mask)
    sca_long_normal_kl_mean = safe_mean(kl_sca_long, normal_mask)
    sca_all_normal_kl_mean  = safe_mean(kl_sca_all, normal_mask)

    ce_abnorm_kl_mean       = safe_mean(kl_ce, abnorm_mask)
    sca_long_abnorm_kl_mean = safe_mean(kl_sca_long, abnorm_mask)
    sca_all_abnorm_kl_mean  = safe_mean(kl_sca_all, abnorm_mask)

    # 从 herat2 读 freq-informed prior（防止未来改权重）
    prior_normal = getattr(heart, "PRIOR_NORMAL", [0.7, 0.2, 0.1, 0.0])
    prior_abnorm = getattr(heart, "PRIOR_ABNORM", [0.15, 0.35, 0.5, 0.0])

    print("\n===== [Step 1] Mean P_attr over 4 Mel-bands =====")
    print("Bands order: [band1, band2, band3, band4]")
    print(f"Freq-informed prior (normal):   {prior_normal}")
    print(f"Freq-informed prior (abnormal): {prior_abnorm}\n")

    if ce_normal_mean is not None:
        print(
            "CE-60     normal   P_attr mean:",
            np.round(ce_normal_mean, 4),
            f"   mean KL≈{ce_normal_kl_mean:.4f}",
        )
    if sca_long_normal_mean is not None:
        print(
            "SCA-long  normal   P_attr mean:",
            np.round(sca_long_normal_mean, 4),
            f"   mean KL≈{sca_long_normal_kl_mean:.4f}",
        )
    if sca_all_normal_mean is not None:
        print(
            "SCA-all   normal   P_attr mean:",
            np.round(sca_all_normal_mean, 4),
            f"   mean KL≈{sca_all_normal_kl_mean:.4f}",
        )

    if ce_abnorm_mean is not None:
        print(
            "CE-60     abnormal P_attr mean:",
            np.round(ce_abnorm_mean, 4),
            f"   mean KL≈{ce_abnorm_kl_mean:.4f}",
        )
    if sca_long_abnorm_mean is not None:
        print(
            "SCA-long  abnormal P_attr mean:",
            np.round(sca_long_abnorm_mean, 4),
            f"   mean KL≈{sca_long_abnorm_kl_mean:.4f}",
        )
    if sca_all_abnorm_mean is not None:
        print(
            "SCA-all   abnormal P_attr mean:",
            np.round(sca_all_abnorm_mean, 4),
            f"   mean KL≈{sca_all_abnorm_kl_mean:.4f}",
        )


# ------------------------
# Step 2: ΔKL & cosine(P_attr, prior)
# ------------------------

def analyze_kl_delta_and_cosine(data):
    labels = data["labels"]
    ce_P = data["ce_P"]
    sca_long_P = data["sca_long_P"]
    sca_all_P = data["sca_all_P"]
    prior = data["prior"]
    kl_ce = data["kl_ce"]
    kl_sca_long = data["kl_sca_long"]
    kl_sca_all = data["kl_sca_all"]

    # ΔKL: 相对 CE-60 的 KL 改变量
    delta_long = kl_ce - kl_sca_long
    delta_all = kl_ce - kl_sca_all

    cos_ce = cosine_np(ce_P, prior)
    cos_sca_long = cosine_np(sca_long_P, prior)
    cos_sca_all = cosine_np(sca_all_P, prior)

    def summarize_per_class(mask, name):
        n = int(mask.sum())
        if n == 0:
            print(f"[{name}] no samples, skip.")
            return

        kl_ce_mean = kl_ce[mask].mean()
        kl_sca_long_mean = kl_sca_long[mask].mean()
        kl_sca_all_mean = kl_sca_all[mask].mean()

        d_long = delta_long[mask]
        d_all = delta_all[mask]

        d_long_mean = d_long.mean()
        d_long_std = d_long.std()
        frac_long_pos = (d_long > 0).mean()

        d_all_mean = d_all.mean()
        d_all_std = d_all.std()
        frac_all_pos = (d_all > 0).mean()

        cos_ce_mean = cos_ce[mask].mean()
        cos_sca_long_mean = cos_sca_long[mask].mean()
        cos_sca_all_mean = cos_sca_all[mask].mean()

        print(f"\n[{name}] samples: {n}")
        print(f"  KL(CE-60)     mean = {kl_ce_mean:.4f}")
        print(f"  KL(SCA-long)  mean = {kl_sca_long_mean:.4f}")
        print(f"  KL(SCA-all)   mean = {kl_sca_all_mean:.4f}")

        print(f"  ΔKL_long = KL(CE-60) - KL(SCA-long): mean={d_long_mean:.4f}, std={d_long_std:.4f}")
        print(f"      fraction with ΔKL_long > 0 (SCA-long closer to prior) = {frac_long_pos:.3f}")

        print(f"  ΔKL_all  = KL(CE-60) - KL(SCA-all):  mean={d_all_mean:.4f}, std={d_all_std:.4f}")
        print(f"      fraction with ΔKL_all  > 0 (SCA-all  closer to prior) = {frac_all_pos:.3f}")

        print("  Cosine similarity(P_attr, prior):")
        print(f"      CE-60     mean cos = {cos_ce_mean:.4f}")
        print(f"      SCA-long  mean cos = {cos_sca_long_mean:.4f}")
        print(f"      SCA-all   mean cos = {cos_sca_all_mean:.4f}")

    print("\n===== [Step 2] ΔKL & Cosine(P_attr, prior) =====")
    summarize_per_class(labels == 0, "normal")
    summarize_per_class(labels == 1, "abnormal")


# ------------------------
# Step 3: uniform prior sanity check
# ------------------------

def analyze_with_uniform_prior(data):
    labels = data["labels"]
    ce_P = data["ce_P"]
    sca_long_P = data["sca_long_P"]
    sca_all_P = data["sca_all_P"]

    N, N_BANDS = ce_P.shape
    uniform_prior = np.full((N, N_BANDS), 1.0 / N_BANDS, dtype=np.float64)

    kl_ce_uniform = samplewise_kl_numpy(ce_P, uniform_prior)
    kl_sca_long_uniform = samplewise_kl_numpy(sca_long_P, uniform_prior)
    kl_sca_all_uniform = samplewise_kl_numpy(sca_all_P, uniform_prior)

    norm_mask = (labels == 0)
    abn_mask = (labels == 1)

    def safe_mean(x, mask):
        if mask.sum() == 0:
            return None
        return float(x[mask].mean())

    ce_norm = safe_mean(kl_ce_uniform, norm_mask)
    sca_long_norm = safe_mean(kl_sca_long_uniform, norm_mask)
    sca_all_norm = safe_mean(kl_sca_all_uniform, norm_mask)

    ce_abn = safe_mean(kl_ce_uniform, abn_mask)
    sca_long_abn = safe_mean(kl_sca_long_uniform, abn_mask)
    sca_all_abn = safe_mean(kl_sca_all_uniform, abn_mask)

    print("\n===== [Step 3] Uniform prior sanity check =====")
    print(f"Uniform prior: [{1.0 / N_BANDS:.2f}] * {N_BANDS}")

    if ce_norm is not None:
        print(f"[normal]   KL_uniform(CE-60)     mean = {ce_norm:.4f}")
        print(f"[normal]   KL_uniform(SCA-long)  mean = {sca_long_norm:.4f}")
        print(f"[normal]   KL_uniform(SCA-all)   mean = {sca_all_norm:.4f}")
    if ce_abn is not None:
        print(f"[abnormal] KL_uniform(CE-60)     mean = {ce_abn:.4f}")
        print(f"[abnormal] KL_uniform(SCA-long)  mean = {sca_long_abn:.4f}")
        print(f"[abnormal] KL_uniform(SCA-all)   mean = {sca_all_abn:.4f}")

    print("(* 预期：在“无信息”先验下，CE / SCA-long / SCA-all 的 KL 差距会明显缩小，"
          "说明 SCA 的主要优势主要体现在与“有意义先验”的对齐上。)")


# ------------------------
# Step 4: “错误先验” sanity check
# ------------------------

def analyze_with_wrong_prior(data):
    """
    构造一个“刻意错误”的先验：
      - normal:   高频为主 [0.1, 0.3, 0.6, 0.0]
      - abnormal: 低频为主 [0.6, 0.3, 0.1, 0.0]
    然后看在这个错误先验下 CE-60 / SCA-long / SCA-all 的 KL 情况。
    """
    labels = data["labels"]
    ce_P = data["ce_P"]
    sca_long_P = data["sca_long_P"]
    sca_all_P = data["sca_all_P"]

    N, N_BANDS = ce_P.shape
    assert N_BANDS == 4, "当前 wrong-prior 假设写死为 4 bands，如果改 N_BANDS 记得同步这里。"

    wrong_prior = np.zeros((N, N_BANDS), dtype=np.float64)

    # 定义错误先验
    wrong_normal = np.array([0.1, 0.3, 0.6, 0.0], dtype=np.float64)
    wrong_abnorm = np.array([0.6, 0.3, 0.1, 0.0], dtype=np.float64)

    norm_mask = (labels == 0)
    abn_mask = (labels == 1)

    wrong_prior[norm_mask] = wrong_normal
    wrong_prior[abn_mask] = wrong_abnorm

    kl_ce_wrong = samplewise_kl_numpy(ce_P, wrong_prior)
    kl_sca_long_wrong = samplewise_kl_numpy(sca_long_P, wrong_prior)
    kl_sca_all_wrong = samplewise_kl_numpy(sca_all_P, wrong_prior)

    def safe_mean(x, mask):
        if mask.sum() == 0:
            return None
        return float(x[mask].mean())

    ce_norm = safe_mean(kl_ce_wrong, norm_mask)
    sca_long_norm = safe_mean(kl_sca_long_wrong, norm_mask)
    sca_all_norm = safe_mean(kl_sca_all_wrong, norm_mask)

    ce_abn = safe_mean(kl_ce_wrong, abn_mask)
    sca_long_abn = safe_mean(kl_sca_long_wrong, abn_mask)
    sca_all_abn = safe_mean(kl_sca_all_wrong, abn_mask)

    print("\n===== [Step 4] Wrong prior sanity check =====")
    print("Wrong prior (deliberately flipped):")
    print("  normal:   [0.1, 0.3, 0.6, 0.0]  (假装正常心音是高频主导)")
    print("  abnormal: [0.6, 0.3, 0.1, 0.0]  (假装杂音是低频主导)\n")

    if ce_norm is not None:
        print(f"[normal]   KL_wrong(CE-60)     mean = {ce_norm:.4f}")
        print(f"[normal]   KL_wrong(SCA-long)  mean = {sca_long_norm:.4f}")
        print(f"[normal]   KL_wrong(SCA-all)   mean = {sca_all_norm:.4f}")
    if ce_abn is not None:
        print(f"[abnormal] KL_wrong(CE-60)     mean = {ce_abn:.4f}")
        print(f"[abnormal] KL_wrong(SCA-long)  mean = {sca_long_abn:.4f}")
        print(f"[abnormal] KL_wrong(SCA-all)   mean = {sca_all_abn:.4f}")

    print("(* 如果 SCA 真的在利用“正确的临床/结构先验”，那么在这个故意错误的先验下，"
          "它的 KL 优势应该明显减弱甚至反转。)")


# ------------------------
# main
# ------------------------

if __name__ == "__main__":
    # Step 0: 在 CirCor 的 val 上跑一遍，收集 CE-60 / SCA-long / SCA-all attribution 数据
    data = collect_attr_data(batch_size=4)

    # Step 1: mean P_attr + mean KL（freq-informed prior）
    summarize_prior_alignment(data)

    # Step 2: ΔKL & cosine(P_attr, prior)
    analyze_kl_delta_and_cosine(data)

    # Step 3: uniform prior sanity check
    analyze_with_uniform_prior(data)

    # Step 4: “错误先验” sanity check
    analyze_with_wrong_prior(data)
