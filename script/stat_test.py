import os
import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, confusion_matrix

import herat as h16
import herat2 as h22


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, sens, spec


def bootstrap_ci(y_true, y_pred, n_iter=10000):  # 升级为 10,000 次
    """使用 Bootstrap 计算 95% 置信区间"""
    n = len(y_true)
    accs, senss, specs = [], [], []
    for _ in range(n_iter):
        indices = np.random.choice(n, n, replace=True)
        a, se, sp = calc_metrics(y_true[indices], y_pred[indices])
        accs.append(a)
        senss.append(se)
        specs.append(sp)

    return (
        (np.percentile(accs, 2.5), np.percentile(accs, 97.5)),
        (np.percentile(senss, 2.5), np.percentile(senss, 97.5)),
        (np.percentile(specs, 2.5), np.percentile(specs, 97.5))
    )


def mcnemar_test(y_true, y_pred1, y_pred2):
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    b = np.sum(correct1 & ~correct2)  # 1对2错
    c = np.sum(~correct1 & correct2)  # 1错2对
    if b + c == 0:
        return 1.0
    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = chi2.sf(statistic, 1)
    return p_value


def get_predictions(model_class, model_path, loader, device):
    model = model_class(n_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel_batch, labels, _ in loader:
            mel_batch = mel_batch.to(device)
            logits = model(mel_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(2025)  # 保证每次跑出来的 CI 一模一样

    datasets = [
        ("PhysioNet 2016", h16.PCGResNet18, h16.CE_LONG_BEST_PATH, h16.SCA_ALL_LONG_BEST_PATH, h16.val_loader),
        ("CirCor 2022", h22.PCGResNet18, h22.CE_LONG_BEST_PATH2, h22.SCA_ALL_BEST_PATH2, h22.test_loader)
    ]

    for name, m_class, p_ce, p_sca, loader in datasets:
        print(f"\n[{name}]")
        y_true, y_pred_ce = get_predictions(m_class, p_ce, loader, device)
        _, y_pred_sca = get_predictions(m_class, p_sca, loader, device)

        # CE
        a_ce, se_ce, sp_ce = calc_metrics(y_true, y_pred_ce)
        ci_a_ce, ci_se_ce, ci_sp_ce = bootstrap_ci(y_true, y_pred_ce, 10000)

        # SCA
        a_sa, se_sa, sp_sa = calc_metrics(y_true, y_pred_sca)
        ci_a_sa, ci_se_sa, ci_sp_sa = bootstrap_ci(y_true, y_pred_sca, 10000)

        pval = mcnemar_test(y_true, y_pred_ce, y_pred_sca)

        print(
            f"CE-60   | Acc: {a_ce:.4f} ({ci_a_ce[0]:.4f}-{ci_a_ce[1]:.4f}) | Sens: {se_ce:.4f} ({ci_se_ce[0]:.4f}-{ci_se_ce[1]:.4f}) | Spec: {sp_ce:.4f} ({ci_sp_ce[0]:.4f}-{ci_sp_ce[1]:.4f})")
        print(
            f"SCA-all | Acc: {a_sa:.4f} ({ci_a_sa[0]:.4f}-{ci_a_sa[1]:.4f}) | Sens: {se_sa:.4f} ({ci_se_sa[0]:.4f}-{ci_se_sa[1]:.4f}) | Spec: {sp_sa:.4f} ({ci_sp_sa[0]:.4f}-{ci_sp_sa[1]:.4f})")
        print(f"McNemar p-value: {pval:.4e} -> {'SIGNIFICANT (p<0.05)' if pval < 0.05 else 'NOT SIGNIFICANT'}")