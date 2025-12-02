
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import kendalltau
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
import matplotlib.pyplot as plt
import matplotlib.cm as cm

times_new_roman = {'fontname':'Times New Roman', 'fontsize':9}


def _device_of_model(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _positions_from_scores(arr):
    order = np.argsort(-arr)        
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order))
    return pos


def _unpack_batch(batch):
    """
    Accepts a batch that may be (X, mask, prefs) or (X, mask, prefs, confs).
    Returns (X, mask, prefs, confs_or_None)
    """
    if len(batch) == 3:
        Xb, maskb, prefsb = batch
        confs = None
        groups = None
    elif len(batch) == 4:
        Xb, maskb, prefsb, confs = batch
        groups = None
    elif len(batch) == 5:
        Xb, maskb, prefsb, confs, groups = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return Xb, maskb, prefsb, confs, groups


import torch
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_kendall(model, loader):
    """
    Mean Kendall Tau (positions derived from preference scores).
    Skips cases with <2 valid items or undefined Tau.
    """
    model.eval()
    device = _device_of_model(model)
    taus = []

    for batch in loader:
        Xb, maskb, prefsb, _, _ = _unpack_batch(batch)
        Xb = Xb.to(device)
        maskb = maskb.to(device)
        prefsb = prefsb.to(device)

        with torch.no_grad():
            scores = model(Xb, maskb)


        scores_cpu = scores.cpu().numpy()
        prefsb_cpu = prefsb.cpu().numpy()
        maskb_cpu = maskb.cpu().numpy().astype(bool)

        B = scores_cpu.shape[0]

        for i in range(B):
            valid_mask = maskb_cpu[i] & (~np.isnan(prefsb_cpu[i]))
            if valid_mask.sum() < 2:
                continue

            pred_scores = scores_cpu[i][valid_mask]
            true_scores = prefsb_cpu[i][valid_mask]

            if np.allclose(true_scores, true_scores[0]) or np.allclose(pred_scores, pred_scores[0]):
                continue

            pos_true = _positions_from_scores(true_scores)
            pos_pred = _positions_from_scores(pred_scores)

            tau, _ = kendalltau(pos_true, pos_pred)
            if not np.isnan(tau):
                taus.append(tau)

    return float(np.mean(taus)) if len(taus) > 0 else float('nan')


def evaluate_regression(model, loader):
    """
    MAE and MSE between predicted scores and true preference scores (masked).
    """
    model.eval()
    device = _device_of_model(model)

    y_true_all = []
    y_pred_all = []

    for batch in loader:
        Xb, maskb, prefsb, _, _ = _unpack_batch(batch)
        Xb = Xb.to(device)
        maskb = maskb.to(device)
        prefsb = prefsb.to(device)

        with torch.no_grad():
            scores = model(Xb, maskb)

        scores_cpu = scores.cpu().numpy()
        prefsb_cpu = prefsb.cpu().numpy()
        maskb_cpu = maskb.cpu().numpy().astype(bool)

        B = scores_cpu.shape[0]

        for i in range(B):
            valid_mask = maskb_cpu[i] & (~np.isnan(prefsb_cpu[i]))
            if valid_mask.sum() == 0:
                continue

            pred = scores_cpu[i][valid_mask]
            true = prefsb_cpu[i][valid_mask]

            y_pred_all.extend(pred.tolist())
            y_true_all.extend(true.tolist())

    if len(y_true_all) == 0:
        return float('nan'), float('nan')

    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    return float(mae), float(mse)


def evaluate_retrieval(model, loader):
    """
    MRR and nDCG (binary relevance: top true-pref alternatives are relevant).
    """
    model.eval()
    device = _device_of_model(model)

    mrr_metric = RetrievalMRR()
    ndcg_metric = RetrievalNormalizedDCG()

    for batch in loader:
        Xb, maskb, prefsb, _, _ = _unpack_batch(batch)
        Xb = Xb.to(device)
        maskb = maskb.to(device)
        prefsb = prefsb.to(device)

        with torch.no_grad():
            scores = model(Xb, maskb)

        scores_cpu = scores.cpu()
        prefsb_cpu = prefsb.cpu()
        maskb_cpu = maskb.cpu().bool()

        B = scores_cpu.shape[0]

        for i in range(B):
            valid_mask = maskb_cpu[i] & (~torch.isnan(prefsb_cpu[i]))
            if valid_mask.sum() == 0:
                continue

            preds = scores_cpu[i][valid_mask]
            true_prefs = prefsb_cpu[i][valid_mask]

            max_pref = torch.max(true_prefs)
            target = (true_prefs == max_pref).long()

            idxs = torch.full_like(preds, i, dtype=torch.long)

            mrr_metric(preds, target, indexes=idxs)
            ndcg_metric(preds, target, indexes=idxs)

    try:
        mrr_val = mrr_metric.compute().item()
    except Exception:
        mrr_val = float('nan')
    try:
        ndcg_val = ndcg_metric.compute().item()
    except Exception:
        ndcg_val = float('nan')

    return float(mrr_val), float(ndcg_val)


def _device_of_model(model):
    return next(model.parameters()).device

def _unpack_batch(batch):
    return batch  #  batch is already unpacked as (X, mask, prefs, confs, groups)

def _positions_from_scores_tensor(scores):
    """
    Given a 1D tensor of scores, return the ranking positions (0-based)
    ranking higher scores as better (rank 0 is highest score).
    """
    _, sorted_indices = torch.sort(scores, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(scores), device=scores.device)
    return ranks

def plot_rank_differences(model, loader):
    """
    GPU-optimized histogram of (true_rank_pos - pred_rank_pos) over all valid items.
    """
    model.eval()
    device = _device_of_model(model)
    diffs_all = []

    with torch.no_grad():
        for batch in loader:
            Xb, maskb, prefsb, _, _ = _unpack_batch(batch)
            Xb = Xb.to(device)
            maskb = maskb.to(device)
            prefsb = prefsb.to(device)

            scores = model(Xb, maskb)  # [B, max_len]

            B = scores.size(0)

            for i in range(B):
                valid_mask = maskb[i] & (~torch.isnan(prefsb[i]))
                if valid_mask.sum() < 2:
                    continue

                true_scores = prefsb[i][valid_mask]
                pred_scores = scores[i][valid_mask]

                pos_true = _positions_from_scores_tensor(true_scores)
                pos_pred = _positions_from_scores_tensor(pred_scores)

                diffs = (pos_true - pos_pred).cpu()  
                diffs_all.append(diffs)

    if len(diffs_all) == 0:
        print("No rank differences to plot (no valid items).")
        return

    diffs_cat = torch.cat(diffs_all).numpy()

    plt.hist(diffs_cat, bins=50)
    plt.xlabel("True rank position - predicted rank position")
    plt.ylabel("Frequency")
    plt.show()


times_new_roman = {'fontname': 'Times New Roman', 'fontsize': 9}

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(4,3))
    cmap = cm.get_cmap("Blues", 3)
    plt.plot(train_losses, label='Training loss', color=cmap(2))
    plt.plot(val_losses, label='Validation loss', color=cmap(1))
    plt.xlabel('Epochs', **times_new_roman)
    plt.ylabel('Loss', **times_new_roman)
    plt.legend(prop={'family': 'Times New Roman', 'size': 9}, frameon=False)
    plt.xticks(fontname='Times New Roman', fontsize=9)
    plt.yticks(fontname='Times New Roman', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_combined(diffs_cat, train_losses, val_losses):
    fig, axs = plt.subplots(1, 2, figsize=(8,3))

    # Left subplot: histogram from diffs_cat
    axs[0].hist(diffs_cat, bins=50, color='navy', edgecolor='black')
    axs[0].set_xlabel("True rank position - predicted rank position", **times_new_roman)
    axs[0].set_ylabel("Frequency", **times_new_roman)
    axs[0].tick_params(axis='both', labelsize=9)
    for lbl in axs[0].get_xticklabels() + axs[0].get_yticklabels():
        lbl.set_fontname('Times New Roman')

    # Right subplot: loss curves 
    cmap = cm.get_cmap("Blues", 3)
    axs[1].plot(train_losses, label='Training Loss', color=cmap(2))
    axs[1].plot(val_losses, label='Validation Loss', color=cmap(1))
    axs[1].set_xlabel('Epochs', **times_new_roman)
    axs[1].set_ylabel('Loss', **times_new_roman)
    axs[1].legend(fontsize=9, frameon=False)
    axs[1].tick_params(axis='both', labelsize=9)
    for lbl in axs[1].get_xticklabels() + axs[1].get_yticklabels():
        lbl.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.show()






def stratified_evaluation(model, test_loader, batch_size=None):
    """ stratified evaluation using only test set"""
    if not hasattr(test_loader.dataset, 'scenarios'):
        print("Warning: No scenarios attribute found in test dataset")
        return

    batch_size = batch_size or test_loader.batch_size

    scenarios = test_loader.dataset.scenarios
    alt_counts = [len(s['features']) for s in scenarios]
    unique_counts = sorted(set(alt_counts))

    print("\nStratified Evaluation by Number of Alternatives (Test Set Only):")
    print(f"Total test scenarios: {len(scenarios)}")

    results = []
    for count in unique_counts:
        indices = [i for i, x in enumerate(alt_counts) if x == count]
        subset = Subset(test_loader.dataset, indices)
        subset_loader = DataLoader(subset,
                                  batch_size=batch_size,
                                  collate_fn=test_loader.collate_fn,
                                  shuffle=False)

        kendall = evaluate_kendall(model, subset_loader)
        mrr, ndcg = evaluate_retrieval(model, subset_loader)
        mae, mse = evaluate_regression(model, subset_loader)

        results.append({
            'alternatives': count,
            'samples': len(indices),
            'kendall': kendall,
            'mrr': mrr,
            'ndcg': ndcg,
            'mae': mae,
            'mse': mse
        })

    print("\n| Alternatives | Samples | Kendall | MRR   | nDCG  | MAE   | MSE   |")
    print("|--------------|---------|---------|-------|-------|-------|-------|")
    for r in results:
        kend = r['kendall']
        if kend is None or (isinstance(kend, float) and np.isnan(kend)):
            kend = float('nan')
        print(f"| {r['alternatives']:12} | {r['samples']:7} | {kend:.3f} | "
              f"{r['mrr']:.3f} | {r['ndcg']:.3f} | {r['mae']:.3f} | {r['mse']:.3f} |")

    return results
