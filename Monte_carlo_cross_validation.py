import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from data.config import *
from data.loader import load_data
from utils.dataset import ScenarioDataset, collate_fn
from utils.set_seed import set_seed
from model.architecture import SetRanker
from model.trainer import train

K_FOLDS   = 5
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(input_dim, scene_dim):
    return SetRanker(
        feat_dim=input_dim,
        scenario_dim=scene_dim,
        hidden_dims=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)


def preload_tensors(scenarios):
    for s in scenarios:
        s["features"] = torch.tensor(s["features"], dtype=torch.float32, device=device)
        s["prefs"]    = torch.tensor(
            [p if p is not None else float("nan") for p in s.get("prefs", [])],
            device=device
        )
        s["confs"]    = torch.tensor(
            [c if c is not None else float("nan") for c in s.get("confs", [])],
            device=device
        )
    return scenarios


if __name__ == "__main__":
    # Global seed for data splitting reproducibility only
    set_seed(RANDOM_STATE)

    # ── load and prepare data
    scenarios = load_data(FROZEN_PATH, LABELED_PATH)
    labeled   = [s for s in scenarios if any(p is not None for p in s.get("prefs", []))]
    labeled   = preload_tensors(labeled)

    # ── stratify by group (same logic as main.py)
    control_sc = [s for s in labeled if str(s["id"]).startswith("control")]
    expert_sc  = [s for s in labeled if str(s["id"]).startswith("expert")]
    llm_sc     = [s for s in labeled if not str(s["id"]).startswith(("control", "expert"))]


    # ── build stratified k-fold splits
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_indices = [{"train": [], "test": []} for _ in range(K_FOLDS)]
    for stratum in [control_sc, expert_sc, llm_sc]:
        if len(stratum) == 0:
            continue
        for fold_i, (tr_idx, te_idx) in enumerate(kf.split(stratum)):
            fold_indices[fold_i]["train"].extend([stratum[i] for i in tr_idx])
            fold_indices[fold_i]["test"].extend([stratum[i]  for i in te_idx])

    # ── compute input dims once
    first     = next(s for s in labeled if s["features"].numel() > 0)
    input_dim = int(first["features"].shape[1])
    scene_dim = len(STAKEHOLDER_PREFS) + len(SCENARIO_PREFS)

    # ── run folds
    fold_train_losses = []
    fold_test_losses  = []

    for fold_i, splits in enumerate(fold_indices):
        print(f"{'='*45}")
        print(f"  FOLD {fold_i+1}/{K_FOLDS}  |  "
              f"train={len(splits['train'])}  test={len(splits['test'])}")
        print(f"{'='*45}")

        set_seed(RANDOM_STATE + fold_i)

        random.shuffle(splits["train"])
        random.shuffle(splits["test"])

        train_ds = ScenarioDataset(splits["train"])
        test_ds  = ScenarioDataset(splits["test"])

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model     = build_model(input_dim, scene_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        train_losses, val_losses = train(
            model, train_loader, test_loader, optimizer,
            EPOCHS, scheduler, device="cuda",
            early_stopping_patience=5,
            early_stopping_min_delta=1e-6,
        )

        best_train = min(train_losses)
        best_val   = min(val_losses)
        fold_train_losses.append(best_train)
        fold_test_losses.append(best_val)

        print(f"  Fold {fold_i+1} best train loss : {best_train:.6f}")
        print(f"  Fold {fold_i+1} best test  loss : {best_val:.6f}\n")

    # ── aggregate
    train_arr = np.array(fold_train_losses)
    test_arr  = np.array(fold_test_losses)

    print(f"{'='*45}")
    print(f"  {K_FOLDS}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*45}")
    print(f"  Train loss : {train_arr.mean():.6f} ± {train_arr.std():.6f}")
    print(f"  Test  loss : {test_arr.mean():.6f}  ± {test_arr.std():.6f}")
    print(f"\n  Per-fold results:")
    for i, (tr, te) in enumerate(zip(fold_train_losses, fold_test_losses)):
        print(f"    Fold {i+1} | train={tr:.6f} | test={te:.6f}")