"""Training (Adam + MSE, early stopping, L2 weight decay, gradient clipping),
random hyper-parameter search over the CV folds, and final multi-seed runs."""
from __future__ import annotations

import copy
import itertools
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .cells import build_model, n_params
from .cv import early_stopping_split, expanding_window_folds
from .data import Series, load_series
from .evaluate import metrics

torch.set_num_threads(1)          # tiny models: one thread is fastest and deterministic

MODELS = ["elman", "jordan", "mrn"]
DATASET_OVERRIDES = {"mackey_glass": {"max_epochs": 1000}}

# ------------------------- search space -----------------------------------
SEARCH_SPACE = {
    "hidden": [4, 8, 16, 32],
    "window": [6, 12, 24],
    "lr": [1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}
MRN_DECAYS = [(0.25, 0.5, 0.75), (0.0, 0.5, 0.9), (0.0, 0.25, 0.5, 0.75), (0.5,)]

TRAIN_SETTINGS = dict(max_epochs=300, patience=15, batch_size=64, clip=1.0, es_frac=0.15)
N_CONFIGS = 24
TUNE_SEEDS = [0, 1]  
N_FOLDS = 5
CV_GAP = 0
FINAL_SEEDS = list(range(10))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

"""
Random search: the same n base configurations are drawn for every model
of a dataset (common random numbers -> fairer comparison); the MRN
additionally draws its decay-rate set
"""
def sample_configs(dataset: str, n: int | None = None):

    n = n or N_CONFIGS
    rng = np.random.default_rng(sum(map(ord, dataset)))   # stable seed per dataset
    grid = list(itertools.product(*SEARCH_SPACE.values()))
    picks = rng.choice(len(grid), size=n, replace=False)
    base = [dict(zip(SEARCH_SPACE.keys(), grid[i])) for i in picks]
    dec = rng.integers(0, len(MRN_DECAYS), size=n)
    return base, [MRN_DECAYS[d] for d in dec]


def configs_for(dataset: str, kind: str):
    base, decs = sample_configs(dataset)
    out = []
    for b, d in zip(base, decs):
        c = dict(b)
        if kind == "mrn":
            c["decays"] = list(d)
        out.append(c)
    return out


# ------------------------- core training ----------------------------------
def train_model(kind: str, cfg: dict, ser: Series, fit_targets: np.ndarray,
                es_targets: np.ndarray, seed: int, settings: dict = TRAIN_SETTINGS):
    set_seed(seed)
    settings = {**settings, **DATASET_OVERRIDES.get(ser.name, {})}
    W = cfg["window"]
    scaler = ser.fit_scaler(fit_targets)
    Xtr, Ytr = map(torch.from_numpy, ser.windows(fit_targets, W, scaler))
    Xes, Yes = map(torch.from_numpy, ser.windows(es_targets, W, scaler))
    model = build_model(kind, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    mse = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    hist = {"train": [], "val": []}
    for epoch in range(settings["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g) # shuffles windows, not time order
        for i in range(0, n, settings["batch_size"]):
            b = perm[i: i + settings["batch_size"]]
            opt.zero_grad()
            loss = mse(model(Xtr[b]), Ytr[b]) # all-step one-step-ahead supervision
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["clip"])
            opt.step()
        model.eval()
        with torch.no_grad(): # monitor the actual forecast
            tr = mse(model(Xtr)[:, -1], Ytr[:, -1]).item()
            va = mse(model(Xes)[:, -1], Yes[:, -1]).item()
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best - 1e-7:
            best, best_state, best_epoch, wait = va, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return dict(model=model, scaler=scaler, history=hist, best_epoch=best_epoch,
                epochs_run=len(hist["train"]), n_params=n_params(model))


def predict(res: dict, ser: Series, targets: np.ndarray, W: int) -> np.ndarray:
    X, _ = ser.windows(targets, W, res["scaler"])
    with torch.no_grad():
        u_hat = res["model"](torch.from_numpy(X))[:, -1].numpy()
    return ser.to_original(u_hat.astype(float), targets, res["scaler"])


# ---------------------------- hyper-parameter search -------------------------------
def tune(dataset: str, kind: str, out_dir: str, log=print):
    path = os.path.join(out_dir, f"tuning_{dataset}_{kind}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    ser = load_series(dataset)
    folds = expanding_window_folds(ser.dev_targets, N_FOLDS, CV_GAP)
    records = []
    t0 = time.time()
    for ci, cfg in enumerate(configs_for(dataset, kind)):
        fold_rmse = []
        for k, (tr, va) in enumerate(folds):
            fit, es = early_stopping_split(tr, TRAIN_SETTINGS["es_frac"])
            seed_rmse = []
            for sd in TUNE_SEEDS:
                res = train_model(kind, cfg, ser, fit, es, seed=sd)
                pred = predict(res, ser, va, cfg["window"])
                seed_rmse.append(metrics(ser.y[va], pred)["rmse"])
            fold_rmse.append(float(np.mean(seed_rmse)))
        records.append(dict(config=cfg, fold_rmse=fold_rmse, mean_rmse=float(np.mean(fold_rmse)), std_rmse=float(np.std(fold_rmse))))
        log(f"  [{dataset}/{kind}] cfg {ci+1}/{N_CONFIGS} cv-RMSE={np.mean(fold_rmse):.4f} "
            f"({time.time()-t0:.0f}s)")
    best = min(records, key=lambda r: r["mean_rmse"])
    out = dict(dataset=dataset, model=kind, records=records, best=best, folds=[dict(train=[int(tr[0]), int(tr[-1])], val=[int(va[0]), int(va[-1])]) for tr, va in folds])
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


# ----------------------------- final multi-seed runs -----------------------------------

"""
Re-train the selected configuration on the whole development set (the last
15% of it used only for early stopping)
"""
def final_runs(dataset: str, kind: str, cfg: dict, out_dir: str, seeds=None, log=print):

    seeds = FINAL_SEEDS if seeds is None else seeds
    path = os.path.join(out_dir, f"final_{dataset}_{kind}.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    ser = load_series(dataset)
    fit, es = early_stopping_split(ser.dev_targets, TRAIN_SETTINGS["es_frac"])
    test = ser.test_targets
    preds, rows, hists = [], [], []
    for s in seeds:
        res = train_model(kind, cfg, ser, fit, es, seed=s)
        p = predict(res, ser, test, cfg["window"])
        preds.append(p)
        m = metrics(ser.y[test], p)
        rows.append([m["rmse"], m["mae"], m["smape"], res["best_epoch"], res["epochs_run"]])
        hists.append(res["history"])
    log(f"  [{dataset}/{kind}] test RMSE {np.mean([r[0] for r in rows]):.4f} "
        f"+- {np.std([r[0] for r in rows]):.4f}  (params={res['n_params']})")
    out = dict(preds=np.array(preds), metrics=np.array(rows), targets=test,
               y_true=ser.y[test], histories=np.array(hists, dtype=object),
               n_params=res["n_params"], config=json.dumps(cfg))
    np.savez(path, **out)
    return out
