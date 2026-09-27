
"""
Reproduce everything:  python run_all.py

---- Stages ----
(1) data summaries + stationarity tests
(2) cell unit tests
(3) random hyper-parameter search with 5-fold expanding-window CV
(4) final training of the selected configurations over 10 seeds
(5) metrics + Friedman/Nemenyi/Wilcoxon tests
(6) figures

"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import figures, tables, train
from src.data import DATASET_ORDER, load_series, stationarity_table, summary_table
from src.evaluate import friedman_nemenyi, metrics, persistence, wilcoxon_holm

MODELS = train.MODELS


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    args = ap.parse_args()

    res_dir = os.path.join(ROOT, "results_quick" if args.quick else "results")
    fig_dir = os.path.join(ROOT, "figures_quick" if args.quick else "figures")
    gen_dir = os.path.join(ROOT, "tables", "generated_quick" if args.quick else "generated")
    if args.force and os.path.isdir(res_dir):
        shutil.rmtree(res_dir)
    for d in (res_dir, fig_dir, gen_dir):
        os.makedirs(d, exist_ok=True)
    if args.quick:
        train.N_CONFIGS = 2
        train.FINAL_SEEDS = [0, 1, 2]
        train.TRAIN_SETTINGS.update(max_epochs=15, patience=5)

    t0 = time.time()
    # ---------------- 1 data + stationarity --------------------------------
    log("== Stage 1: datasets and stationarity tests")
    summ, st = summary_table(), stationarity_table()
    summ.to_csv(os.path.join(res_dir, "datasets.csv"), index=False)
    st.to_csv(os.path.join(res_dir, "stationarity.csv"), index=False)
    log(st[["dataset", "series", "adf_stat", "adf_p", "kpss_stat", "kpss_p", "verdict"]].round(4).to_string())

    # ---------------- 2 unit tests --------------------------------------------
    log("== Stage 2: unit tests of the recurrent cells")
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_cells")
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        sys.exit("Unit tests failed")

    # ---------------- 3 tuning -----------------------------------------------
    log("== Stage 3: hyper-parameter search (5-fold expanding-window CV)")
    tuning = {n: {k: train.tune(n, k, res_dir, log) for k in MODELS} for n in DATASET_ORDER}
    for n in DATASET_ORDER:
        for k in MODELS:
            b = tuning[n][k]["best"]
            log(f"  best {n}/{k}: {b['config']}  cv-RMSE {b['mean_rmse']:.4f}")

    # ---------------- 4 final runs -------------------------------------------
    log("== Stage 4: final training, 10 seeds per model")
    finals = {n: {k: train.final_runs(n, k, tuning[n][k]["best"]["config"], res_dir, log=log)
                  for k in MODELS} for n in DATASET_ORDER}

    # ---------------- 5 evaluation + statistics -----------------------------
    log("== Stage 5: evaluation and statistical tests")
    persist, stats_ds, best_model = {}, {}, {}
    overall_blocks = []
    for n in DATASET_ORDER:
        ser = load_series(n)
        tt = ser.test_targets
        persist[n] = metrics(ser.y[tt], persistence(ser.y, tt))
        S = np.column_stack([finals[n][k]["metrics"][:, 0] for k in MODELS])   # seeds x models
        fr = friedman_nemenyi(S, MODELS)
        wil = wilcoxon_holm(S, MODELS)
        best = MODELS[int(np.argmin(S.mean(0)))]
        best_model[n] = best
        stats_ds[n] = dict(friedman=fr, wilcoxon=wil, best=best,
                           mean_rmse={k: float(S[:, i].mean()) for i, k in enumerate(MODELS)},
                           persistence=persist[n])
        overall_blocks.append(np.column_stack([S, np.full(len(S), persist[n]["rmse"])]))
        log(f"  {n}: Friedman chi2={fr['chi2']:.2f} p={fr['p']:.4f} ranks={fr['ranks']} best={best}")
    names4 = MODELS + ["persistence"]
    blocks50 = np.vstack(overall_blocks)
    overall = friedman_nemenyi(blocks50, names4)
    overall_wil = wilcoxon_holm(blocks50, names4)
    ds_means = np.array([[stats_ds[n]["mean_rmse"][k] for k in MODELS] + [persist[n]["rmse"]]
                         for n in DATASET_ORDER])
    overall_n5 = friedman_nemenyi(ds_means, names4)
    log(f"  overall (dataset x seed blocks): {overall['ranks']} CD={overall['cd']:.3f} p={overall['p']:.2e}")
    log(f"  overall (5 dataset blocks): {overall_n5['ranks']} CD={overall_n5['cd']:.3f} p={overall_n5['p']:.3f}")
    summary = dict(persistence=persist, per_dataset=stats_ds, overall_50=overall,
                   overall_50_wilcoxon=overall_wil, overall_5=overall_n5, best_model=best_model,
                   selected={n: {k: tuning[n][k]["best"] for k in MODELS} for n in DATASET_ORDER},
                   final_metrics={n: {k: dict(
                       rmse=[float(v) for v in finals[n][k]["metrics"][:, 0]],
                       mae=[float(v) for v in finals[n][k]["metrics"][:, 1]],
                       smape=[float(v) for v in finals[n][k]["metrics"][:, 2]],
                       best_epoch=[int(v) for v in finals[n][k]["metrics"][:, 3]],
                       epochs_run=[int(v) for v in finals[n][k]["metrics"][:, 4]],
                       n_params=int(finals[n][k]["n_params"])) for k in MODELS} for n in DATASET_ORDER},
                   settings=dict(train=train.TRAIN_SETTINGS, n_configs=train.N_CONFIGS,
                                 n_folds=train.N_FOLDS, gap=train.CV_GAP, seeds=train.FINAL_SEEDS,
                                 search_space=train.SEARCH_SPACE, mrn_decays=train.MRN_DECAYS))
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    # ---------------- 6 figures + tables --------------------------
    log("== Stage 6: figures and tables")
    figures.fig_datasets(fig_dir)
    figures.fig_cv(fig_dir)
    figures.fig_loss_curves(fig_dir, finals)
    figures.fig_forecasts(fig_dir, finals, best_model)
    figures.fig_boxplots(fig_dir, finals, persist)
    figures.fig_cd(fig_dir, overall)
    tables.table_datasets(gen_dir, summ)
    tables.table_stationarity(gen_dir, st)
    tables.table_selected(gen_dir, tuning, finals)
    tables.table_results(gen_dir, finals, persist, best_model)
    tables.table_stats(gen_dir, stats_ds)



if __name__ == "__main__":
    main()
