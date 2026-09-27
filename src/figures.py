"""Figures for the report (vector PDF, sized for an IEEE two-column layout)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy as np

from .cv import expanding_window_folds
from .data import DATASET_ORDER, DATASETS, load_raw, load_series

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "lines.linewidth": 0.8, "figure.dpi": 150, "pdf.fonttype": 42})
COL_W, FULL_W = 3.5, 7.16
SHORT_TITLE = {"airline": "Airline", "sunspots": "Sunspots", "temperature": "Temperature",
               "mackey_glass": "Mackey-Glass", "sp500": "S&P 500"}
MODEL_LABEL = {"elman": "Elman", "jordan": "Jordan", "mrn": "MRN", "persistence": "Persistence"}
MODEL_COLOR = {"elman": "tab:blue", "jordan": "tab:orange", "mrn": "tab:green",
               "persistence": "0.5"}


def _save(fig, d, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)


def fig_datasets(d):
    fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 4.4))
    for ax, name in zip(axes.flat, DATASET_ORDER):
        s = load_raw(name)
        ser = load_series(name)
        ax.plot(s.index, s.values, color="k", lw=0.5)
        ax.axvspan(s.index[ser.N - ser.n_test], s.index[-1], color="tab:red", alpha=0.12,
                   label="test (20%)")
        ax.set_title(DATASETS[name]["title"])
        ax.set_ylabel(DATASETS[name]["unit"] if name != "mackey_glass" else "x(t)")
        if name == "sp500":
            ax.set_yscale("log")
    ser = load_series("airline")
    ax = axes.flat[5]
    ax.plot(load_raw("airline").index[1:], ser.u, color="tab:purple", lw=0.6)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_title("Airline after log + first difference")
    ax.set_ylabel(r"$\Delta\log y$")
    axes.flat[0].legend(loc="upper left")
    _save(fig, d, "fig_datasets.pdf")


def fig_cv(d):
    ser = load_series("airline")
    folds = expanding_window_folds(ser.dev_targets, 5)
    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    for k, (tr, va) in enumerate(folds):
        yk = 5 - k
        es_n = max(int(round(0.15 * len(tr))), 5)
        ax.barh(yk, len(tr) - es_n, left=tr[0], color="tab:blue", height=0.6)
        ax.barh(yk, es_n, left=tr[-1] - es_n + 1, color="tab:cyan", height=0.6)
        ax.barh(yk, len(va), left=va[0], color="tab:orange", height=0.6)
    ax.barh(0, ser.n_test, left=ser.test_targets[0], color="tab:red", height=0.6)
    ax.set_yticks(range(6), ["Test"] + [f"Fold {k}" for k in range(5, 0, -1)])
    ax.set_xlim(0, ser.N)
    ax.set_xlabel("Target index (Airline, months since 1949-01)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:blue", label="fit"),
                       Patch(color="tab:cyan", label="early stop"),
                       Patch(color="tab:orange", label="validation"),
                       Patch(color="tab:red", label="held-out test")],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
    _save(fig, d, "fig_cv.pdf")


def fig_loss_curves(d, finals):
    from matplotlib.ticker import NullFormatter, LogLocator
    fig, axes = plt.subplots(3, 5, figsize=(FULL_W, 3.9))
    for j, name in enumerate(DATASET_ORDER):
        for i, kind in enumerate(["elman", "jordan", "mrn"]):
            ax = axes[i, j]
            h = finals[name][kind]["histories"][0]
            l1, = ax.plot(np.log10(h["train"]), color="tab:blue")
            l2, = ax.plot(np.log10(h["val"]), color="tab:orange")
            be = int(np.argmin(h["val"]))
            l3 = ax.axvline(be, color="k", ls=":", lw=0.6)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if i == 0:
                ax.set_title(SHORT_TITLE[name])
            if j == 0:
                ax.set_ylabel(MODEL_LABEL[kind] + "\n" + r"$\log_{10}$ scaled MSE")
            if i == 2:
                ax.set_xlabel("epoch")
    fig.legend([l1, l2, l3], ["training set", "early-stopping set", "best epoch (weights restored)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), frameon=False)
    _save(fig, d, "fig_loss_curves.pdf")


def fig_forecasts(d, finals, best_model, last_n=120):
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 4.3))
    for ax, name in zip(axes.flat, DATASET_ORDER):
        ser = load_series(name)
        bm = best_model[name]
        f = finals[name][bm]
        t = f["targets"][-last_n:]
        x = load_raw(name).index[t]
        ax.plot(x, ser.y[t], color="k", lw=0.9)
        ax.plot(x, ser.y[t - 1], color=MODEL_COLOR["persistence"], lw=0.7, ls="--")
        mean = f["preds"].mean(0)[-last_n:]
        lo, hi = f["preds"].min(0)[-last_n:], f["preds"].max(0)[-last_n:]
        ax.plot(x, mean, color=MODEL_COLOR[bm], lw=0.9)
        ax.fill_between(x, lo, hi, color=MODEL_COLOR[bm], alpha=0.25, lw=0)
        n_t = len(f["targets"])
        ax.set_title(f"{SHORT_TITLE[name]}: lowest-RMSE RNN = {MODEL_LABEL[bm]} " +
                     (f"(last {last_n} of {n_t} test pts)" if n_t > last_n else f"({n_t} test pts)"))
        if hasattr(x, "year"):
            loc = mdates.AutoDateLocator(maxticks=6)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax = axes.flat[5]
    ax.axis("off")
    handles = [Line2D([], [], color="k", lw=0.9, label="actual"),
               Line2D([], [], color="0.5", ls="--", lw=0.7, label="persistence"),
               Line2D([], [], color=MODEL_COLOR["elman"], label="Elman (mean of 10 seeds)"),
               Line2D([], [], color=MODEL_COLOR["jordan"], label="Jordan (mean of 10 seeds)"),
               Line2D([], [], color=MODEL_COLOR["mrn"], label="MRN (mean of 10 seeds)"),
               Patch(color="0.6", alpha=0.4, label="min-max over the 10 seeds")]
    ax.legend(handles=handles, loc="center", fontsize=7, frameon=False)
    _save(fig, d, "fig_forecasts.pdf")


def fig_boxplots(d, finals, persist):
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 1.9))
    for ax, name in zip(axes, DATASET_ORDER):
        data = [finals[name][k]["metrics"][:, 0] for k in ["elman", "jordan", "mrn"]]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="k"))
        for patch, k in zip(bp["boxes"], ["elman", "jordan", "mrn"]):
            patch.set_facecolor(MODEL_COLOR[k])
            patch.set_alpha(0.6)
        ax.axhline(persist[name]["rmse"], color="0.4", ls="--", lw=0.7, label="persistence")
        lo = min(min(map(np.min, data)), persist[name]["rmse"])
        hi = max(max(map(np.max, data)), persist[name]["rmse"])
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_xticks([1, 2, 3], ["Elman", "Jordan", "MRN"])
        ax.set_title(SHORT_TITLE[name])
        if name == DATASET_ORDER[0]:
            ax.set_ylabel("test RMSE (10 seeds)")
            ax.legend(loc="upper left", fontsize=5.5)
    _save(fig, d, "fig_boxplot.pdf")


def fig_cd(d, fn):
    """Critical-difference diagram (Demsar, 2006): methods joined by a thick bar
    are not significantly different (Nemenyi, alpha = 0.05)."""
    ranks, cd, k = fn["ranks"], fn["cd"], fn["k"]
    names = sorted(ranks, key=ranks.get)
    fig, ax = plt.subplots(figsize=(COL_W, 1.5))
    lo, hi = 1, k
    ax.set_xlim(lo - 1.3, hi + 1.3)
    ax.set_ylim(-1.5, 1.0)
    ax.hlines(0, lo, hi, color="k", lw=0.8)
    for r in range(lo, hi + 1):
        ax.vlines(r, 0, 0.12, color="k", lw=0.8)
        ax.text(r, 0.22, str(r), ha="center", fontsize=6.5)
    half = (len(names) + 1) // 2
    for i, n in enumerate(names):
        left = i < half
        y = -0.55 - 0.28 * (i if left else len(names) - 1 - i)
        xe = lo - 0.15 if left else hi + 0.15
        ax.plot([ranks[n], ranks[n], xe], [0, y, y], color="0.3", lw=0.6)
        ax.text(xe + (-0.05 if left else 0.05), y, f"{MODEL_LABEL[n]} ({ranks[n]:.2f})", ha="right" if left else "left", va="center", fontsize=6.5)
    ax.hlines(0.65, lo, lo + cd, color="tab:red", lw=1.5)
    ax.text(lo + cd / 2, 0.75, f"CD = {cd:.2f}", ha="center", fontsize=6.5, color="tab:red")
    ordered = [ranks[n] for n in names]
    cliques = []
    for i in range(len(ordered)):
        j = max(jj for jj in range(len(ordered)) if ordered[jj] - ordered[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        ax.hlines(-0.14 - 0.12 * c, ordered[i] - 0.03, ordered[j] + 0.03, color="k", lw=2.2)
    ax.axis("off")
    _save(fig, d, "fig_cd.pdf")
