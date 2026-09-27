"""Write LaTeX tables (report/generated/*.tex) straight from the results"""
from __future__ import annotations

import os

import numpy as np

from .data import DATASET_ORDER, DATASETS

SHORT = {"airline": "Airline", "sunspots": "Sunspots", "temperature": "Temperature",
         "mackey_glass": "Mackey-Glass", "sp500": "S\\&P 500"}
ML = {"elman": "Elman", "jordan": "Jordan", "mrn": "MRN", "persistence": "Persist."}


def _w(d, name, body):
    with open(os.path.join(d, name), "w") as f:
        f.write(body)

"""
Format a p-value.  KPSS p-values are interpolated from a table and are therefore only known to lie in [0.01, 0.10]
"""
def _p(p, bounded=False):
    
    if not bounded:
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"
    if p >= 0.1 - 1e-12:
        return "$\\geq$0.10"
    if p <= 0.01 + 1e-12 and p >= 0.01 - 1e-12:
        return "$\\leq$0.01"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _fmt(x):
    ax = abs(x)
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def _dec(x):
    ax = abs(x)
    return 1 if ax >= 100 else 2 if ax >= 1 else 4


def _pm(m, s):
    k = _dec(m)
    return f"{m:.{k}f}$\\pm${s:.{k}f}"


def table_datasets(d, summ):
    rows = []
    for _, r in summ.iterrows():
        rows.append(f"{SHORT[r.dataset]} & {r.frequency} & {r.span} & {r.length} & {r.missing} & "
                    f"{_fmt(r['mean'])} & {_fmt(r['std'])} & {_fmt(r['min'])} & {_fmt(r['max'])} \\\\")
    _w(d, "tab_datasets.tex", "\n".join(rows) + "\n")


def table_stationarity(d, st):
    rows = []
    for _, r in st.iterrows():
        rows.append(f"{SHORT[r.dataset]} & {r.series.replace('_', ' ')} & {r.adf_stat:.2f} & {_p(r.adf_p)} & "
                    f"{r.kpss_stat:.3f} & {_p(r.kpss_p, bounded=True)} & {r.verdict} \\\\")
    _w(d, "tab_stationarity.tex", "\n".join(rows) + "\n")


def table_selected(d, tuning, finals):
    rows = []
    for name in DATASET_ORDER:
        for i, kind in enumerate(["elman", "jordan", "mrn"]):
            b = tuning[name][kind]["best"]
            c = b["config"]
            dec = ",".join(f"{a:g}" for a in c["decays"]) if "decays" in c else "--"
            lab = f"\\multirow{{3}}{{*}}{{{SHORT[name]}}}" if i == 0 else ""
            ep = finals[name][kind]["metrics"][:, 3]
            rows.append(f"{lab} & {ML[kind]} & {c['hidden']} & {c['window']} & {c['lr']:g} & "
                        f"{c['weight_decay']:g} & {dec} & {int(finals[name][kind]['n_params'])} & "
                        f"{_pm(b['mean_rmse'], b['std_rmse'])} & {np.median(ep):.0f} \\\\")
        rows.append("\\midrule" if name != DATASET_ORDER[-1] else "")
    _w(d, "tab_selected.tex", "\n".join(rows) + "\n")


def table_results(d, finals, persist, best_model):
    rows = []
    for name in DATASET_ORDER:
        for mi, (mkey, mname) in enumerate([("rmse", "RMSE"), ("mae", "MAE"), ("smape", "sMAPE (\\%)")]):
            col = {"rmse": 0, "mae": 1, "smape": 2}[mkey]
            means = {k: finals[name][k]["metrics"][:, col].mean() for k in ["elman", "jordan", "mrn"]}
            best = min(means, key=means.get)
            cells = []
            for k in ["elman", "jordan", "mrn"]:
                v = finals[name][k]["metrics"][:, col]
                s = _pm(v.mean(), v.std())
                cells.append(f"\\textbf{{{s}}}" if k == best else s)
            pv = persist[name][mkey]
            lab = f"\\multirow{{3}}{{*}}{{{SHORT[name]}}}" if mi == 0 else ""
            rows.append(f"{lab} & {mname} & " + " & ".join(cells) + f" & {_fmt(pv)} \\\\")
        rows.append("\\midrule" if name != DATASET_ORDER[-1] else "")
    _w(d, "tab_results.tex", "\n".join(rows) + "\n")


def table_stats(d, stats_ds):
    rows = []
    for name in DATASET_ORDER:
        s = stats_ds[name]
        fr = s["friedman"]
        rk = " / ".join(f"{fr['ranks'][k]:.1f}" for k in ["elman", "jordan", "mrn"])
        sig = [p.replace("elman", "E").replace("jordan", "J").replace("mrn", "M")
               for p, v in fr["pairs"].items() if v["significant"]] if fr["p"] < 0.05 else []
        wil = " / ".join(_p(s["wilcoxon"][p]["p_holm"]) for p in ["elman-jordan", "elman-mrn", "jordan-mrn"])
        rows.append(f"{SHORT[name]} & {fr['chi2']:.2f} & {_p(fr['p'])} & {rk} & "
                    f"{', '.join(sig) if sig else 'none'} & {wil} & {ML[s['best']]} \\\\")
    _w(d, "tab_stats.tex", "\n".join(rows) + "\n")
