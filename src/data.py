
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Dataset registry
DATASETS = {
    "airline": dict(
        title="Airline Passengers", file="airline-passengers.csv", date_col="Month",
        value_col="Passengers", freq="MS", log=True, diff=True, unit="thousands",
        source="Box & Jenkins (1976), via github.com/jbrownlee/Datasets",
        frequency="Monthly"),
    "sunspots": dict(
        title="Monthly Sunspots", file="monthly-sunspots.csv", date_col="Month",
        value_col="Sunspots", freq="MS", log=False, diff=False, unit="count",
        source="SIDC/SILSO (Zurich series), via github.com/jbrownlee/Datasets",
        frequency="Monthly"),
    "temperature": dict(
        title="Min. Temperatures (Melbourne)", file="daily-min-temperatures.csv",
        date_col="Date", value_col="Temp", freq="D", log=False, diff=False, unit="deg C",
        source="Australian Bureau of Meteorology, via github.com/jbrownlee/Datasets",
        frequency="Daily"),
    "mackey_glass": dict(
        title="Mackey-Glass (tau=17)", file=None, log=False, diff=False, unit="-",
        source="Synthetic: Mackey & Glass (1977) delay differential equation",
        frequency="Unit time step"),
    "sp500": dict(
        title="S&P 500 Index", file="sp500.csv", date_col="Date", value_col="SP500",
        freq="MS", log=True, diff=True, unit="index points",
        start="1950-01-01", end="2024-12-01",
        source="Shiller/S&P via github.com/datasets/s-and-p-500",
        frequency="Monthly"),
}
DATASET_ORDER = ["airline", "sunspots", "temperature", "mackey_glass", "sp500"]


def mackey_glass(n: int = 1500, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                 p: float = 10.0, dt: float = 0.01, sample_every: float = 1.0,
                 discard: int = 500, x0: float = 1.2) -> np.ndarray:
    """
    Integrate dx/dt = beta*x(t-tau)/(1+x(t-tau)^p) - gamma*x(t) with Euler.
    """
    delay = int(round(tau / dt))
    stride = int(round(sample_every / dt))
    total = (n + discard) * stride
    x = np.empty(total + delay + 1)
    x[: delay + 1] = x0
    for i in range(delay, total + delay):
        xd = x[i - delay]
        x[i + 1] = x[i] + dt * (beta * xd / (1.0 + xd ** p) - gamma * x[i])
    samples = x[delay::stride][: n + discard]
    return samples[discard:]


def load_raw(name: str) -> pd.Series:
    """
    Return the raw series.
    """
    meta = DATASETS[name]
    if meta["file"] is None:
        cache = os.path.join(DATA_DIR, "mackey_glass.csv")
        if not os.path.exists(cache):
            s = pd.Series(mackey_glass(), name="x")
            s.index.name = "t"
            s.to_csv(cache)
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]
        s.attrs["n_missing"] = 0
        return s.astype(float)
    df = pd.read_csv(os.path.join(DATA_DIR, meta["file"]))
    df[meta["date_col"]] = pd.to_datetime(df[meta["date_col"]])
    s = df.set_index(meta["date_col"])[meta["value_col"]]
    s = pd.to_numeric(s, errors="coerce")
    if "start" in meta:
        s = s.loc[meta["start"]:meta["end"]]
    full = pd.date_range(s.index.min(), s.index.max(), freq=meta["freq"])
    s = s.reindex(full)
    n_missing = int(s.isna().sum())
    s = s.interpolate(method="linear", limit_direction="both")
    s.attrs["n_missing"] = n_missing
    s.name = meta["title"]
    return s.astype(float)



# --------------- Transforms -------------------
@dataclass
class Transform:
    """Log / first-difference transform with one-step inverse."""
    log: bool = False
    diff: bool = False

    @property
    def offset(self) -> int:
        return 1 if self.diff else 0

    def level(self, y: np.ndarray) -> np.ndarray:
        return np.log(y) if self.log else np.asarray(y, dtype=float)

    def forward(self, y: np.ndarray) -> np.ndarray:
        z = self.level(y)
        return np.diff(z) if self.diff else z

    def inverse_one_step(self, u_hat: np.ndarray, y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Map predictions of u at raw indices ``targets`` back to the original scale.
        """
        z = self.level(y)
        lvl = z[targets - 1] + u_hat if self.diff else u_hat
        return np.exp(lvl) if self.log else lvl


@dataclass
class Scaler:
    """z-score scaler; fitted on training data only."""
    mean: float = 0.0
    std: float = 1.0

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean = float(np.mean(x))
        self.std = float(np.std(x)) or 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


# Series container with split bookkeeping
MAX_WINDOW = 24     # largest window in the search space; fixes the first target
TEST_FRAC = 0.20


@dataclass
class Series:
    name: str
    y: np.ndarray
    index: pd.Index
    tf: Transform
    u: np.ndarray = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self):
        self.u = self.tf.forward(self.y)
        self.n_test = int(round(TEST_FRAC * len(self.y)))

    @property
    def N(self):
        return len(self.y)

    @property
    def first_target(self):
        return MAX_WINDOW + self.tf.offset

    @property
    def dev_targets(self) -> np.ndarray:
        return np.arange(self.first_target, self.N - self.n_test)

    @property
    def test_targets(self) -> np.ndarray:
        return np.arange(self.N - self.n_test, self.N)

    def u_index(self, t):
        return np.asarray(t) - self.tf.offset

    def fit_scaler(self, train_targets: np.ndarray) -> Scaler:
        """Fit on every transformed value observable up to the last training target."""
        last = self.u_index(train_targets.max())
        return Scaler().fit(self.u[: last + 1])

    def windows(self, targets: np.ndarray, W: int, scaler: Scaler):
        """Return X (n, W, 1) and Y (n, W): Y[:, s] is the one-step target after x_s.

        Y[:, -1] is the actual forecast target u[t]; earlier columns are the
        next values inside the window (used for all-step supervision).
        """
        us = scaler.transform(self.u)
        ui = self.u_index(targets)
        idx = ui[:, None] + np.arange(-W, 0)[None, :]
        X = us[idx][..., None].astype(np.float32)
        Y = us[idx + 1].astype(np.float32)
        return X, Y

    def to_original(self, u_hat_scaled: np.ndarray, targets: np.ndarray, scaler: Scaler):
        return self.tf.inverse_one_step(scaler.inverse(u_hat_scaled), self.y, targets)


def load_series(name: str) -> Series:
    s = load_raw(name)
    meta = DATASETS[name]
    return Series(name=name, y=s.values.astype(float), index=s.index, tf=Transform(log=meta["log"], diff=meta["diff"]))


# Stationarity
def adf_kpss(x: np.ndarray) -> dict:
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, autolag="AIC", regression="c")
        kp = kpss(x, regression="c", nlags="auto")
    return dict(adf_stat=adf[0], adf_p=adf[1], kpss_stat=kp[0], kpss_p=kp[1])


def verdict(r: dict, alpha: float = 0.05) -> str:
    adf_rej = r["adf_p"] < alpha          # reject unit root  -> evidence for stationarity
    kpss_rej = r["kpss_p"] < alpha        # reject stationarity
    if adf_rej and not kpss_rej:
        return "Stationary"
    if not adf_rej and kpss_rej:
        return "Non-stationary"
    if adf_rej and kpss_rej:
        return "Conflicting (diff.-stat.)"
    return "Inconclusive"


def stationarity_table() -> pd.DataFrame:
    """ADF + KPSS on the development part (first 80%) only, raw and transformed."""
    rows = []
    for name in DATASET_ORDER:
        ser = load_series(name)
        dev_end = ser.N - ser.n_test
        raw = ser.y[:dev_end]
        r = adf_kpss(raw)
        rows.append(dict(dataset=name, series="raw", **r, verdict=verdict(r)))
        if ser.tf.log or ser.tf.diff:
            tr = ser.u[: dev_end - ser.tf.offset]
            r = adf_kpss(tr)
            lab = ("log-diff" if ser.tf.log and ser.tf.diff else "diff" if ser.tf.diff else "log")
            rows.append(dict(dataset=name, series=lab, **r, verdict=verdict(r)))
        if name == "airline":
            # Considered alternative: additional seasonal (lag-12) difference.
            d = ser.u[: dev_end - ser.tf.offset]
            r = adf_kpss(d[12:] - d[:-12])
            rows.append(dict(dataset=name, series="log-diff-sdiff12 (not used)", **r, verdict=verdict(r)))
    return pd.DataFrame(rows)


def summary_table() -> pd.DataFrame:
    rows = []
    for name in DATASET_ORDER:
        s = load_raw(name)
        meta = DATASETS[name]
        idx = s.index
        span = (f"{idx[0]:%Y-%m}--{idx[-1]:%Y-%m}" if isinstance(idx, pd.DatetimeIndex)
                else f"t=0--{len(s)-1}")
        rows.append(dict(dataset=name, title=meta["title"], frequency=meta["frequency"],
                         length=len(s), span=span, missing=s.attrs.get("n_missing", 0),
                         mean=s.mean(), std=s.std(), min=s.min(), max=s.max(),
                         source=meta["source"]))
    return pd.DataFrame(rows)
