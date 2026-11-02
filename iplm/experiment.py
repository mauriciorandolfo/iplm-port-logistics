"""Replications, scenarios and sensitivity analysis (Sections 4 and 5)."""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from .config import Config, MIN_PER_DAY
from .data import generate_service_times, generate_trucks, training_dataset
from .digital_twin import DigitalTwin
from .hra import human_error_probabilities
from .human import HumanDecisionLayer
from .kpi_tracker import compute_kpis, daily_mean_queue
from .ml_module import AdaptiveTAS, classification_metrics


def split_historical_records(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split of the (illustrative) historical records: 70% training, 30% test (Section 3.3)."""
    df = training_dataset(cfg, seed=cfg.seed + 999)
    return train_test_split(df, test_size=cfg.test_size, stratify=df["truck_status"], random_state=cfg.seed)


def train_prediction_module(cfg: Config) -> AdaptiveTAS:
    """Trains the four classifiers on the training subset of the historical records."""
    train, _ = split_historical_records(cfg)
    return AdaptiveTAS(seed=cfg.seed).fit(train)


def run_replication(cfg: Config, scenario: str, rep: int, tas: AdaptiveTAS, keep_series: bool = False) -> Dict:
    """One replication of one scenario. All scenarios of a replication use the same random seeds."""
    streams = np.random.SeedSequence([cfg.seed, rep]).spawn(5)
    rng_tr, rng_sv, rng_eq, rng_ops, rng_hu = (np.random.default_rng(s) for s in streams)
    days = cfg.warmup_days + cfg.horizon_days
    trucks = generate_trucks(cfg, rng_tr, days)
    services = generate_service_times(cfg, rng_sv, len(trucks))
    ens = None
    ml_info = {}
    if scenario != "baseline":
        preds = tas.predict_all(trucks)
        ens, sel = AdaptiveTAS.dynamic_selection(preds, trucks["truck_status"].to_numpy(),
                                                 trucks["t_pred"].to_numpy(), trucks["arrival"].to_numpy())
        ml_info = {"ensemble_accuracy": float((ens == trucks["truck_status"].to_numpy()).mean())}
        ml_info.update({f"selected_{n}": float((sel == k).mean()) for k, n in enumerate(("DT", "RF", "GBM", "LR"))})
    human = HumanDecisionLayer(cfg, human_error_probabilities(cfg), rng_hu, active=(scenario == "human_centric"))
    twin = DigitalTwin(cfg, scenario, trucks, services, ens, human, rng_eq, rng_ops, seed=cfg.seed + 7919 * rep)
    t0, t1 = cfg.warmup_days * MIN_PER_DAY, days * MIN_PER_DAY
    wall = time.perf_counter()
    twin.run(until=t1 + MIN_PER_DAY)          # one extra day to complete the trucks in the system
    out = {"scenario": scenario, "replication": rep, "hep_multiplier": cfg.hep_multiplier}
    out.update(compute_kpis(twin, t0, t1))
    out.update(ml_info)
    out["wall_time_s"] = time.perf_counter() - wall
    if keep_series:
        out["daily_mean_queue"] = daily_mean_queue(twin.rec["gate_arrival"], twin.rec["gate_start"], t0, t1)
    return out


def run_experiment(cfg: Config, scenarios: Iterable[str], replications: Optional[int] = None,
                   tas: Optional[AdaptiveTAS] = None, verbose: bool = True) -> pd.DataFrame:
    tas = tas or train_prediction_module(cfg)
    rows: List[Dict] = []
    for rep in range(replications or cfg.replications):
        for sc in scenarios:
            r = run_replication(cfg, sc, rep, tas)
            rows.append(r)
            if verbose:
                print(f"  rep {rep + 1:3d}  {sc:14s}  dwell = {r['avg_dwell_s']:8.1f} s  ({r['wall_time_s']:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by=("scenario", "hep_multiplier"), conf: float = 0.95) -> pd.DataFrame:
    """Mean and confidence half-width of every KPI."""
    kpis = [c for c in df.columns if c not in ("scenario", "replication", "hep_multiplier") and df[c].dtype != object]
    rows = []
    for key, g in df.groupby(list(by)):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        n = len(g)
        for k in kpis:
            x = g[k].to_numpy(float)
            h = stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            row[k] = x.mean()
            row[k + "_hw"] = h
        row["n_replications"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def required_replications(df: pd.DataFrame, kpi: str = "avg_wait_gate_s", rel_error: float = 0.05,
                          conf: float = 0.95) -> pd.DataFrame:
    """Number of replications for a given relative error of the mean, from a pilot sample (Section 4):
    n = (z * s / (rel_error * mean))^2, normal approximation."""
    z = stats.norm.ppf(0.5 + conf / 2)
    rows = []
    for sc, g in df.groupby("scenario"):
        x = g[kpi].to_numpy(float)
        s = x.std(ddof=1)
        rows.append({"scenario": sc, "n0": len(x), "mean": x.mean(), "sd": s,
                     "n_required": int(np.ceil((z * s / (rel_error * x.mean())) ** 2))})
    return pd.DataFrame(rows)


def evaluate_prediction_module(cfg: Config, tas: Optional[AdaptiveTAS] = None) -> pd.DataFrame:
    """Accuracy and class-wise metrics of the classifiers and of the dynamic selection on the test subset (Tables 11-12)."""
    tas = tas or train_prediction_module(cfg)
    _, test = split_historical_records(cfg)
    y = test["truck_status"].to_numpy()
    preds = tas.predict_all(test)
    ens, _ = AdaptiveTAS.dynamic_selection(preds, y, test["t_pred"].to_numpy(), test["arrival"].to_numpy())
    rows = []
    for name, p in list(zip(("DT", "RF", "GBM", "LR"), preds)) + [("Ensemble", ens)]:
        m = classification_metrics(y, p)
        row = {"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        for k, cls in enumerate(("on_time", "early", "delayed")):
            row[f"precision_{cls}"] = m["precision"][k]
            row[f"recall_{cls}"] = m["recall"][k]
            row[f"f1_{cls}"] = m["f1"][k]
        rows.append(row)
    return pd.DataFrame(rows)
