"""Key performance indicators (Sections 3.2 and 4, Table 13)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import MIN_PER_DAY


def queue_statistics(arrive: np.ndarray, start: np.ndarray, t0: float, t1: float):
    """Time-average and maximum number of entities waiting between `arrive` and `start` in [t0, t1]."""
    ok = np.isfinite(arrive) & np.isfinite(start)
    a, s = arrive[ok], start[ok]
    overlap = np.clip(np.minimum(s, t1) - np.maximum(a, t0), 0, None)
    avg = float(overlap.sum() / (t1 - t0))
    times = np.concatenate([a, s])
    sign = np.concatenate([np.ones(a.size), -np.ones(s.size)])
    order = np.lexsort((sign, times))                 # departures before arrivals at equal times
    level = np.cumsum(sign[order])
    tt = times[order]
    m = (tt >= t0) & (tt < t1)
    return avg, float(level[m].max()) if m.any() else 0.0


def daily_mean_queue(arrive: np.ndarray, start: np.ndarray, t0: float, t1: float, step: float = 10.0):
    """Daily mean number of trucks in the gate queue (for the 7-day moving-average plot of Figure 6)."""
    ok = np.isfinite(arrive) & np.isfinite(start)
    grid = np.arange(t0, t1, step)
    q = np.searchsorted(np.sort(arrive[ok]), grid, side="right") - np.searchsorted(np.sort(start[ok]), grid, side="right")
    per_day = int(MIN_PER_DAY / step)
    nd = q.size // per_day
    return q[: nd * per_day].reshape(nd, per_day).mean(axis=1)


def compute_kpis(twin, t0: float, t1: float) -> Dict[str, float]:
    r = twin.rec
    cfg = twin.cfg
    tr = twin.tr
    m = (r["gate_start"] >= t0) & (r["gate_start"] < t1) & np.isfinite(r["gateout_end"])
    gq = np.asarray(twin.gate_queue_log, float).reshape(-1, 2)
    q_avg, q_max = queue_statistics(gq[:, 0], gq[:, 1], t0, t1)
    gate_wait = r["gate_wait"][m] * 60.0                 # queue at the gate only (LSA time excluded)
    yard_wait = (r["yard_start"] - r["yard_request"])[m] * 60.0
    dwell = (r["gateout_end"] - r["gate_start"])[m] * 60.0
    w = float(cfg.window_minutes)
    # trucks arriving outside their booked window (carrier punctuality, input of the model)
    ws0 = tr["window_start"].to_numpy()
    arr = tr["arrival"].to_numpy()
    arriving_outside = ((arr < ws0) | (arr > ws0 + w))[m]
    # trucks admitted at the gate outside their (booked or rescheduled) window
    ws = twin.final_window * w
    admitted_outside = ((r["gate_start"] < ws) | (r["gate_start"] > ws + w))[m]
    dt = [(k, min(e, t1) - max(s, t0)) for (_, k, s, e) in twin.downtime if e > t0 and s < t1]
    total = sum(d for _, d in dt) / 60.0
    corrective = sum(d for k, d in dt if k == "corrective") / 60.0
    out = {
        "avg_trucks_queue_gate": q_avg,
        "max_trucks_queue_gate": q_max,
        "avg_wait_gate_s": float(gate_wait.mean()),
        "max_wait_gate_s": float(gate_wait.max()),
        "avg_wait_yard_s": float(yard_wait.mean()),
        "avg_dwell_s": float(dwell.mean()),
        "downtime_total_h": total,
        "pct_corrective_maintenance": 100.0 * corrective / total if total else 0.0,
        "pct_arriving_outside_time_window": 100.0 * float(arriving_outside.mean()),
        "pct_admitted_outside_time_window": 100.0 * float(admitted_outside.mean()),
        "pct_rescheduled": 100.0 * float(twin.rescheduled[m].mean()),
        "pct_sent_to_lsa_at_gate": 100.0 * float(twin.fallback[m].mean()),
        "avg_lsa_wait_min": float(r["lsa_wait"][m].mean()),
        "n_trucks": int(m.sum()),
        "ga_executions": len(twin.ga_log),
        "ga_cpu_mean_s": float(np.mean([g[3] for g in twin.ga_log])) if twin.ga_log else 0.0,
        "ga_cpu_max_s": float(np.max([g[3] for g in twin.ga_log])) if twin.ga_log else 0.0,
    }
    for d, log in twin.human.log.items():
        out[f"human_{d}_events"] = log["events"]
        out[f"human_{d}_deviations"] = log["delay"] + log["suboptimal"] + log["improvement"]
    return out
