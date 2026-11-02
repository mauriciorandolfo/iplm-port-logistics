"""Generation of ILLUSTRATIVE input data (truck appointments, arrivals, ETT readings, service times).

The real operational records of the terminal are confidential. This module generates
synthetic data with the same structure as the variables listed in Table 8 of the paper
(truck_id, ett_min, truck_schedule, hour, truck_status, gate/yard times), so that the
model can be executed. Distributions and parameters are illustrative.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import Config

STATUS_ON_TIME, STATUS_EARLY, STATUS_DELAYED = 0, 1, 2
STATUS_LABELS = {0: "on_time", 1: "early", 2: "delayed"}
FEATURES = ["ett_min", "slack_start_min", "slack_end_min", "hour"]   # ETT, time window and hour


def _lognormal(rng, mean, sd, size):
    s2 = np.log(1.0 + (sd / mean) ** 2)
    return rng.lognormal(np.log(mean) - s2 / 2.0, np.sqrt(s2), size)


def generate_trucks(cfg: Config, rng: np.random.Generator, days: int) -> pd.DataFrame:
    """Appointments booked in time windows (with quota) and the corresponding real arrivals."""
    n_win = days * 24
    hp = np.asarray(cfg.hourly_profile) / np.sum(cfg.hourly_profile)
    daily = cfg.trucks_per_month * 12.0 / 365.0
    desired = np.concatenate([rng.multinomial(rng.poisson(daily), hp) for _ in range(days)])
    booked = np.zeros(n_win, dtype=int)
    carry = 0
    for w in range(n_win):                       # demand above the quota is moved to the next windows
        tot = desired[w] + carry
        booked[w] = min(tot, cfg.window_quota)
        carry = tot - booked[w]
    win = np.repeat(np.arange(n_win), booked)
    n = win.size
    ws = win * float(cfg.window_minutes)
    we = ws + cfg.window_minutes

    u = rng.random(n)
    nominal = np.where(u < cfg.p_on_time, STATUS_ON_TIME,
                       np.where(u < cfg.p_on_time + cfg.p_early, STATUS_EARLY, STATUS_DELAYED))
    arrival = np.where(nominal == STATUS_ON_TIME, ws + rng.random(n) * cfg.window_minutes,
                       np.where(nominal == STATUS_EARLY, ws - 1 - rng.exponential(cfg.early_mean, n),
                                we + 1 + rng.exponential(cfg.delay_mean, n)))
    arrival = np.maximum(arrival, 0.0)
    lead = np.clip(_lognormal(rng, cfg.tracking_lead_mean, cfg.tracking_lead_sd, n), 20, 400)
    t_pred = arrival - lead                      # moment of the ETT reading used by the classifiers
    ett = np.maximum(1.0, lead * (1.0 + rng.normal(0, cfg.ett_noise_sd, n)))
    # contextual estimate available to the dispatcher (e.g. contact with the driver): second, independent
    # noisy reading of the remaining travel time (Table 5)
    ett_ctx = np.maximum(1.0, lead * (1.0 + rng.normal(0, cfg.contextual_eta_noise_sd, n)))
    status = np.where(arrival < ws, STATUS_EARLY, np.where(arrival > we, STATUS_DELAYED, STATUS_ON_TIME))
    df = pd.DataFrame({
        "truck_id": np.arange(n),
        "window": win, "window_start": ws, "arrival": arrival, "t_pred": t_pred,
        "ett_min": ett, "ett_contextual_min": ett_ctx, "slack_start_min": ws - t_pred, "slack_end_min": we - t_pred,
        "hour": (np.maximum(t_pred, 0) // 60) % 24,
        "truck_status": status,
    })
    return df


def generate_service_times(cfg: Config, rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    return {
        "gate_in": np.maximum(0.5, rng.normal(cfg.gate_service_mean, cfg.gate_service_sd, n)),
        "gate_out": np.maximum(0.5, rng.normal(cfg.gate_service_mean, cfg.gate_service_sd, n)),
        "travel": np.maximum(1.0, rng.normal(cfg.internal_travel_mean, cfg.internal_travel_sd, n)),
        "handling": np.maximum(1.0, rng.normal(cfg.handling_mean, cfg.handling_sd, n)),
    }


def training_dataset(cfg: Config, seed: int) -> pd.DataFrame:
    """Synthetic historical sample used to train the classifiers (independent of the simulated year)."""
    rng = np.random.default_rng(seed)
    days = max(1, int(np.ceil(cfg.n_training_records / (cfg.trucks_per_month * 12 / 365))))
    return generate_trucks(cfg, rng, days).head(cfg.n_training_records)
