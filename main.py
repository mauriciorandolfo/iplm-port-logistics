"""Command-line entry point of the IPLM.

Examples
--------
python main.py hra                                   # Tables 3, 4 and 9 (HRA)
python main.py ml                                    # classifier metrics (structure of Tables 11-12)
python main.py pilot                                 # pilot sample and number of replications (Section 4)
python main.py run --reps 25                         # baseline, autonomous and human-centric (Table 13)
python main.py sensitivity --reps 25                 # HEP multipliers 0.5, 1, 1.5, 2 (Table 14)
python main.py run --days 30 --reps 2                # quick test

Results are written to the folder `outputs/`. With the illustrative parameters of
iplm/config.py, the results do NOT reproduce those reported in the paper.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from iplm.config import SCENARIOS, Config
from iplm.experiment import (evaluate_prediction_module, required_replications, run_experiment, summarise,
                             train_prediction_module)
from iplm.hra import hra_tables

OUT = Path(__file__).resolve().parent / "outputs"


def _save(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description="Intelligent Port Logistics Model (IPLM)")
    ap.add_argument("command", choices=["hra", "ml", "run", "sensitivity", "pilot"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--reps", type=int, default=None, help="number of replications (default: Config.replications)")
    ap.add_argument("--days", type=int, default=None, help="simulated days per replication (default: 365)")
    ap.add_argument("--multipliers", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    cfg = Config()
    if a.days:
        cfg = cfg.with_(horizon_days=a.days)
    if a.seed is not None:
        cfg = cfg.with_(seed=a.seed)

    if a.command == "hra":
        print(hra_tables(cfg))
        return
    tas = train_prediction_module(cfg)
    if a.command == "ml":
        df = evaluate_prediction_module(cfg, tas)
        print(df.round(4).to_string(index=False))
        print("saved:", _save(df, "ml_metrics"))
    elif a.command == "pilot":
        df = run_experiment(cfg, a.scenarios, a.reps or 10, tas)
        print(required_replications(df).round(3).to_string(index=False))
    elif a.command == "run":
        df = run_experiment(cfg, a.scenarios, a.reps, tas)
        print("saved:", _save(df, "replications"))
        s = summarise(df)
        print(s[["scenario", "avg_trucks_queue_gate", "max_trucks_queue_gate", "avg_wait_gate_s", "max_wait_gate_s",
                 "avg_wait_yard_s", "avg_dwell_s", "pct_corrective_maintenance", "pct_arriving_outside_time_window",
                 "pct_admitted_outside_time_window"]].round(2).to_string(index=False))
        print("saved:", _save(s, "summary"))
    elif a.command == "sensitivity":
        frames = []
        for mlt in a.multipliers:
            print(f"HEP multiplier {mlt}")
            frames.append(run_experiment(cfg.with_(hep_multiplier=mlt), ["human_centric"], a.reps, tas))
        df = pd.concat(frames, ignore_index=True)
        print("saved:", _save(df, "sensitivity_replications"))
        s = summarise(df)
        print(s[["hep_multiplier", "avg_trucks_queue_gate", "avg_wait_gate_s", "avg_wait_yard_s", "avg_dwell_s",
                 "pct_corrective_maintenance", "pct_admitted_outside_time_window"]].round(2).to_string(index=False))
        print("saved:", _save(s, "sensitivity_summary"))


if __name__ == "__main__":
    main()
