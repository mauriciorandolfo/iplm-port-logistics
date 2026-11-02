"""Verification tests of the IPLM. Run with `python -m pytest tests` or `python tests/test_verification.py`."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iplm import ga_optimizer  # noqa: E402
from iplm.config import Config, MIN_PER_DAY  # noqa: E402
from iplm.experiment import run_replication, train_prediction_module  # noqa: E402
from iplm.hra import human_error_probabilities  # noqa: E402

QUICK = Config().with_(horizon_days=10, n_training_records=20_000)
_TAS = None


def tas():
    global _TAS
    if _TAS is None:
        _TAS = train_prediction_module(QUICK)
    return _TAS


def test_hra_values_of_the_paper():
    """Equation 5 reproduces Table 4 (0.0791, 0.0479, 0.1092)."""
    h = human_error_probabilities(Config())
    assert round(h["maint"], 4) == 0.0791
    assert round(h["storage"], 4) == 0.0479
    assert round(h["sched"], 4) == 0.1092


def test_ga_fitness():
    """Equation 6: concurrency penalty and night-time reward."""
    cfg = Config()
    night = 2 * 60.0                      # 02:00
    day = 14 * 60.0                       # 14:00
    assert ga_optimizer.fitness([night], ["E1"], cfg, 0.0, []) == -cfg.ga_night_reward
    assert ga_optimizer.fitness([day], ["E1"], cfg, 0.0, []) == 0.0
    assert ga_optimizer.fitness([day, day], ["E1", "E2"], cfg, 0.0, []) == cfg.ga_penalty
    # consecutive interventions (one ends when the other starts) are not concurrent
    assert ga_optimizer.fitness([day, day + cfg.preventive_duration], ["E1", "E2"], cfg, 0.0, []) == 0.0


def test_ga_short_overlap_is_detected():
    """Short overlaps between interventions are detected by the interval sweep."""
    s1 = 10 * 60.0 + 5.0                  # 10:05-12:05
    s2 = s1 + 116.0                       # 12:01-14:01 (4-min overlap)
    assert ga_optimizer.max_concurrency([(s1, s1 + 120), (s2, s2 + 120)], 0, 1440) == 2
    assert ga_optimizer.max_concurrency([(s1, s1 + 120), (s1 + 120, s1 + 240)], 0, 1440) == 1


def test_ga_same_unit_overlap_is_penalised():
    """Two overlapping interventions on the same unit are infeasible (penalised even if concurrency is allowed)."""
    cfg = Config().with_(max_concurrent_maint=5)
    day = 14 * 60.0
    assert ga_optimizer.fitness([day, day + 60.0], ["E1", "E2"], cfg, 0.0, []) == 0.0
    assert ga_optimizer.fitness([day, day + 60.0], ["E1", "E1"], cfg, 0.0, []) == cfg.ga_penalty


def test_ga_integer_genes_within_horizon():
    cfg = Config()
    plan, _, _ = ga_optimizer.optimize(cfg, ["E1", "E2", "E3"], 0.0, [], seed=7)
    hi = cfg.ga_horizon_h * 60.0 - cfg.preventive_duration
    for jobs in plan.values():
        assert len(jobs) == cfg.ga_jobs_per_unit
        for s, e in jobs:
            assert float(s).is_integer() and 0 <= s <= hi and e - s == cfg.preventive_duration


def test_human_layer_without_deviations_equals_autonomous():
    """With HEP = 0 and no discretionary admission, the human-centric logic reduces to the autonomous one."""
    cfg = QUICK.with_(hep_multiplier=0.0, admission_queue_limit=-1)
    a = run_replication(QUICK, "autonomous", 0, tas())
    h = run_replication(cfg, "human_centric", 0, tas())
    for k in ("avg_dwell_s", "avg_wait_gate_s", "avg_wait_yard_s", "pct_corrective_maintenance"):
        assert abs(a[k] - h[k]) < 1e-9, k


def test_gate_admission_rules():
    """Autonomous: no truck admitted outside its window (fallback to the LSA).
    Human-centric: discretionary admission. Baseline: admissions outside the window include queue effects."""
    b = run_replication(QUICK, "baseline", 0, tas())
    a = run_replication(QUICK, "autonomous", 0, tas())
    h = run_replication(QUICK, "human_centric", 0, tas())
    assert a["pct_admitted_outside_time_window"] == 0.0
    assert h["pct_admitted_outside_time_window"] > 0.0
    assert b["pct_admitted_outside_time_window"] >= b["pct_arriving_outside_time_window"] - 1e-9


def test_conservation_and_littles_law():
    r = run_replication(QUICK, "autonomous", 1, tas())
    lam = r["n_trucks"] / (QUICK.horizon_days * MIN_PER_DAY)
    little = lam * r["avg_wait_gate_s"] / 60.0
    assert r["n_trucks"] > 0
    assert abs(little - r["avg_trucks_queue_gate"]) / r["avg_trucks_queue_gate"] < 0.02


def test_deterministic_case():
    """Without variability, failures and queues, the dwell time equals the sum of service and travel times."""
    cfg = QUICK.with_(trucks_per_month=1500, gate_service_sd=1e-9, internal_travel_sd=1e-9, handling_sd=1e-9,
                      breakdown_rate_per_h=1e-12, initial_rul_h=(1e6, 1e6 + 1), efficiency_decay_per_day=0.0,
                      baseline_pm_interval_days=10_000)
    r = run_replication(cfg, "baseline", 0, tas())
    expected = (cfg.gate_service_mean * 2 + cfg.internal_travel_mean + cfg.handling_mean) * 60.0
    assert r["avg_wait_yard_s"] < 1e-6
    assert abs(r["avg_dwell_s"] - expected) < 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
