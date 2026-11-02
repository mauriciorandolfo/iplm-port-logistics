"""Configuration of the Intelligent Port Logistics Model (IPLM).

IMPORTANT - ILLUSTRATIVE VALUES
-------------------------------
The operational records of the terminal used in the paper and the input
distributions fitted to them are confidential. All numerical values in this file
are ILLUSTRATIVE: they allow the model to be executed and inspected, but they do
NOT reproduce the results reported in the paper. Exceptions are the values that are
published in the paper itself (human-factor contributions, alpha, nominal HEPs,
GA parameters, human decision logic of Table 5, gate admission limit, punctuality shares and
monthly truck volume), which are indicated with the corresponding section/table/equation.

Time unit: minutes, unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

MIN_PER_DAY = 24 * 60

SCENARIOS = ("baseline", "autonomous", "human_centric")
DECISIONS = ("maint", "storage", "sched")   # maintenance programming, storage management, truck scheduling
HF_FACTORS = ("ERG", "PER", "LER", "COG", "ASP")


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------- experiment
    seed: int = 12345
    horizon_days: int = 365                  # one operational year per replication (Section 5)
    replications: int = 25                   # Section 4
    warmup_days: int = 0

    # ----------------------------------------------------------- truck demand
    trucks_per_month: float = 40_000.0       # published order of magnitude (Section 4)
    hourly_profile: Tuple[float, ...] = (0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 1.0, 1.4, 1.4, 1.4, 1.3,
                                         1.1, 1.3, 1.4, 1.4, 1.3, 1.2, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4)
    window_minutes: int = 60                 # TAS time-window length
    window_quota: int = 70                   # appointments per time window

    # ------------------------------------------------- punctuality / tracking
    p_on_time: float = 0.72                  # published (Section 4)
    p_early: float = 0.14
    p_delayed: float = 0.14
    early_mean: float = 60.0                 # mean earliness before the window (illustrative)
    delay_mean: float = 50.0                 # mean lateness after the window (illustrative)
    tracking_lead_mean: float = 120.0        # time between the ETT reading and the arrival
    tracking_lead_sd: float = 45.0
    ett_noise_sd: float = 0.12               # relative error of the ETT
    contextual_eta_noise_sd: float = 0.12    # Table 5: relative error SD of the contextual ETA (e.g. call to the driver)
    n_training_records: int = 100_000        # synthetic historical records for the classifiers
    test_size: float = 0.30                  # stratified hold-out split (Section 3.3)

    # ------------------------------------------------------------------ gates
    n_gates_in: int = 4                      # eight reversible gates, 50% per direction (Section 4)
    n_gates_out: int = 4
    gate_service_mean: float = 3.3
    gate_service_sd: float = 1.0
    lsa_capacity: int = 200                  # Logistics Support Area (pre-gate buffer)
    admission_queue_limit: int = 5           # human-centric gate admission: queue <= 5 trucks (Section 3.1.4)

    # ------------------------------------------------------------------- yard
    n_yard_equipment: int = 20               # handling units serving road trucks (illustrative)
    internal_travel_mean: float = 15.0       # gate -> block -> gate (not using yard equipment)
    internal_travel_sd: float = 5.0
    handling_mean: float = 17.0              # loading/unloading per truck
    handling_sd: float = 5.0

    # ---------------------------------------------------- equipment condition
    initial_rul_h: Tuple[float, float] = (60.0, 400.0)   # uniform range of the initial RUL (operating hours)
    rul_after_maintenance_h: Tuple[float, float] = (300.0, 500.0)
    efficiency_decay_per_day: float = 0.01   # loss of efficiency per day since the last intervention
    min_efficiency: float = 0.70
    breakdown_rate_per_h: float = 1 / 2500   # random breakdowns (calendar hours)
    corrective_duration_h: Tuple[float, float] = (4.0, 12.0)
    monitor_interval: float = 15.0           # condition-monitoring interval
    rul_threshold_h: float = 24.0            # GA trigger (Table 7)
    efficiency_threshold: float = 0.80       # GA trigger by efficiency level (Section 3.2)
    preventive_duration: float = 120.0       # Table 7
    max_concurrent_maint: int = 1            # Table 7
    baseline_pm_interval_days: int = 15      # calendar-based preventive plan of the baseline
    baseline_pm_hour: int = 8

    # -------------------------------------------- Genetic Algorithm (Table 7)
    ga_pop_size: int = 24
    ga_generations: int = 10
    ga_cx_prob: float = 0.8
    ga_mut_prob: float = 0.2
    ga_mut_indpb: float = 0.2
    ga_mut_sigma: float = 60.0               # minutes
    ga_tournament: int = 3
    ga_jobs_per_unit: int = 2                # chromosome with 2|E| genes
    ga_horizon_h: float = 24.0               # planning horizon of each GA execution
    ga_penalty: float = 1000.0               # Equation 6, concurrency penalty
    ga_night_reward: float = 50.0            # Equation 6, night-time reward
    ga_night_hours: Tuple[int, int] = (0, 6) # interventions starting between 00:00 and 06:59

    # ---------------------------------------- HRA (Equations 1-5, Tables 2-4)
    hf_contributions: Dict[str, Tuple[float, ...]] = field(default_factory=lambda: {
        # contribution (%) of ERG, PER, LER, COG, ASP to each decision (Flores et al. 2026; Table 2)
        "maint": (27.9, 1.3, 0.4, 0.5, 0.8),
        "storage": (3.5, 0.1, 11.0, 8.0, 5.3),
        "sched": (5.9, 8.5, 14.8, 4.1, 2.9),
    })
    alpha: float = 0.5                       # scaling coefficient of Equation 1
    nhep: Dict[str, float] = field(default_factory=lambda: {"maint": 0.05, "storage": 0.03, "sched": 0.06})
    hep_multiplier: float = 1.0              # sensitivity analysis (Section 5.6, Table 14)

    # ------------------------- human decision logic (Section 3.1.4, Table 5)
    # shares of the deviation outcomes: (delay, suboptimal decision, improvement) - Table 5 (scenario assumptions)
    outcome_shares: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "maint": (0.60, 0.30, 0.10),
        "storage": (0.30, 0.60, 0.10),
        "sched": (0.50, 0.40, 0.10),
    })
    # response time added by a delay outcome: Normal(mean, sd) truncated at 0, minutes - Table 5
    response_time: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "maint": (240.0, 120.0),
        "storage": (5.0, 2.0),
        "sched": (5.0, 2.0),
    })
    maint_peak_hour: int = 10                # Table 5: suboptimal maintenance moved to 10:00 (day shift, high demand)
    sched_window_shift: int = 1              # suboptimal scheduling: adjacent time window
    improvement_handling_factor: float = 0.90    # Table 5: handling time -10%
    improvement_maint_factor: float = 0.90       # Table 5: intervention duration -10%

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
