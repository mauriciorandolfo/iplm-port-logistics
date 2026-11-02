"""Digital Twin of the container terminal (discrete-event simulation, SimPy) - Sections 3.2-3.4.

Flow of each truck (Figure 4): truck arrival-status prediction and rescheduling ->
Logistics Support Area (LSA) -> gate-in -> yard operations -> gate-out.
Equipment condition (RUL, efficiency decay, breakdowns) and maintenance are simulated in
parallel. The three decision nodes - truck scheduling, storage management and maintenance
programming - are executed according to the scenario:

* baseline      predefined rules (no rescheduling, first available equipment, calendar-based
                preventive maintenance), without the prediction module, decision support and GA;
* autonomous    recommendations of the IPLM executed directly;
* human_centric recommendations of the IPLM executed through the human decision layer.

Gate admission of trucks outside their time window (Sections 3.1.4 and 3.4): in the autonomous
scenario they are sent to the LSA and assigned to the next available window; in the
human-centric scenario the gate supervisor admits them when the gate queue does not exceed
`admission_queue_limit` trucks, otherwise they are also sent to the LSA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy

from . import ga_optimizer
from .config import Config, MIN_PER_DAY
from .data import STATUS_DELAYED, STATUS_EARLY
from .human import DELAY, IMPROVEMENT, SUBOPTIMAL, HumanDecisionLayer
from .ml_module import AdaptiveTAS

PREVENTIVE, CORRECTIVE = "preventive", "corrective"


@dataclass(eq=False)
class Equipment:
    name: str
    rul_h: float
    last_maintenance: float = 0.0
    busy: bool = False
    down: bool = False
    pending: Optional[Tuple[str, float]] = None      # maintenance waiting for the current job to end
    flagged: bool = False
    planned: List[Tuple[float, float]] = field(default_factory=list)

    def efficiency(self, now: float, cfg: Config) -> float:
        days = (now - self.last_maintenance) / MIN_PER_DAY
        return max(cfg.min_efficiency, 1.0 - cfg.efficiency_decay_per_day * days)


class DigitalTwin:
    def __init__(self, cfg: Config, scenario: str, trucks: pd.DataFrame, services: Dict[str, np.ndarray],
                 ens_pred: Optional[np.ndarray], human: HumanDecisionLayer, rng_eq: np.random.Generator,
                 rng_ops: np.random.Generator, seed: int):
        self.cfg, self.scenario, self.seed = cfg, scenario, seed
        self.smart = scenario in ("autonomous", "human_centric")
        self.env = simpy.Environment()
        self.tr = trucks
        self.sv = services
        self.pred = ens_pred
        self.human = human
        self.rng_eq, self.rng_ops = rng_eq, rng_ops
        self.gate_in = simpy.Resource(self.env, cfg.n_gates_in)
        self.gate_out = simpy.Resource(self.env, cfg.n_gates_out)
        self.lsa = simpy.Resource(self.env, cfg.lsa_capacity)
        self.yard = simpy.FilterStore(self.env)
        self.units: List[Equipment] = []
        for k in range(cfg.n_yard_equipment):
            u = Equipment(f"EQ{k + 1:02d}", rul_h=float(rng_eq.uniform(*cfg.initial_rul_h)))
            self.units.append(u)
            self.yard.items.append(u)
        n = len(trucks)
        self.booked = np.bincount(trucks["window"].to_numpy(), minlength=int(trucks["window"].max()) + 10)
        nan = lambda: np.full(n, np.nan)
        self.rec = {k: nan() for k in ("gate_arrival", "gate_start", "yard_request", "yard_start",
                                       "yard_end", "gateout_end")}
        self.rec["gate_wait"] = np.zeros(n)
        self.rec["lsa_wait"] = np.zeros(n)
        self.gate_queue_log: List[Tuple[float, float]] = []          # (joins the gate queue, admitted)
        self.final_window = trucks["window"].to_numpy().copy()
        self.rescheduled = np.zeros(n, dtype=bool)                   # rescheduled before arriving
        self.fallback = np.zeros(n, dtype=bool)                      # sent to the LSA at the gate
        self.downtime: List[Tuple[str, str, float, float]] = []      # (unit, kind, start, end)
        self.ga_log: List[Tuple[float, int, float, float]] = []      # (time, n units, fitness, cpu s)

    # ================================================================ trucks
    def truck_source(self):
        tr = self.tr
        start = np.minimum(tr["t_pred"].to_numpy(), tr["arrival"].to_numpy()) if self.smart else tr["arrival"].to_numpy()
        start = np.maximum(start, 0.0)
        for i in np.argsort(start, kind="stable"):
            if start[i] > self.env.now:
                yield self.env.timeout(start[i] - self.env.now)
            self.env.process(self.truck(int(i)))

    def schedule_decision(self, pred: int, window: int, t_pred: float, ett: float,
                          ett_ctx: float) -> Tuple[Optional[int], float]:
        """Truck-scheduling decision node, based on the information available at the ETT reading
        (booked window, time of the reading, ETT and, for the dispatcher, the contextual estimate).

        Returns (new window or None, time at which the instruction is issued).
        """
        cfg, env = self.cfg, self.env
        eat = t_pred + ett                                                   # estimated arrival time
        proposal = AdaptiveTAS.suggest_reschedule(pred, eat, window, cfg.window_minutes,
                                                  self.booked, cfg.window_quota)
        if proposal is None:
            return None, env.now
        t_eff = env.now
        outcome = self.human.review("sched")
        if outcome == DELAY:
            t_eff += self.human.response_time("sched")
        elif outcome == SUBOPTIMAL:
            proposal = max(0, proposal + int(self.rng_ops.choice([-1, 1])) * cfg.sched_window_shift)
        elif outcome == IMPROVEMENT:
            # the manager combines the ETT with the contextual ETA (average of two independent estimates)
            eat_mgr = t_pred + 0.5 * (ett + ett_ctx)
            w = int(eat_mgr // cfg.window_minutes)
            if w == window:
                return None, t_eff                                           # truck kept in its own window
            status = STATUS_EARLY if w < window else STATUS_DELAYED
            proposal = AdaptiveTAS.suggest_reschedule(status, eat_mgr, window,
                                                      cfg.window_minutes, self.booked, cfg.window_quota)
        return proposal, t_eff

    def truck(self, i: int):
        cfg, env, tr = self.cfg, self.env, self.tr
        arrival = float(tr.at[i, "arrival"])          # realised arrival time (physical process)
        # ---------------------------------------------------- truck scheduling
        if self.smart:
            proposal, t_eff = self.schedule_decision(int(self.pred[i]), int(tr.at[i, "window"]), float(tr.at[i, "t_pred"]),
                                                     float(tr.at[i, "ett_min"]), float(tr.at[i, "ett_contextual_min"]))
            # the instruction only takes effect if it reaches the driver before the truck arrives
            if proposal is not None and t_eff < arrival:
                self._move(i, proposal)
                self.rescheduled[i] = True
        if arrival > env.now:
            yield env.timeout(arrival - env.now)
        # ---------------------------------------------------------------- LSA
        if self.rescheduled[i]:
            yield from self._lsa(i)
        # ------------------------------------------------------------ gate-in
        self.rec["gate_arrival"][i] = env.now
        while True:
            t_q = env.now
            req = self.gate_in.request()
            yield req
            self.gate_queue_log.append((t_q, env.now))
            self.rec["gate_wait"][i] += env.now - t_q
            if not self.smart or self._admit(i):
                break
            # outside the window and not admitted: LSA and next available window
            self.gate_in.release(req)
            self.fallback[i] = True
            ws = self.final_window[i] * float(cfg.window_minutes)
            if env.now > ws + cfg.window_minutes:
                self._move(i, self._next_window(env.now))
            yield from self._lsa(i)
        self.rec["gate_start"][i] = env.now
        yield env.timeout(self.sv["gate_in"][i])
        self.gate_in.release(req)
        yield env.timeout(self.sv["travel"][i] / 2.0)
        # ------------------------------------------------- storage management
        self.rec["yard_request"][i] = env.now
        factor = 1.0
        if self.smart:
            rule = "best"                                                # recommendation: most efficient unit
            outcome = self.human.review("storage")
            if outcome == DELAY:
                yield env.timeout(self.human.response_time("storage"))
            elif outcome == SUBOPTIMAL:
                rule = "other"                                           # a unit different from the recommended one
            elif outcome == IMPROVEMENT:
                factor = cfg.improvement_handling_factor
        else:
            rule = "first_available"                                     # current practice
        unit = yield from self._get_unit(rule)
        self.rec["yard_start"][i] = env.now
        unit.busy = True
        handling = self.sv["handling"][i] * factor / unit.efficiency(env.now, cfg)
        yield env.timeout(handling)
        unit.rul_h -= handling / 60.0
        unit.busy = False
        self.rec["yard_end"][i] = env.now
        self._release_unit(unit)
        yield env.timeout(self.sv["travel"][i] / 2.0)
        # ----------------------------------------------------------- gate-out
        with self.gate_out.request() as req:
            yield req
            yield env.timeout(self.sv["gate_out"][i])
        self.rec["gateout_end"][i] = env.now

    def _move(self, i: int, window: int):
        self.booked[self.final_window[i]] -= 1
        self.booked[window] += 1
        self.final_window[i] = window

    def _next_window(self, now: float, max_ahead: int = 8) -> int:
        w = int(now // self.cfg.window_minutes) + 1
        for k in range(max_ahead + 1):
            if w + k < self.booked.size and self.booked[w + k] < self.cfg.window_quota:
                return w + k
        return min(w, self.booked.size - 1)

    def _lsa(self, i: int):
        """Wait in the LSA until the truck is called within its (new) time window."""
        env, cfg = self.env, self.cfg
        ws = self.final_window[i] * float(cfg.window_minutes)
        if env.now >= ws:
            return
        release = ws + self.rng_ops.random() * 0.8 * cfg.window_minutes      # trucks called progressively
        t0 = env.now
        with self.lsa.request() as req:
            yield req
            yield env.timeout(max(0.0, release - env.now))
        self.rec["lsa_wait"][i] += env.now - t0

    def _admit(self, i: int) -> bool:
        """Gate admission of a truck at the moment it reaches the gate booth."""
        ws = self.final_window[i] * float(self.cfg.window_minutes)
        if ws <= self.env.now <= ws + self.cfg.window_minutes:
            return True
        return self.scenario == "human_centric" and len(self.gate_in.queue) <= self.cfg.admission_queue_limit

    def _get_unit(self, rule: str):
        """Storage-management decision: selection of the yard unit that serves the truck.

        first_available - unit idle for the longest time (FIFO order of the idle units);
        best            - unit with the highest efficiency (IPLM recommendation);
        other           - a unit different from the recommended one, chosen at random (suboptimal decision).
        If no unit is idle, the truck waits for the first unit released.
        """
        items = self.yard.items
        if items:
            if rule == "first_available":
                target = items[0]
            else:
                best = max(items, key=lambda u: u.efficiency(self.env.now, self.cfg))
                if rule == "best" or len(items) == 1:
                    target = best
                else:
                    others = [u for u in items if u is not best]
                    target = others[int(self.rng_ops.integers(len(others)))]
            unit = yield self.yard.get(lambda u: u is target)
        else:
            unit = yield self.yard.get()
        return unit

    def _release_unit(self, unit: Equipment):
        if unit.pending is not None:
            kind, dur = unit.pending
            unit.pending = None
            self.env.process(self._maintain(unit, kind, dur, already_out=True))
        elif unit.rul_h <= 0:
            self.env.process(self._maintain(unit, CORRECTIVE, self._corrective_duration(), already_out=True))
        else:
            self.yard.put(unit)

    # =========================================================== maintenance
    def _corrective_duration(self) -> float:
        return float(self.rng_eq.uniform(*self.cfg.corrective_duration_h)) * 60.0

    def request_maintenance(self, unit: Equipment, kind: str, duration: float):
        if unit.down:
            return
        if unit in self.yard.items:
            self.yard.items.remove(unit)
            self.env.process(self._maintain(unit, kind, duration, already_out=True))
        elif unit.busy and (unit.pending is None or kind == CORRECTIVE):
            unit.pending = (kind, duration)

    def _maintain(self, unit: Equipment, kind: str, duration: float, already_out: bool):
        unit.down = True
        start = self.env.now
        yield self.env.timeout(duration)
        self.downtime.append((unit.name, kind, start, self.env.now))
        unit.rul_h = float(self.rng_eq.uniform(*self.cfg.rul_after_maintenance_h))
        unit.last_maintenance = self.env.now
        unit.down = False
        unit.flagged = False
        self.yard.put(unit)

    def breakdowns(self, unit: Equipment):
        rate = self.cfg.breakdown_rate_per_h / 60.0
        while True:
            yield self.env.timeout(float(self.rng_eq.exponential(1.0 / rate)))
            self.request_maintenance(unit, CORRECTIVE, self._corrective_duration())

    def planned_job(self, unit: Equipment, start: float, duration: float):
        if start > self.env.now:
            yield self.env.timeout(start - self.env.now)
        self.request_maintenance(unit, PREVENTIVE, duration)

    def baseline_plan(self, unit: Equipment, k: int):
        cfg = self.cfg
        t = ((k % cfg.baseline_pm_interval_days) * MIN_PER_DAY) + cfg.baseline_pm_hour * 60.0
        while True:
            if t > self.env.now:
                yield self.env.timeout(t - self.env.now)
            self.request_maintenance(unit, PREVENTIVE, cfg.preventive_duration)
            t += cfg.baseline_pm_interval_days * MIN_PER_DAY

    def condition_monitoring(self):
        """CBM trigger: units whose RUL or efficiency falls below the threshold are scheduled by the GA."""
        cfg = self.cfg
        while True:
            yield self.env.timeout(cfg.monitor_interval)
            now = self.env.now
            flagged = [u for u in self.units if not u.flagged and not u.down
                       and (u.rul_h <= cfg.rul_threshold_h or u.efficiency(now, cfg) <= cfg.efficiency_threshold)]
            if not flagged:
                continue
            planned = [iv for u in self.units for iv in u.planned if iv[1] > now]
            plan, fit, cpu = ga_optimizer.optimize(cfg, [u.name for u in flagged], now, planned,
                                                   seed=self.seed + len(self.ga_log))
            self.ga_log.append((now, len(flagged), fit, cpu))
            for u in flagged:
                u.flagged = True
                u.planned = []
                for s, e in plan[u.name]:
                    duration = e - s
                    outcome = self.human.review("maint")                    # maintenance programming
                    if outcome == DELAY:
                        s += self.human.response_time("maint")
                    elif outcome == SUBOPTIMAL:
                        day = np.floor(s / MIN_PER_DAY) * MIN_PER_DAY
                        peak = day + cfg.maint_peak_hour * 60.0
                        s = peak if peak >= s else peak + MIN_PER_DAY
                    elif outcome == IMPROVEMENT:
                        duration *= cfg.improvement_maint_factor
                    u.planned.append((s, s + duration))
                    self.env.process(self.planned_job(u, s, duration))

    # ================================================================== run
    def run(self, until: float):
        self.env.process(self.truck_source())
        for k, u in enumerate(self.units):
            self.env.process(self.breakdowns(u))
            if not self.smart:
                self.env.process(self.baseline_plan(u, k))
        if self.smart:
            self.env.process(self.condition_monitoring())
        self.env.run(until=until)
