"""Maintenance programming with a Genetic Algorithm (DEAP) - Section 3.5, Equation 6, Table 7, Appendix A.1.

Chromosome: integer start times (minutes from the GA execution, within the planning horizon)
of up to two maintenance jobs per critical equipment unit (2|E| genes).

Fitness (minimisation, Equation 6):   F = P_concurrency - R_night

* P_concurrency - 1,000 for each intervention in excess of the maximum number of simultaneous
  interventions, computed over the planning horizon and including the interventions already
  planned; overlapping interventions on the same unit are penalised with the same coefficient.
* R_night - 50 for each intervention starting between 00:00 and 06:59.

The fitness is computed from the state provided by the Digital Twin; candidate plans are not
simulated individually. The selected plan is executed in the Digital Twin.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence, Tuple

from deap import base, creator, tools

from .config import Config, MIN_PER_DAY

if not hasattr(creator, "FitnessMinIPLM"):
    creator.create("FitnessMinIPLM", base.Fitness, weights=(-1.0,))
    creator.create("IndividualIPLM", list, fitness=creator.FitnessMinIPLM)


def max_concurrency(jobs: Sequence[Tuple[float, float]], t0: float, t1: float) -> int:
    """Maximum number of intervals [s, e) active at the same time within [t0, t1) (interval sweep)."""
    events = []
    for s, e in jobs:
        s, e = max(s, t0), min(e, t1)
        if e > s:
            events.append((s, 1))
            events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))          # an interval ending at t does not overlap one starting at t
    level = best = 0
    for _, d in events:
        level += d
        best = max(best, level)
    return best


def fitness(starts_abs: Sequence[float], job_units: Sequence[str], cfg: Config, now: float,
            planned: Sequence[Tuple[float, float]]) -> float:
    d = cfg.preventive_duration
    jobs = [(s, s + d) for s in starts_abs]
    horizon_end = now + cfg.ga_horizon_h * 60.0
    excess = max(0, max_concurrency(jobs + list(planned), now, horizon_end) - cfg.max_concurrent_maint)
    penalty = excess * cfg.ga_penalty
    for u in set(job_units):                          # overlapping interventions on the same unit
        mine = sorted(j for j, ju in zip(jobs, job_units) if ju == u)
        penalty += cfg.ga_penalty * sum(1 for a, b in zip(mine, mine[1:]) if b[0] < a[1])
    lo, hi = cfg.ga_night_hours
    reward = sum(cfg.ga_night_reward for s, _ in jobs if lo <= int((s % MIN_PER_DAY) // 60) <= hi)
    return penalty - reward


def optimize(cfg: Config, units: List[str], now: float, planned: Sequence[Tuple[float, float]],
             seed: int) -> Tuple[Dict[str, List[Tuple[float, float]]], float, float]:
    """Returns (plan per unit, best fitness, CPU time in seconds)."""
    t0 = time.perf_counter()
    rnd = random.Random(seed)
    hi = int(max(0.0, cfg.ga_horizon_h * 60.0 - cfg.preventive_duration))
    job_units = [u for u in units for _ in range(cfg.ga_jobs_per_unit)]
    n_genes = len(job_units)

    def repair(ind):
        ind[:] = [int(min(max(round(g), 0), hi)) for g in ind]
        return ind

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.IndividualIPLM([rnd.randint(0, hi) for _ in range(n_genes)]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (fitness([now + g for g in ind], job_units, cfg, now, planned),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=cfg.ga_mut_sigma, indpb=cfg.ga_mut_indpb)
    toolbox.register("select", tools.selTournament, tournsize=cfg.ga_tournament)

    state = random.getstate()
    random.seed(seed)                      # DEAP operators use the global random module
    pop = toolbox.population(n=cfg.ga_pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    for _ in range(cfg.ga_generations):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if n_genes > 1 and rnd.random() < cfg.ga_cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for ind in offspring:
            if rnd.random() < cfg.ga_mut_prob:
                toolbox.mutate(ind)
                repair(ind)                # integer genes within the planning horizon
                del ind.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = tools.selBest(offspring + pop, k=len(pop))   # elitist replacement
    random.setstate(state)

    best = tools.selBest(pop, 1)[0]
    plan: Dict[str, List[Tuple[float, float]]] = {u: [] for u in units}
    for g, u in zip(best, job_units):
        s = now + g
        plan[u].append((s, s + cfg.preventive_duration))
    for u in units:
        plan[u].sort()
    return plan, best.fitness.values[0], time.perf_counter() - t0
