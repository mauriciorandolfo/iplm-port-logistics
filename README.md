# IPLM – Intelligent Port Logistics Model

Source code of the simulation-driven decision-support model described in:

> Flores, M. R. et al. *Human-centric decision-making reliability in intelligent port logistics: a
> simulation-based assessment*. Flexible Services and Manufacturing Journal (under review).

The IPLM integrates a discrete-event **Digital Twin** of a container terminal (SimPy) with three decision
modules – **truck arrival-status prediction** (supervised Machine Learning), **storage management** and
**maintenance programming** (Genetic Algorithm, DEAP) – and a **human decision layer** parameterised through
**Human Reliability Analysis** (THERP). It compares three scenarios: **baseline** (current practice),
**autonomous** and **human-centric** decision-making.

## Data confidentiality – illustrative parameters

The operational records of the terminal and the input distributions fitted to them are confidential.
All numerical values in `iplm/config.py` are **illustrative**; they allow the model to be executed and
inspected, but **do not reproduce the results reported in the paper**. The only values taken from the paper
are those published in it (human-factor contributions, α, nominal HEPs, GA parameters, human decision logic of
Table 5, gate admission limit, punctuality shares and monthly truck volume). Input data are generated synthetically by `iplm/data.py`.

## Correspondence between the code and the paper

| Paper | Code |
|---|---|
| Section 3.1, Equations 1–5, Tables 2–4 (HRA / THERP) | `iplm/hra.py` |
| Section 3.1.4, Table 5 (human decision logic) | `iplm/human.py`, `Config.outcome_shares`, `Config.response_time` |
| Section 3.3, Figure 3, Appendix A.2 (DT, RF, GBM, LR; dynamic selection) | `iplm/ml_module.py` |
| Sections 3.1.4 and 3.4, Figure 4 (Digital Twin: LSA, gate admission, yard, equipment) | `iplm/digital_twin.py` |
| Section 3.5, Figure 5, Equation 6, Table 7, Appendix A.1 (Genetic Algorithm) | `iplm/ga_optimizer.py` |
| Table 8 (data structure) | `iplm/data.py` (synthetic) |
| Section 4 (replications, KPIs), Tables 10, 13, 14 and 15 | `iplm/experiment.py`, `iplm/kpi_tracker.py` |
| Section 3.4 (verification) | `tests/test_verification.py` |

### Decision nodes in each scenario

| Decision | Baseline | Autonomous | Human-centric |
|---|---|---|---|
| Truck scheduling | no rescheduling | prediction module → new time window / LSA | same, reviewed by the dispatcher |
| Gate admission outside the window | admitted on arrival | LSA and next available window | admitted if the gate queue ≤ 5 trucks, otherwise LSA |
| Storage management | first available unit (the one idle for the longest time, FIFO) | unit with the highest efficiency | same, reviewed by the yard planner |
| Maintenance programming | calendar-based preventive plan + corrective | CBM trigger (RUL or efficiency) + GA | same, reviewed by the maintenance planner |

In the human-centric scenario, a Bernoulli trial with probability HEP_d defines whether the manager deviates
from each recommendation; deviations are classified as **delay**, **suboptimal decision** or **improvement**
(`iplm/human.py`, Table 5). In truck scheduling, an improvement corresponds to the combination of the ETT with an
additional contextual estimate of the travel time (`ett_contextual_min`), available to the dispatcher.

**Genetic Algorithm.** Candidate plans are evaluated with the fitness of Equation 6, computed from the state
provided by the Digital Twin (planned interventions, units to be maintained, time of day); they are not simulated
individually. The selected plan is executed in the Digital Twin. Genes are integer start times (minutes) within the
planning horizon; after Gaussian mutation they are rounded and limited to the horizon.

**KPIs.** `pct_arriving_outside_time_window` is the share of trucks arriving outside their booked window (input
of the model), and `pct_admitted_outside_time_window` the share admitted at the gate outside their booked or
rescheduled window, which also includes trucks admitted after the end of their window because of gate queues.
The gate waiting time excludes the time spent in the LSA.

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10.

## Usage

```bash
python main.py hra                       # HRA multipliers, HEPs and alternative parameters (Tables 3, 4, 9)
python main.py ml                        # accuracy and class-wise metrics of the classifiers (Tables 11-12)
python main.py pilot                     # pilot sample and required number of replications (Section 4)
python main.py run --reps 25             # baseline, autonomous and human-centric (Table 13)
python main.py sensitivity --reps 25     # HEP multipliers 0.5, 1, 1.5 and 2 (Table 14)
python main.py sensitivity-oat --reps 25 # one decision at a time, HEP x0.5 and x2 (Table 15)
python main.py run --days 30 --reps 2    # quick test
python -m pytest tests                   # verification tests
```

Results are written to `outputs/` (one CSV with all replications and one summary with means and 95%
confidence half-widths). One year of one scenario takes about 1–2 min on a standard laptop.

## Structure

```
iplm/
  config.py         parameters (illustrative) and scenarios
  hra.py            Human Reliability Analysis (Equations 1-5)
  human.py          human decision layer (delay / suboptimal / improvement)
  data.py           synthetic input data (structure of Table 8)
  ml_module.py      truck arrival-status prediction (DT, RF, GBM, LR) and dynamic selection
  ga_optimizer.py   Genetic Algorithm for maintenance programming (Equation 6)
  digital_twin.py   discrete-event simulation of the terminal (SimPy)
  kpi_tracker.py    key performance indicators
  experiment.py     replications, summaries, sensitivity analysis, classifier evaluation
main.py             command-line interface
tests/              verification tests
```

## License

MIT (see `LICENSE`). Please cite the paper when using this code.
