"""Human decision layer (Section 3.1.4, Figure 2, Table 5).

At each decision event of type d (maintenance programming, storage management or truck
scheduling), the recommendation of the IPLM is reviewed by the responsible manager in the
human-centric scenario. A Bernoulli trial with probability HEP_d (Equation 5) defines
whether the manager deviates from the recommendation; if so, the outcome is sampled among
(i) delay, (ii) suboptimal decision and (iii) improvement, with the shares of Table 5.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import Config, DECISIONS

DELAY, SUBOPTIMAL, IMPROVEMENT = "delay", "suboptimal", "improvement"
OUTCOMES = (DELAY, SUBOPTIMAL, IMPROVEMENT)


class HumanDecisionLayer:
    def __init__(self, cfg: Config, hep: Dict[str, float], rng: np.random.Generator, active: bool):
        self.cfg, self.hep, self.rng, self.active = cfg, hep, rng, active
        self.log = {d: {"events": 0, DELAY: 0, SUBOPTIMAL: 0, IMPROVEMENT: 0} for d in DECISIONS}

    def review(self, decision: str) -> Optional[str]:
        """Returns None when the recommendation is executed as proposed, otherwise the deviation outcome."""
        if not self.active:
            return None
        self.log[decision]["events"] += 1
        if self.rng.random() >= self.hep[decision]:
            return None
        outcome = OUTCOMES[int(self.rng.choice(3, p=self.cfg.outcome_shares[decision]))]
        self.log[decision][outcome] += 1
        return outcome

    def response_time(self, decision: str) -> float:
        mean, sd = self.cfg.response_time[decision]
        return max(0.0, float(self.rng.normal(mean, sd)))
