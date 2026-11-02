"""Human Reliability Analysis (THERP-based) - Section 3.1 of the paper.

Equation 1:  M_{i,d} = 1 + alpha * C_{i,d} / C_max
Equation 5:  HEP_d   = NHEP_d * prod_i M_{i,d}
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import Config, DECISIONS, HF_FACTORS


def psf_multipliers(cfg: Config) -> Dict[str, np.ndarray]:
    """PSF multipliers M_{i,d} (Equation 1, Table 3)."""
    cmax = max(max(v) for v in cfg.hf_contributions.values())
    return {d: 1.0 + cfg.alpha * np.asarray(cfg.hf_contributions[d], float) / cmax for d in DECISIONS}


def human_error_probabilities(cfg: Config) -> Dict[str, float]:
    """HEP of each decision (Equation 5, Table 4), scaled by the sensitivity multiplier."""
    m = psf_multipliers(cfg)
    return {d: float(min(1.0, cfg.hep_multiplier * cfg.nhep[d] * np.prod(m[d]))) for d in DECISIONS}


def hra_tables(cfg: Config) -> str:
    """Text version of Tables 3, 4 and 9 (multipliers, HEPs and alternative HRA parameters)."""
    m = psf_multipliers(cfg)
    hep = human_error_probabilities(cfg.with_(hep_multiplier=1.0))
    lines = ["PSF multipliers (Equation 1)", "factor   " + "  ".join(f"{d:>8s}" for d in DECISIONS)]
    for i, f in enumerate(HF_FACTORS):
        lines.append(f"{f:8s} " + "  ".join(f"{m[d][i]:8.3f}" for d in DECISIONS))
    lines.append("product  " + "  ".join(f"{np.prod(m[d]):8.4f}" for d in DECISIONS))
    lines += ["", "Human error probabilities (Equation 5)"]
    lines += [f"HEP_{d:8s} = {cfg.nhep[d]:.2f} x {np.prod(m[d]):.4f} = {hep[d]:.4f}" for d in DECISIONS]
    lines += ["", "Alternative HRA parameters (sensitivity analysis)"]
    variants = [("alpha=0.25", dict(alpha=0.25)), ("alpha=0.50", {}), ("alpha=0.75", dict(alpha=0.75)),
                ("alpha=1.00", dict(alpha=1.0)),
                ("NHEP -50%", dict(nhep={k: 0.5 * v for k, v in cfg.nhep.items()})),
                ("NHEP +50%", dict(nhep={k: 1.5 * v for k, v in cfg.nhep.items()}))]
    for name, kw in variants:
        h = human_error_probabilities(cfg.with_(hep_multiplier=1.0, **kw))
        ratio = [h[d] / hep[d] for d in DECISIONS]
        lines.append(f"{name:11s} " + "  ".join(f"{h[d]:.4f}" for d in DECISIONS)
                     + f"   multiplier {min(ratio):.2f}-{max(ratio):.2f}")
    return "\n".join(lines)
