"""Truck arrival-status prediction module (supervised Machine Learning) - Section 3.3 and Appendix A.2.

Four algorithms are executed in parallel - Decision Tree (DT), Random Forest (RF),
Gradient Boosting Machine (GBM) and Linear Regression (LR) - and the ensemble follows
a dynamic "winner-takes-all" selection: at each prediction, the output of the model
with the best accumulated accuracy on the outcomes already observed is used.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

from .data import FEATURES, STATUS_DELAYED, STATUS_EARLY, STATUS_ON_TIME

MODEL_NAMES = ("DT", "RF", "GBM", "LR")
# ordinal coding used by the Linear Regression: early (-1) < on-time (0) < delayed (+1)
_TO_ORD = {STATUS_EARLY: -1.0, STATUS_ON_TIME: 0.0, STATUS_DELAYED: 1.0}


class LinearRegressionClassifier:
    """Linear Regression on the ordinal status code; the prediction is rounded to the nearest class."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, np.vectorize(_TO_ORD.get)(y))
        return self

    def predict(self, X):
        z = np.clip(np.rint(self.model.predict(X)), -1, 1)
        return np.where(z < 0, STATUS_EARLY, np.where(z > 0, STATUS_DELAYED, STATUS_ON_TIME))


class AdaptiveTAS:
    def __init__(self, seed: int = 0):
        self.models = {
            "DT": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=seed),
            "RF": RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=10, n_jobs=-1, random_state=seed),
            "GBM": HistGradientBoostingClassifier(max_iter=150, random_state=seed),
            "LR": LinearRegressionClassifier(),
        }

    def fit(self, df: pd.DataFrame) -> "AdaptiveTAS":
        X, y = df[FEATURES].to_numpy(), df["truck_status"].to_numpy()
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict_all(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].to_numpy()
        return np.vstack([self.models[k].predict(X) for k in MODEL_NAMES]).astype(int)

    @staticmethod
    def dynamic_selection(preds: np.ndarray, y_true: np.ndarray, t_pred: np.ndarray,
                          t_observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction: model with the highest accumulated accuracy at the moment of each prediction.

        Only outcomes already observed (trucks arrived before t_pred) are used, so that no future
        information is used. Returns (ensemble prediction, index of the selected model).
        """
        n_models, n = preds.shape
        correct = (preds == y_true[None, :])
        order_p = np.argsort(t_pred, kind="stable")
        order_o = np.argsort(t_observed, kind="stable")
        hits = np.zeros(n_models)
        sel = np.zeros(n, dtype=int)
        k = 0
        for j in order_p:
            while k < n and t_observed[order_o[k]] <= t_pred[j]:
                hits += correct[:, order_o[k]]
                k += 1
            sel[j] = int(np.argmax(hits))
        return preds[sel, np.arange(n)], sel

    @staticmethod
    def suggest_reschedule(pred_status: int, predicted_arrival: float, own_window: int, window_minutes: int,
                           booked: np.ndarray, quota: int, max_ahead: int = 8):
        """Next compatible time window with available quota (None if the truck is predicted on time)."""
        if pred_status == STATUS_ON_TIME:
            return None
        w = max(0, int(predicted_arrival // window_minutes))
        if pred_status == STATUS_EARLY and w >= own_window:
            return own_window                     # keep the window; the truck waits in the LSA
        for k in range(max_ahead + 1):
            ww = w + k
            if ww < booked.size and booked[ww] < quota:
                return ww
        return min(w, booked.size - 1)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Accuracy and class-wise precision, recall and F1-score (Tables 11 and 12)."""
    labels = [STATUS_ON_TIME, STATUS_EARLY, STATUS_DELAYED]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    _, _, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "macro_f1": fm, "support": s, "confusion": confusion_matrix(y_true, y_pred, labels=labels)}
