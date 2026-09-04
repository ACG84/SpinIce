"""Linear readout: ridge regression on the spectral fingerprints.

Following Gartside et al. only the outputs of the *current* time step are
used (no time-multiplexing, no software memory): all memory has to come from
the magnetic reservoir.  A "raw input" baseline, i.e. the same regression on
the scalar input alone, is reported alongside so that the contribution of
the physical reservoir can be judged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeReadout:
    alpha: float = 1e-3
    standardise: bool = True
    fit_intercept: bool = True
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    w_: np.ndarray | None = None
    b_: float = 0.0

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        if self.standardise:
            X = (X - self.mean_) / self.std_
        return X

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0) if self.standardise else np.zeros(X.shape[1])
        if self.standardise:
            std = X.std(axis=0)
            # features that are constant on the training set (discrete microstates!) must not be
            # blown up by a tiny divisor when they change on unseen data: leave them unscaled
            floor = 1e-3 * max(float(np.median(std[std > 0])) if np.any(std > 0) else 1.0, 1e-12)
            self.std_ = np.where(std > floor, std, 1.0)
        else:
            self.std_ = np.ones(X.shape[1])
        Xs = self._prep(X)
        if self.fit_intercept:
            Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
        n_feat = Xs.shape[1]
        reg = self.alpha * np.eye(n_feat)
        if self.fit_intercept:
            reg[-1, -1] = 0.0
        w = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)
        if self.fit_intercept:
            self.w_, self.b_ = w[:-1], float(w[-1])
        else:
            self.w_, self.b_ = w, 0.0
        return self

    def predict(self, X):
        return self._prep(X) @ self.w_ + self.b_


def mse(y, yhat) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def nrmse(y, yhat) -> float:
    y = np.asarray(y)
    return float(np.sqrt(mse(y, yhat) / (np.var(y) + 1e-300)))


def split(n: int, n_train: int, washout: int = 0):
    """Sequential split indices (train, test) after discarding ``washout`` steps."""
    tr = np.arange(washout, washout + n_train)
    te = np.arange(washout + n_train, n)
    if len(te) == 0:
        raise ValueError("no test data: reduce n_train/washout")
    return tr, te


def evaluate(X, y, n_train: int, alpha: float = 1e-3, washout: int = 0, u=None) -> dict:
    """Train on the first n_train steps (after washout) and test on the rest.

    Returns predictions and metrics for the reservoir and, if ``u`` is given,
    for the raw-input baseline (regression on [u, u^2, u^3] of the same step).
    """
    tr, te = split(len(y), n_train, washout)
    model = RidgeReadout(alpha=alpha).fit(X[tr], y[tr])
    yhat = model.predict(X)
    out = {
        "train_idx": tr, "test_idx": te, "y_pred": yhat, "model": model,
        "mse_train": mse(y[tr], yhat[tr]), "mse_test": mse(y[te], yhat[te]),
        "nrmse_train": nrmse(y[tr], yhat[tr]), "nrmse_test": nrmse(y[te], yhat[te]),
    }
    if u is not None:
        U = np.stack([u, u ** 2, u ** 3], axis=1)
        base = RidgeReadout(alpha=alpha).fit(U[tr], y[tr])
        yb = base.predict(U)
        out.update({"y_baseline": yb, "mse_test_baseline": mse(y[te], yb[te]),
                    "nrmse_test_baseline": nrmse(y[te], yb[te])})
    return out


def select_alpha(X, y, n_train: int, alphas=None, washout: int = 0, n_val: int | None = None) -> float:
    """Pick the ridge parameter on a validation slice carved from the training data."""
    alphas = np.logspace(-6, 2, 17) if alphas is None else alphas
    n_val = n_val or max(10, n_train // 5)
    tr, _ = split(len(y), n_train, washout)
    fit_idx, val_idx = tr[:-n_val], tr[-n_val:]
    best, best_err = alphas[0], np.inf
    for a in alphas:
        m = RidgeReadout(alpha=a).fit(X[fit_idx], y[fit_idx])
        err = mse(y[val_idx], m.predict(X[val_idx]))
        if err < best_err:
            best, best_err = a, err
    return float(best)


def memory_capacity(X, u, n_train: int, max_delay: int = 20, alpha: float = 1e-3, washout: int = 0):
    """Linear short-term memory capacity: sum_k R^2 of reconstructing u(t-k)."""
    caps = []
    for k in range(1, max_delay + 1):
        y = np.roll(u, k)
        tr, te = split(len(u), n_train, max(washout, k))
        m = RidgeReadout(alpha=alpha).fit(X[tr], y[tr])
        yhat = m.predict(X[te])
        c = np.corrcoef(y[te], yhat)[0, 1] ** 2 if np.std(yhat) > 0 else 0.0
        caps.append(float(c))
    return np.array(caps), float(np.sum(caps))
