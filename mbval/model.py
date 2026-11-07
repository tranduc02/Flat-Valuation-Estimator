"""LightGBM on index-adjusted log prices.

Early stopping uses the latest 10% of training sales (a mini time split), then the model is
refit on everything with that number of rounds plus 10%. With N_MODELS > 1, that refit is done
N times with different random seeds (same rounds) and the predictions are averaged, which
smooths out run-to-run noise.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import config


class Model:
    def fit(self, x: pd.DataFrame, y: np.ndarray, w: np.ndarray, dates: pd.Series):
        order = np.argsort(dates.to_numpy(), kind="stable")
        n_valid = max(len(x) // 10, 1)
        tr, va = order[:-n_valid], order[-n_valid:]
        params = {**config.LGBM_PARAMS, "seed": config.SEED}

        dtrain = lgb.Dataset(x.iloc[tr], y[tr], weight=w[tr])
        dvalid = lgb.Dataset(x.iloc[va], y[va], weight=w[va], reference=dtrain)
        probe = lgb.train(params, dtrain, num_boost_round=config.MAX_ROUNDS, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])
        self.best_rounds = probe.best_iteration or config.MAX_ROUNDS
        self.rounds = int(self.best_rounds * 1.1) + 1
        full = lgb.Dataset(x, y, weight=w, free_raw_data=False)
        self.boosters = [lgb.train({**params, "seed": config.SEED + i}, full, num_boost_round=self.rounds)
                         for i in range(max(config.N_MODELS, 1))]
        self.booster = self.boosters[0]
        self.features = list(x.columns)
        return self

    def predict(self, x: pd.DataFrame, members: int | None = None) -> np.ndarray:
        """Average log-price prediction of the first `members` models (default: all)."""
        use = self.boosters[:members] if members else self.boosters
        return np.mean([b.predict(x[self.features]) for b in use], axis=0)

    def importance(self, top: int = 20) -> list[tuple[str, float]]:
        gain = np.mean([b.feature_importance(importance_type="gain") for b in self.boosters], axis=0)
        share = gain / gain.sum() if gain.sum() else gain
        ranked = sorted(zip(self.features, share), key=lambda t: -t[1])[:top]
        return [(name, round(float(s), 4)) for name, s in ranked]
