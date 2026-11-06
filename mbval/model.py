"""LightGBM on index-adjusted log prices.

Early stopping uses the latest 10% of training sales (a mini time split), then
the model is refit on everything with that number of rounds plus 10%.
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
        self.booster = lgb.train(params, lgb.Dataset(x, y, weight=w), num_boost_round=self.rounds)
        self.features = list(x.columns)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(x[self.features])

    def importance(self, top: int = 20) -> list[tuple[str, float]]:
        gain = self.booster.feature_importance(importance_type="gain")
        share = gain / gain.sum() if gain.sum() else gain
        ranked = sorted(zip(self.features, share), key=lambda t: -t[1])[:top]
        return [(name, round(float(s), 4)) for name, s in ranked]
