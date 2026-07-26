"""
romacs_tuning.py — hyperparameter tuning + retuned robustness sweep.

Addresses the fairness concern that fixed default hyperparameters could confound the
tree-vs-non-tree comparison. For every learner (tree AND non-tree) we run a random
search over a per-model grid, selecting hyperparameters by macro-F1 on a
trajectory-grouped validation split carved from the TRAINING data only (no test
leakage). Tuning is done once (seed 0) on missingness-aware training data; the chosen
configs are then fixed and evaluated across the full missingness sweep over all seeds.
"""
import numpy as np
import pandas as pd
from itertools import product

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from imblearn.ensemble import RUSBoostClassifier
from xgboost import XGBClassifier

from romacs_datagen import (
    GenConfig, generate_dataset, inject_mcar_missingness,
    add_missingness_indicators, train_test_split_by_trajectory,
)
from romacs_experiment import indicator_feature_cols, ALL_LABELS

FEATS = indicator_feature_cols()
RNG = np.random.default_rng(0)


# ----- per-model search spaces (random search samples from these) -----
def sample_configs(rng, n=12):
    grids = {
        "DecisionTree": {"max_depth": [8, 12, 16, None], "min_samples_leaf": [1, 3, 5, 10]},
        "Bagging": {"n_estimators": [15, 25, 40], "max_samples": [0.7, 0.85, 1.0]},
        "AdaBoost": {"n_estimators": [30, 50, 80], "learning_rate": [0.5, 1.0],
                     "estimator_depth": [2, 3, 4]},
        "RUSBoost": {"n_estimators": [30, 50, 80], "learning_rate": [0.5, 1.0],
                     "estimator_depth": [2, 3, 4]},
        "XGBoost": {"n_estimators": [100, 200], "max_depth": [4, 6, 8],
                    "learning_rate": [0.1, 0.3], "subsample": [0.8, 1.0]},
        "LogReg": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "MLP": {"hidden": [(64,), (128, 64), (256, 128)],
                "lr": [1e-3, 1e-2], "alpha": [1e-4, 1e-3, 1e-2],
                "activation": ["relu", "tanh"]},
    }
    out = {}
    for name, grid in grids.items():
        keys = list(grid)
        combos = list(product(*[grid[k] for k in keys]))
        rng.shuffle(combos)
        out[name] = [dict(zip(keys, c)) for c in combos[:n]]
    return out


def build(name, cfg, seed):
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed, **cfg)
    if name == "Bagging":
        return BaggingClassifier(random_state=seed, n_jobs=-1, **cfg)
    if name == "AdaBoost":
        d = cfg.copy(); dep = d.pop("estimator_depth")
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=dep),
                                  random_state=seed, **d)
    if name == "RUSBoost":
        d = cfg.copy(); dep = d.pop("estimator_depth")
        return RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=dep),
                                  random_state=seed, **d)
    if name == "XGBoost":
        return XGBClassifier(tree_method="hist", objective="multi:softprob",
                             num_class=len(ALL_LABELS), random_state=seed,
                             verbosity=0, **cfg)
    if name == "LogReg":
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=1000, C=cfg["C"]))])
    if name == "MLP":
        return Pipeline([("s", StandardScaler()),
                         ("c", MLPClassifier(hidden_layer_sizes=cfg["hidden"],
                                             learning_rate_init=cfg["lr"], alpha=cfg["alpha"],
                                             activation=cfg["activation"], max_iter=300,
                                             early_stopping=True, random_state=seed))])


def tune(seed=0, train_p=0.25):
    """Random-search each model on a trajectory-grouped train/val split; return best cfgs."""
    df = generate_dataset(GenConfig(n_trajectories=150, seed=seed))
    tr, _ = train_test_split_by_trajectory(df, test_size=0.30, seed=seed)
    # inner trajectory-grouped validation split (from TRAIN only)
    tr_in, val = train_test_split_by_trajectory(tr, test_size=0.25, seed=seed + 5)
    tr_in_m = inject_mcar_missingness(tr_in, train_p, seed + 1000)
    val_m = inject_mcar_missingness(val, train_p, seed + 2000)
    tr_feat, fills = add_missingness_indicators(tr_in_m)
    val_feat, _ = add_missingness_indicators(val_m, impute_values=fills)
    Xtr, ytr = tr_feat[FEATS].to_numpy(float), tr_feat["label"].to_numpy(int)
    Xv, yv = val_feat[FEATS].to_numpy(float), val_feat["label"].to_numpy(int)

    configs = sample_configs(RNG, n=12)
    best = {}
    for name, cand in configs.items():
        scored = []
        for cfg in cand:
            try:
                clf = build(name, cfg, seed)
                clf.fit(Xtr, ytr)
                f1 = f1_score(yv, clf.predict(Xv), labels=ALL_LABELS,
                              average="macro", zero_division=0)
                scored.append((f1, cfg))
            except Exception as e:
                continue
        scored.sort(key=lambda t: t[0], reverse=True)
        best[name] = {"cfg": scored[0][1], "val_f1": round(scored[0][0], 4),
                      "n_tried": len(scored)}
    return best


if __name__ == "__main__":
    best = tune()
    print("Tuned hyperparameters (selected on trajectory-grouped validation macro-F1):\n")
    for name, info in best.items():
        print(f"  {name:13s} valF1={info['val_f1']}  (tried {info['n_tried']})  ->  {info['cfg']}")
    import json
    with open("../results/tuned_configs.json", "w") as f:
        json.dump({k: v["cfg"] for k, v in best.items()}, f, indent=2)
    print("\nsaved -> results/tuned_configs.json")
