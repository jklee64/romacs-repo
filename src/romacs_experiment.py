"""
romacs_experiment.py
====================

RoMaCS (Robust Maritime Channel Selection) — experiment runner.

Given the trajectory benchmark produced by romacs_datagen.py, this script runs the
full evaluation described in the paper (Sections V–VI):

  (1) Multi-seed training of five tree-based ensemble classifiers
        - Decision Tree
        - Bagging
        - AdaBoost (SAMME multiclass; the practical analog of AdaBoost.M2)
        - RUSBoost (imbalanced-learn)
        - XGBoost
      All five receive the SAME missingness-aware, 52-dimensional representation
      (20 imputed QoS values + 20 missingness indicators + 7 scenario + 5 availability)
      so the comparison is apples-to-apples.

  (2) Robustness sweep: each trained model is evaluated on test sets with MCAR QoS
      missingness at rates p in {0, 0.10, 0.25, 0.50, 0.75}, tracing a degradation curve.

  (3) Policy baseline: a rule-based selector that assumes complete information but,
      at inference, sees only incomplete QoS filled by LOCF (cold-start channels are
      treated as non-selectable). This is the "what heuristics do today" comparator.

  (4) Metrics: macro-F1, balanced accuracy, per-class F1 (incl. minority classes
      LTE_5G and NO_CHANNEL), macro-F1 restricted to EMERGENCY-priority traffic
      (safety-critical), retention F1(p)/F1(0), and a single-number robustness
      summary AUDC (area under the retention-vs-p curve).

  (5) Statistics: Wilcoxon signed-rank tests across seeds — (i) between models at
      p=0, and (ii) best learner vs. policy baseline at each p.

  (6) Ablations:
        A. XGBoost native missing-value handling vs. the indicator representation.
        B. Missingness-aware training vs. clean-only training.

  (7) Figures: degradation curve and confusion matrices (p=0 and p=0.5).

IMPORTANT ON RESULTS: every number produced here is COMPUTED on the synthetic
benchmark; nothing is fabricated. When reporting in the paper, frame results as
"on the RoMaCS benchmark" and remember the physical constants in romacs_datagen.py
are the [CALIBRATE] targets.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from scipy.stats import wilcoxon

from imblearn.ensemble import RUSBoostClassifier
from xgboost import XGBClassifier

# Reuse the data-generation module (must be in the same directory / import path).
import romacs_datagen as dg
from romacs_datagen import (
    GenConfig, generate_dataset, inject_mcar_missingness, locf_impute,
    add_missingness_indicators, train_test_split_by_trajectory, qos_columns,
    oracle_label, CHANNEL_NAMES, QOS_METRICS, LABELS, NO_CHANNEL_LABEL,
)

warnings.filterwarnings("ignore")  # silence sklearn/xgboost convergence chatter

ALL_LABELS = list(range(len(CHANNEL_NAMES) + 1))  # [0..5], includes NO_CHANNEL
CLASS_NAMES = [LABELS[i] for i in ALL_LABELS]

# System taxonomy used for the "why tree ensembles" comparison.
TREE_SYSTEMS = ["DecisionTree", "Bagging", "AdaBoost", "RUSBoost", "XGBoost"]
NONTREE_SYSTEMS = ["LogReg", "MLP"]  # linear + shallow neural baselines (non-tree)
LEARNER_SYSTEMS = TREE_SYSTEMS + NONTREE_SYSTEMS


# ----------------------------------------------------------------------------- #
#  EXPERIMENT CONFIGURATION
# ----------------------------------------------------------------------------- #
@dataclass
class ExpConfig:
    seeds: list[int] = field(default_factory=lambda: list(range(8)))
    n_trajectories: int = 600            # trajectories per seed
    missing_rates: tuple = (0.0, 0.10, 0.25, 0.50, 0.75)  # robustness sweep (test side)
    train_missing_rate: float = 0.25     # missingness injected into TRAIN (missingness-aware)
    test_size: float = 0.30              # trajectory-level held-out fraction
    outdir: str = "../results"


# ----------------------------------------------------------------------------- #
#  FEATURE COLUMN GROUPS
# ----------------------------------------------------------------------------- #
SCENARIO_COLS = [
    "distance_to_shore_km", "sea_state", "traffic_density",
    "weather_severity", "msg_priority", "msg_size_kb", "hour_of_day",
]
AVAIL_COLS = [f"{name}__available" for name in CHANNEL_NAMES]
QOS_COLS = qos_columns()                       # 20 QoS columns
IND_COLS = [f"{c}__isnan" for c in QOS_COLS]    # 20 missingness indicators


def indicator_feature_cols() -> list[str]:
    """52-dim representation given to ALL five learners (fair comparison)."""
    return SCENARIO_COLS + QOS_COLS + AVAIL_COLS + IND_COLS


def native_feature_cols() -> list[str]:
    """32-dim raw representation (QoS may contain NaN) for the XGBoost-native ablation."""
    return SCENARIO_COLS + QOS_COLS + AVAIL_COLS


# ----------------------------------------------------------------------------- #
#  MODEL FACTORY (fresh, seeded instances each call)
# ----------------------------------------------------------------------------- #
import json as _json, os as _os

def _load_tuned_configs():
    """Load tuned hyperparameters from ../configs/tuned_configs.json (the
    configuration selected by romacs_tuning.py and reported in Appendix B).
    Falls back to per-model defaults if the file is absent."""
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "configs", "tuned_configs.json")
    if _os.path.exists(path):
        with open(path) as f:
            return _json.load(f)
    return None

_TUNED = _load_tuned_configs()


def make_models(seed: int) -> dict:
    """Return a dict {name: fresh classifier}. If results/tuned_configs.json exists,
    every learner uses the hyperparameters selected by trajectory-grouped validation
    random search (romacs_tuning.py); otherwise modest defaults are used. Reported in
    full in the paper for reproducibility."""
    if _TUNED is not None:
        t = _TUNED
        ada = t["AdaBoost"].copy(); ada_dep = ada.pop("estimator_depth", 3)
        rus = t["RUSBoost"].copy(); rus_dep = rus.pop("estimator_depth", 3)
        mlp = t["MLP"]
        return {
            "DecisionTree": DecisionTreeClassifier(random_state=seed, **t["DecisionTree"]),
            "Bagging": BaggingClassifier(random_state=seed, n_jobs=-1, **t["Bagging"]),
            "AdaBoost": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=ada_dep),
                random_state=seed, **ada),
            "RUSBoost": RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=rus_dep),
                random_state=seed, **rus),
            "XGBoost": XGBClassifier(
                tree_method="hist", objective="multi:softprob",
                num_class=len(ALL_LABELS), random_state=seed, verbosity=0, **t["XGBoost"]),
            "LogReg": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=t["LogReg"]["C"]))]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=tuple(mlp["hidden"]),
                                      learning_rate_init=mlp["lr"], alpha=mlp["alpha"],
                                      activation=mlp["activation"], max_iter=300,
                                      early_stopping=True, random_state=seed))]),
        }
    # ---- fallback defaults (untuned) ----
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=seed),
        "Bagging": BaggingClassifier(
            n_estimators=25, random_state=seed, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50, random_state=seed),
        "RUSBoost": RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.3,
            tree_method="hist", objective="multi:softprob",
            num_class=len(ALL_LABELS), random_state=seed, verbosity=0),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                  early_stopping=True, random_state=seed)),
        ]),
    }


# ----------------------------------------------------------------------------- #
#  POLICY BASELINE (rule-based selector under LOCF-filled incomplete observation)
# ----------------------------------------------------------------------------- #
def policy_predict(df_locf: pd.DataFrame) -> np.ndarray:
    """Predict labels with the ORACLE policy applied to LOCF-imputed observations.

    This is the baseline a heuristic operator would achieve: the rule set is the same
    utility policy that generated the labels, but it now runs on stale/incomplete
    inputs. A channel whose QoS is STILL missing after LOCF (cold start) has no usable
    reading and is treated as non-selectable (availability forced to 0) — the
    conservative rule agreed in the design.
    """
    preds = np.empty(len(df_locf), dtype=int)
    records = df_locf.to_dict("records")
    for i, row in enumerate(records):
        qos_by_channel, availability = {}, {}
        for name in CHANNEL_NAMES:
            vals = {m: row[f"{name}__{m}"] for m in QOS_METRICS}
            # Cold-start missing after LOCF -> channel not usable by the policy.
            if any(pd.isna(v) for v in vals.values()):
                availability[name] = 0
                vals = {m: -1e9 if m != "per" else 1.0 for m in QOS_METRICS}  # dummy, unused
            else:
                availability[name] = int(row[f"{name}__available"])
            qos_by_channel[name] = vals
        preds[i] = oracle_label(
            qos_by_channel, availability, int(row["msg_priority"]), float(row["msg_size_kb"]))
    return preds


# ----------------------------------------------------------------------------- #
#  METRICS
# ----------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, priority: np.ndarray) -> dict:
    """Compute the full metric bundle for one (model, p, seed) evaluation."""
    out = {
        "macro_f1": f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
    # Per-class F1 (highlights minority classes LTE_5G and NO_CHANNEL).
    per_class = f1_score(y_true, y_pred, labels=ALL_LABELS, average=None, zero_division=0)
    for lbl, f1 in zip(ALL_LABELS, per_class):
        out[f"f1_{LABELS[lbl]}"] = f1
    # Safety-critical view: macro-F1 restricted to EMERGENCY-priority traffic.
    emask = priority == 2
    if emask.sum() > 0:
        out["macro_f1_emergency"] = f1_score(
            y_true[emask], y_pred[emask], labels=ALL_LABELS, average="macro", zero_division=0)
    else:
        out["macro_f1_emergency"] = np.nan
    return out


# ----------------------------------------------------------------------------- #
#  CORE EXPERIMENT
# ----------------------------------------------------------------------------- #
def run_seed(seed: int, cfg: ExpConfig) -> dict:
    """Run the full pipeline for a single seed; return a dict of result records."""
    # --- generate benchmark for this seed ------------------------------------
    df = generate_dataset(GenConfig(n_trajectories=cfg.n_trajectories, seed=seed))
    train_df, test_df = train_test_split_by_trajectory(df, test_size=cfg.test_size, seed=seed)

    # --- missingness-aware TRAIN representation (indicators) ------------------
    train_missing = inject_mcar_missingness(train_df, p=cfg.train_missing_rate, seed=seed + 1000)
    train_feat, fills = add_missingness_indicators(train_missing)   # fit fills on train
    Xtr = train_feat[indicator_feature_cols()].to_numpy(float)
    ytr = train_feat["label"].to_numpy(int)

    # --- CLEAN-only training variant (ablation B) ----------------------------
    train_clean_feat, fills_clean = add_missingness_indicators(train_df.copy())
    Xtr_clean = train_clean_feat[indicator_feature_cols()].to_numpy(float)

    # --- XGBoost NATIVE-missing training (ablation A) ------------------------
    Xtr_native = train_missing[native_feature_cols()].to_numpy(float)  # keeps NaN

    # --- fit all models (missingness-aware) ----------------------------------
    models = make_models(seed)
    for m in models.values():
        m.fit(Xtr, ytr)
    # ablation B: clean-trained twins
    models_clean = make_models(seed)
    for m in models_clean.values():
        m.fit(Xtr_clean, ytr)
    # ablation A: native XGBoost (same tuned XGBoost config as the main core)
    _xgb_kw = dict(tree_method="hist", objective="multi:softprob",
                   num_class=len(ALL_LABELS), random_state=seed, verbosity=0)
    if _TUNED is not None:
        _xgb_kw.update(_TUNED["XGBoost"])
    else:
        _xgb_kw.update(dict(n_estimators=100, max_depth=6, learning_rate=0.3))
    xgb_native = XGBClassifier(**_xgb_kw)
    xgb_native.fit(Xtr_native, ytr)

    records = []
    conf_store = {}  # (model, p) -> confusion matrix, for the best model later

    # --- robustness sweep on the TEST set ------------------------------------
    for p in cfg.missing_rates:
        test_missing = inject_mcar_missingness(test_df, p=p, seed=seed + int(1e6 * p) + 7)

        # learner representation (reuse TRAIN fills -> no leakage)
        test_feat, _ = add_missingness_indicators(test_missing, impute_values=fills)
        Xte = test_feat[indicator_feature_cols()].to_numpy(float)
        yte = test_feat["label"].to_numpy(int)
        prio = test_feat["msg_priority"].to_numpy(int)

        # (i) the five main models
        for name, m in models.items():
            yp = m.predict(Xte)
            rec = {"seed": seed, "system": name, "p": p, "variant": "aware"}
            rec.update(compute_metrics(yte, yp, prio))
            records.append(rec)
            conf_store[(name, p)] = confusion_matrix(yte, yp, labels=ALL_LABELS)

        # (ii) clean-trained twins (ablation B)
        test_feat_clean, _ = add_missingness_indicators(test_missing, impute_values=fills_clean)
        Xte_clean = test_feat_clean[indicator_feature_cols()].to_numpy(float)
        for name, m in models_clean.items():
            yp = m.predict(Xte_clean)
            rec = {"seed": seed, "system": name, "p": p, "variant": "clean"}
            rec.update(compute_metrics(yte, yp, prio))
            records.append(rec)

        # (iii) XGBoost native (ablation A)
        Xte_native = test_missing[native_feature_cols()].to_numpy(float)  # keeps NaN
        yp = xgb_native.predict(Xte_native)
        rec = {"seed": seed, "system": "XGBoost_native", "p": p, "variant": "aware"}
        rec.update(compute_metrics(yte, yp, prio))
        records.append(rec)

        # (iv) policy baseline: LOCF-impute the incomplete observation, then apply the rule
        test_locf = locf_impute(test_missing)
        yp_pol = policy_predict(test_locf)
        rec = {"seed": seed, "system": "PolicyLOCF", "p": p, "variant": "baseline"}
        rec.update(compute_metrics(yte, yp_pol, prio))
        records.append(rec)
        conf_store[("PolicyLOCF", p)] = confusion_matrix(yte, yp_pol, labels=ALL_LABELS)

    return {"records": records, "conf": conf_store}


# ----------------------------------------------------------------------------- #
#  AGGREGATION, STATISTICS, FIGURES
# ----------------------------------------------------------------------------- #
def robustness_summary(df_main: pd.DataFrame) -> pd.DataFrame:
    """Compute retention and AUDC per (system, seed), then aggregate over seeds.

    retention(p) = macro_f1(p) / macro_f1(0)
    AUDC         = trapezoidal integral of retention over p, normalized by p-range
                   (1.0 == perfectly robust / no degradation).
    """
    ps = sorted(df_main["p"].unique())
    rows = []
    for (system, seed), g in df_main.groupby(["system", "seed"]):
        g = g.set_index("p").sort_index()
        f0 = g.loc[ps[0], "macro_f1"]
        retention = (g["macro_f1"] / f0).reindex(ps)
        audc = np.trapezoid(retention.values, ps) / (ps[-1] - ps[0])
        rows.append({"system": system, "seed": seed, "AUDC": audc,
                     "macro_f1_clean": f0})
    return pd.DataFrame(rows)


def wilcoxon_between_models(df_main: pd.DataFrame, at_p: float = 0.0) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests between learners at a fixed p (across seeds)."""
    systems = ["DecisionTree", "Bagging", "AdaBoost", "RUSBoost", "XGBoost"]
    sub = df_main[(df_main["p"] == at_p)].pivot_table(
        index="seed", columns="system", values="macro_f1")
    rows = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            a, b = systems[i], systems[j]
            try:
                stat, pval = wilcoxon(sub[a], sub[b])
            except ValueError:
                stat, pval = np.nan, np.nan  # e.g., all-zero differences
            rows.append({"model_A": a, "model_B": b,
                         "mean_A": sub[a].mean(), "mean_B": sub[b].mean(),
                         "wilcoxon_stat": stat, "p_value": pval})
    return pd.DataFrame(rows)


def wilcoxon_vs_policy(df_main: pd.DataFrame, best_model: str) -> pd.DataFrame:
    """Paired Wilcoxon of best learner vs. policy baseline at each missingness rate."""
    rows = []
    for p in sorted(df_main["p"].unique()):
        ml = df_main[(df_main["system"] == best_model) & (df_main["p"] == p)] \
            .set_index("seed")["macro_f1"]
        pol = df_main[(df_main["system"] == "PolicyLOCF") & (df_main["p"] == p)] \
            .set_index("seed")["macro_f1"]
        idx = ml.index.intersection(pol.index)
        try:
            stat, pval = wilcoxon(ml.loc[idx], pol.loc[idx])
        except ValueError:
            stat, pval = np.nan, np.nan
        rows.append({"p": p, "mean_ML": ml.mean(), "mean_policy": pol.mean(),
                     "wilcoxon_stat": stat, "p_value": pval})
    return pd.DataFrame(rows)


def plot_degradation(df_main: pd.DataFrame, outpath: str, best_model: str) -> None:
    """Headline figure: macro-F1 vs missingness rate, learners + policy baseline."""
    systems = ["DecisionTree", "Bagging", "AdaBoost", "RUSBoost", "XGBoost",
               "LogReg", "MLP", "PolicyLOCF"]
    ps = sorted(df_main["p"].unique())
    plt.figure(figsize=(7.5, 5.5))
    for system in systems:
        g = df_main[df_main["system"] == system].groupby("p")["macro_f1"]
        if g.count().empty:
            continue
        mean, std = g.mean().reindex(ps), g.std().reindex(ps)
        style = dict(marker="o", linewidth=2)
        if system == "PolicyLOCF":
            style = dict(marker="s", linewidth=2, linestyle="--", color="black")
        elif system in NONTREE_SYSTEMS:  # non-tree baselines: dotted, distinct
            style = dict(marker="^", linewidth=2, linestyle=":")
        plt.plot(ps, mean.values, label=system, **style)
        plt.fill_between(ps, (mean - std).values, (mean + std).values, alpha=0.10)
    plt.xlabel("QoS missingness rate  p")
    plt.ylabel("macro-F1")
    plt.title("Robustness under increasing QoS missingness (mean ± std over seeds)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_confusion(conf: np.ndarray, title: str, outpath: str) -> None:
    """Render a normalized confusion matrix with large, print-legible labels."""
    cm = conf.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    # short, legible axis labels (full names are used in the paper text)
    short = {"VHF_DSC": "VHF", "dPMR": "dPMR", "AIS_VDES": "AIS", "LTE_5G": "LTE",
             "SATELLITE": "SAT", "NO_CHANNEL": "NONE"}
    labels = [short.get(c, c) for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            if row_sums[i] > 0:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=12.5, color="black" if cmn[i, j] < 0.6 else "white")
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- #
#  "WHY TREE ENSEMBLES" COMPARISON
# ----------------------------------------------------------------------------- #
def tree_vs_nontree_table(df_main: pd.DataFrame, rob: pd.DataFrame) -> pd.DataFrame:
    """Build the 'why tree ensembles' comparison table.

    For every learner, report clean macro-F1 (p=0), macro-F1 at the highest
    missingness rate, the absolute drop, and the robustness summary AUDC, tagged by
    model family (Tree vs Non-tree). This is the evidence that tree ensembles are the
    appropriate choice for THIS task, rather than an unjustified default.
    """
    ps = sorted(df_main["p"].unique())
    p_hi = ps[-1]
    audc_mean = rob.groupby("system")["AUDC"].mean()
    rows = []
    for system in LEARNER_SYSTEMS:
        g = df_main[df_main["system"] == system].groupby("p")["macro_f1"].mean()
        f0, fhi = g.get(ps[0], np.nan), g.get(p_hi, np.nan)
        rows.append({
            "family": "Tree" if system in TREE_SYSTEMS else "Non-tree",
            "system": system,
            "macroF1_p0": f0,
            f"macroF1_p{int(p_hi*100)}": fhi,
            "abs_drop": f0 - fhi,
            "AUDC": audc_mean.get(system, np.nan),
        })
    out = pd.DataFrame(rows).sort_values(["family", "AUDC"], ascending=[True, False])
    return out


# ----------------------------------------------------------------------------- #
#  MAIN
# ----------------------------------------------------------------------------- #
def main() -> None:
    import os
    cfg = ExpConfig()
    os.makedirs(cfg.outdir, exist_ok=True)

    print(f"Running RoMaCS experiment: {len(cfg.seeds)} seeds, "
          f"missingness sweep {cfg.missing_rates} ...")
    all_records, all_conf = [], {}
    for seed in cfg.seeds:
        res = run_seed(seed, cfg)
        all_records.extend(res["records"])
        # accumulate confusion matrices across seeds
        for k, cm in res["conf"].items():
            all_conf[k] = all_conf.get(k, np.zeros_like(cm)) + cm
        print(f"  seed {seed} done.")

    df = pd.DataFrame(all_records)
    df.to_csv(f"{cfg.outdir}/results_long.csv", index=False)

    # ----- main table: all learners (tree + non-tree) + policy -----
    main_systems = LEARNER_SYSTEMS + ["PolicyLOCF"]
    df_main = df[df["system"].isin(main_systems) & df["variant"].isin(["aware", "baseline"])].copy()

    metric_cols = ["macro_f1", "balanced_acc", "macro_f1_emergency",
                   "f1_LTE_5G", "f1_NO_CHANNEL"]
    summary = df_main.groupby(["system", "p"])[metric_cols].agg(["mean", "std"])
    summary.to_csv(f"{cfg.outdir}/summary_by_system_p.csv")

    # pick best TREE learner by mean macro-F1 at p=0 (keeps the "why trees" framing)
    p0 = df_main[(df_main["p"] == 0.0) & (df_main["system"].isin(TREE_SYSTEMS))]
    best_model = p0.groupby("system")["macro_f1"].mean().idxmax()

    # ----- robustness summary (AUDC) -----
    rob = robustness_summary(df_main)
    rob_agg = rob.groupby("system")[["macro_f1_clean", "AUDC"]].agg(["mean", "std"])
    rob_agg.to_csv(f"{cfg.outdir}/robustness_audc.csv")

    # ----- "why trees" comparison (tree vs non-tree) -----
    tvn = tree_vs_nontree_table(df_main, rob)
    tvn.to_csv(f"{cfg.outdir}/why_trees_tree_vs_nontree.csv", index=False)

    # ----- statistics -----
    wb = wilcoxon_between_models(df_main, at_p=0.0)
    wb.to_csv(f"{cfg.outdir}/wilcoxon_between_models_p0.csv", index=False)
    wp = wilcoxon_vs_policy(df_main, best_model)
    wp.to_csv(f"{cfg.outdir}/wilcoxon_vs_policy.csv", index=False)

    # ----- ablations -----
    abl_A = df[df["system"].isin(["XGBoost", "XGBoost_native"]) & (df["variant"] == "aware")] \
        .groupby(["system", "p"])["macro_f1"].mean().unstack("system")
    abl_A.to_csv(f"{cfg.outdir}/ablation_A_xgb_native_vs_indicator.csv")
    abl_B = df[(df["system"] == best_model)] \
        .groupby(["variant", "p"])["macro_f1"].mean().unstack("variant")
    abl_B.to_csv(f"{cfg.outdir}/ablation_B_aware_vs_clean.csv")

    # ----- figures -----
    plot_degradation(df_main, f"{cfg.outdir}/fig_degradation.png", best_model)
    plot_confusion(all_conf[(best_model, 0.0)],
                   f"{best_model} confusion matrix (p=0)",
                   f"{cfg.outdir}/fig_confusion_p0.png")
    plot_confusion(all_conf[(best_model, 0.5)],
                   f"{best_model} confusion matrix (p=0.5)",
                   f"{cfg.outdir}/fig_confusion_p50.png")

    # ----- console report -----
    print("\n" + "=" * 72)
    print(f"Best learner by clean macro-F1: {best_model}")
    print("=" * 72)
    print("\nMean macro-F1 by system across missingness rates:")
    piv = df_main.groupby(["system", "p"])["macro_f1"].mean().unstack("p")
    print(piv.round(3).to_string())
    print("\nRobustness (AUDC, 1.0 = no degradation), mean over seeds:")
    print(rob.groupby("system")["AUDC"].mean().round(3).sort_values(ascending=False).to_string())
    print("\nEmergency-priority macro-F1 by system:")
    pive = df_main.groupby(["system", "p"])["macro_f1_emergency"].mean().unstack("p")
    print(pive.round(3).to_string())

    print("\n" + "-" * 72)
    print("WHY TREE ENSEMBLES — tree vs non-tree learners:")
    print(tvn.round(3).to_string(index=False))
    # paired Wilcoxon: best tree vs each non-tree baseline at the highest p
    ps = sorted(df_main["p"].unique()); p_hi = ps[-1]
    print(f"\nPaired Wilcoxon (best tree = {best_model}) vs non-tree at p={p_hi}:")
    for nt in NONTREE_SYSTEMS:
        a = df_main[(df_main.system == best_model) & (df_main.p == p_hi)].set_index("seed")["macro_f1"]
        b = df_main[(df_main.system == nt) & (df_main.p == p_hi)].set_index("seed")["macro_f1"]
        idx = a.index.intersection(b.index)
        try:
            _, pv = wilcoxon(a.loc[idx], b.loc[idx])
        except ValueError:
            pv = np.nan
        print(f"  {best_model} ({a.mean():.3f}) vs {nt} ({b.mean():.3f}):  p-value = {pv:.4f}")
    print("-" * 72)

    print(f"\nWilcoxon {best_model} vs PolicyLOCF (per p):")
    print(wp.round(4).to_string(index=False))
    print("\nAblation A — XGBoost native-missing vs indicator (mean macro-F1):")
    print(abl_A.round(3).to_string())
    print("\nAblation B — missingness-aware vs clean training (mean macro-F1):")
    print(abl_B.round(3).to_string())
    print(f"\nArtifacts saved to: {cfg.outdir}/")


if __name__ == "__main__":
    main()
