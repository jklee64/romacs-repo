"""
romacs_v2_experiment.py — final numbers for the IoTJ revision.

PART A — STALENESS (AoI) SWEEP, 8 seeds x 1,200 trajectories
  Systems (XGBoost core unless noted):
    AgeAware   : AoI features + mixed-rate staleness training      (PROPOSED)
    NoAgeFeat  : staleness training but NO AoI features            (abl. C1)
    StaleBlind : fresh-only training, no AoI features              (abl. C2)
    PolicyStale: oracle utility policy applied to stale observation
  Sweep: probe rate r in {1.0, 0.5, 0.25, 0.1, 0.05}
  Stats: Wilcoxon signed-rank AgeAware vs PolicyStale per r (8 paired seeds).

PART B — MONDRIAN CONFORMAL ABSTENTION, 8 seeds x 1,200 trajectories
  (i)  on the MCAR core across p in {0, .1, .25, .5, .75} (continuity w/ paper)
  (ii) on the AgeAware core across the staleness sweep (full pipeline)
  Metrics: class-cond. coverage, abstention rate, base/selective macro-F1,
           emergency-subset error, false no_channel rate.

Outputs: results_v2/*.csv
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from romacs_datagen import (
    generate_dataset, GenConfig, train_test_split_by_trajectory,
    inject_mcar_missingness, add_missingness_indicators, qos_columns,
    CHANNEL_NAMES, QOS_METRICS,
)
from romacs_experiment import policy_predict

SEEDS = list(range(8))
N_TRAJ = 600
ALL = list(range(6)); NC = 5
ALPHA = 0.05
AGE_CAP = 20
PROBE_RATES = [1.0, 0.5, 0.25, 0.1, 0.05]
MCAR_RATES = [0.0, 0.10, 0.25, 0.50, 0.75]
OUT = "../results/results_v2"
os.makedirs(OUT, exist_ok=True)

SCEN = ["distance_to_shore_km","sea_state","traffic_density",
        "weather_severity","msg_priority","msg_size_kb","hour_of_day"]
AVAIL = [f"{n}__available" for n in CHANNEL_NAMES]


# --------------------------------------------------------------------------- #
# staleness machinery
# --------------------------------------------------------------------------- #
def inject_staleness(df: pd.DataFrame, probe_rate: float, seed: int) -> pd.DataFrame:
    """Per-channel probing process: fresh probe w.p. probe_rate per step, else
    carry last probe forward; AoI (age) increments. Step 0 always probed."""
    rng = np.random.default_rng(seed)
    out = df.sort_values(["traj_id","step"]).copy()
    traj = out["traj_id"].to_numpy()
    n = len(out)
    for ch in CHANNEL_NAMES:
        cols = [f"{ch}__{m}" for m in QOS_METRICS]
        vals = out[cols].to_numpy(float)
        stale = vals.copy()
        age = np.zeros(n)
        probes = rng.random(n) < probe_rate
        last_vals = None; last_traj = None; cur = 0
        for i in range(n):
            if traj[i] != last_traj or probes[i]:
                last_vals = vals[i].copy(); cur = 0
            else:
                cur += 1
            stale[i] = last_vals; age[i] = cur; last_traj = traj[i]
        out[cols] = stale
        out[f"{ch}__age"] = age
    return out


def inject_staleness_mixed(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Training-time staleness: per-trajectory probe rate ~ U[0.05, 1.0]."""
    rng = np.random.default_rng(seed)
    parts = []
    for tid, g in df.groupby("traj_id"):
        parts.append(inject_staleness(g, float(rng.uniform(0.05, 1.0)), seed + int(tid)))
    return pd.concat(parts, ignore_index=True)


def feat_stale(df: pd.DataFrame, with_age: bool) -> np.ndarray:
    X = df[SCEN + qos_columns() + AVAIL].to_numpy(float)
    if with_age:
        A = np.clip(df[[f"{c}__age" for c in CHANNEL_NAMES]].to_numpy(float), 0, AGE_CAP) / AGE_CAP
        X = np.hstack([X, A])
    return X


def feat_mcar_cols():
    return SCEN + qos_columns() + AVAIL + [f"{c}__isnan" for c in qos_columns()]


def xgb(seed):
    # tuned config from the paper (Appendix B)
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                         subsample=0.8, tree_method="hist",
                         objective="multi:softprob", num_class=6,
                         random_state=seed, verbosity=0)


def mondrian_thresholds(P_cal, y_cal, alpha):
    """Class-conditional (Mondrian) thresholds with a MARGINAL FALLBACK for
    classes whose calibration count is too small for a finite (1-alpha)
    quantile to exist (need n_k >= (1-alpha)/alpha, i.e. 19 for alpha=0.05).
    Without the fallback the threshold degenerates to 1.0 and the class is
    included in every prediction set, inflating abstention."""
    s = 1.0 - P_cal[np.arange(len(y_cal)), y_cal]
    n_all = len(s)
    q_marg = np.quantile(s, min(1.0, np.ceil((n_all+1)*(1-alpha))/n_all), method="higher")
    n_min = int(np.ceil((1 - alpha) / alpha))
    thr = np.full(6, q_marg)
    for k in ALL:
        sk = s[y_cal == k]; n = len(sk)
        if n >= n_min:
            thr[k] = np.quantile(sk, min(1.0, np.ceil((n+1)*(1-alpha))/n), method="higher")
    return thr


def selective_metrics(P, y, prio, thr):
    sets = P >= (1.0 - thr)[None, :]
    am = P.argmax(1)
    sets[np.arange(len(y)), am] = True
    size = sets.sum(1)
    commit = size == 1
    yp = am
    rec = dict(
        cov=float(sets[np.arange(len(y)), y].mean()),
        abst=float(1 - commit.mean()),
        baseF1=f1_score(y, yp, average="macro", labels=ALL, zero_division=0),
        selF1=(f1_score(y[commit], yp[commit], average="macro", labels=ALL, zero_division=0)
               if commit.any() else np.nan),
        err_base=float((yp != y).mean()),
        err_sel=float((yp[commit] != y[commit]).mean()) if commit.any() else np.nan,
    )
    em = prio == 2; emc = em & commit
    rec["errEM_base"] = float((yp[em] != y[em]).mean()) if em.any() else np.nan
    rec["errEM_sel"] = float((yp[emc] != y[emc]).mean()) if emc.any() else np.nan
    rec["abstEM"] = float(1 - commit[em].mean()) if em.any() else np.nan
    nz = y != NC
    rec["fNC_base"] = float(((yp == NC) & nz).sum() / max(1, nz.sum()))
    nzc = nz & commit
    rec["fNC_sel"] = float(((yp == NC) & nzc).sum() / max(1, nzc.sum()))
    return rec


# --------------------------------------------------------------------------- #
def run_seed(seed):
    print(f"[seed {seed}] generating...", flush=True)
    df = generate_dataset(GenConfig(n_trajectories=N_TRAJ, seed=seed))
    tr_all, te = train_test_split_by_trajectory(df, 0.3, seed)
    tr, cal = train_test_split_by_trajectory(tr_all, 0.25, seed + 5)

    # ============================ PART A: staleness =========================
    tr_mix = inject_staleness_mixed(tr_all, seed + 11)   # full train for part A
    y_mix = tr_mix["label"].to_numpy(int)
    m_age = xgb(seed);  m_age.fit(feat_stale(tr_mix, True),  y_mix)
    m_noage = xgb(seed); m_noage.fit(feat_stale(tr_mix, False), y_mix)
    tr_fresh = tr_all.copy()
    m_blind = xgb(seed); m_blind.fit(feat_stale(
        inject_staleness(tr_fresh, 1.0, seed + 12), False), tr_fresh.sort_values(["traj_id","step"])["label"].to_numpy(int))

    a_rows = []
    for r in PROBE_RATES:
        te_st = inject_staleness(te, r, seed + int(1000*r) + 7)
        yte = te_st["label"].to_numpy(int)
        mean_age = float(te_st[[f"{c}__age" for c in CHANNEL_NAMES]].to_numpy().mean())
        preds = {
            "AgeAware":   m_age.predict(feat_stale(te_st, True)),
            "NoAgeFeat":  m_noage.predict(feat_stale(te_st, False)),
            "StaleBlind": m_blind.predict(feat_stale(te_st, False)),
            "PolicyStale": policy_predict(te_st),
        }
        for name, yp in preds.items():
            a_rows.append(dict(seed=seed, system=name, r=r, mean_age=mean_age,
                               macroF1=f1_score(yte, yp, average="macro",
                                                labels=ALL, zero_division=0)))

    # ==================== PART B(i): conformal on MCAR core =================
    tr_m = inject_mcar_missingness(tr, 0.25, seed + 21)
    tr_f, fills = add_missingness_indicators(tr_m)
    m_mcar = xgb(seed)
    m_mcar.fit(tr_f[feat_mcar_cols()].to_numpy(float), tr_f["label"].to_numpy(int))

    b_rows = []
    for p in MCAR_RATES:
        cal_f, _ = add_missingness_indicators(
            inject_mcar_missingness(cal, p, seed + int(100*p) + 3), impute_values=fills)
        thr = mondrian_thresholds(
            m_mcar.predict_proba(cal_f[feat_mcar_cols()].to_numpy(float)),
            cal_f["label"].to_numpy(int), ALPHA)
        te_f, _ = add_missingness_indicators(
            inject_mcar_missingness(te, p, seed + int(1e4*p) + 7), impute_values=fills)
        rec = selective_metrics(
            m_mcar.predict_proba(te_f[feat_mcar_cols()].to_numpy(float)),
            te_f["label"].to_numpy(int), te_f["msg_priority"].to_numpy(int), thr)
        rec.update(seed=seed, regime="MCAR", level=p)
        b_rows.append(rec)

    # =========== PART B(ii): conformal on AgeAware under staleness ==========
    tr_mix2 = inject_staleness_mixed(tr, seed + 31)      # train w/o calibration split
    m_age2 = xgb(seed)
    m_age2.fit(feat_stale(tr_mix2, True), tr_mix2["label"].to_numpy(int))
    for r in PROBE_RATES:
        cal_st = inject_staleness(cal, r, seed + int(100*r) + 13)
        thr = mondrian_thresholds(
            m_age2.predict_proba(feat_stale(cal_st, True)),
            cal_st["label"].to_numpy(int), ALPHA)
        te_st = inject_staleness(te, r, seed + int(1000*r) + 7)
        rec = selective_metrics(
            m_age2.predict_proba(feat_stale(te_st, True)),
            te_st["label"].to_numpy(int), te_st["msg_priority"].to_numpy(int), thr)
        rec.update(seed=seed, regime="staleness", level=r)
        b_rows.append(rec)

    print(f"[seed {seed}] done", flush=True)
    return a_rows, b_rows


def main():
    import sys as _sys
    seeds = [int(x) for x in _sys.argv[1:]] if len(_sys.argv) > 1 else SEEDS
    A, B = [], []
    for s in seeds:
        a, b = run_seed(s)
        A += a; B += b
        for path, rows in [(f"{OUT}/staleness_raw.csv", a), (f"{OUT}/abstention_raw.csv", b)]:
            dfp = pd.DataFrame(rows)
            hdr = not os.path.exists(path)
            dfp.to_csv(path, mode="a", header=hdr, index=False)
    # aggregate over whatever is on disk
    A = pd.DataFrame(pd.read_csv(f"{OUT}/staleness_raw.csv")).to_dict("records")
    B = pd.DataFrame(pd.read_csv(f"{OUT}/abstention_raw.csv")).to_dict("records")

    dfA = pd.DataFrame(A)
    aggA = dfA.groupby(["system","r"])["macroF1"].agg(["mean","std"]).reset_index()
    aggA.to_csv(f"{OUT}/staleness_summary.csv", index=False)

    # Wilcoxon AgeAware vs PolicyStale per r
    w_rows = []
    for r in PROBE_RATES:
        x = dfA[(dfA.system=="AgeAware") & (dfA.r==r)].sort_values("seed")["macroF1"].to_numpy()
        yv = dfA[(dfA.system=="PolicyStale") & (dfA.r==r)].sort_values("seed")["macroF1"].to_numpy()
        try:
            stat, pv = wilcoxon(x, yv)
        except ValueError:
            stat, pv = np.nan, np.nan
        w_rows.append(dict(r=r, mean_age=float(dfA[dfA.r==r]["mean_age"].mean()),
                           AgeAware=x.mean(), PolicyStale=yv.mean(), p_val=pv))
    pd.DataFrame(w_rows).to_csv(f"{OUT}/staleness_wilcoxon.csv", index=False)

    dfB = pd.DataFrame(B)
    aggB = dfB.groupby(["regime","level"]).mean().drop(columns="seed").reset_index()
    aggB.to_csv(f"{OUT}/abstention_summary.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print("\n=== STALENESS (macro-F1, mean over 8 seeds) ===")
    print(aggA.pivot(index="r", columns="system", values="mean").sort_index(ascending=False).to_string())
    print("\n=== Wilcoxon AgeAware vs PolicyStale ===")
    print(pd.DataFrame(w_rows).to_string(index=False))
    print("\n=== ABSTENTION (mean over 8 seeds) ===")
    print(aggB.to_string(index=False))


if __name__ == "__main__":
    main()
