"""
romacs_datagen.py
=================

RoMaCS (Robust Maritime Channel Selection) — synthetic benchmark generator.

This module generates a *trajectory-based* (time-series) benchmark for the task of
selecting the most appropriate maritime communication channel under *incomplete*
channel-quality observations.

Design rationale (see accompanying paper, Sections III & V)
-----------------------------------------------------------
1. TRAJECTORY-BASED, NOT I.I.D. SNAPSHOTS.
   Each sample belongs to a short vessel trajectory in which the vessel moves and
   the environment evolves smoothly over time. This is required because the policy
   baseline imputes missing values with LOCF (Last Observation Carried Forward),
   which only has meaning along a temporal sequence. It also lets us split
   train/test by trajectory (no temporal leakage) and later enables staleness
   experiments for free.

2. PHYSICALLY-MOTIVATED QoS, NOT ARBITRARY NUMBERS.
   The 20 per-channel QoS values are NOT independent random draws. They are produced
   by a physical chain:
        distance -> path loss -> RSSI -> (noise + interference) -> SINR
                 -> PER (waterfall curve) and throughput (Shannon w/ MCS cap).
   This is the single most important defence against the "your synthetic data is
   meaningless" reviewer objection. All physical constants are gathered in
   CHANNELS / PHYS below and are the calibration targets that must be cited to
   literature in the Methods section (marked with [CALIBRATE]).

3. ORACLE LABELS FROM COMPLETE INFORMATION.
   Labels are produced by an oracle that sees the COMPLETE QoS of every channel and
   applies a deterministic utility policy. Crucially, at inference time the learners
   and the policy baseline see only the *incomplete* (masked) observation. This is
   what justifies the ML framing: the model must recover the oracle policy under
   partial observability, which a rule-based selector cannot do robustly.

Feature layout (32 dimensions), matching the paper:
    -  7 scenario attributes   (always observed)
    - 20 per-channel QoS values (5 channels x 4 metrics; SUBJECT TO MISSINGNESS)
    -  5 channel availability flags (always observed)
Target:
    - label in {0..4} = channel index, or 5 = NO_CHANNEL fallback.

NOTE ON CHANNEL LATENCY: channel latency is a *static* per-channel property used by
the oracle for feasibility checking. It is intentionally NOT one of the four QoS
metrics and NOT a feature (consistent with the paper's feature design).

Author: (RoMaCS)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
#  0. CONSTANTS AND CHANNEL SPECIFICATIONS
# ----------------------------------------------------------------------------- #
# Physical constants used throughout the link-budget computation.
PHYS = {
    "thermal_noise_dbm_per_hz": -174.0,  # kTB at ~290 K, per Hz, in dBm/Hz
    "speed_of_light": 3e8,               # m/s (not directly used; kept for clarity)
}

# Ordered list of channels. The index in this list IS the class label 0..4.
# label 5 is reserved for NO_CHANNEL (see LABELS below).
CHANNEL_NAMES = ["VHF_DSC", "dPMR", "AIS_VDES", "LTE_5G", "SATELLITE"]
N_CHANNELS = len(CHANNEL_NAMES)
NO_CHANNEL_LABEL = N_CHANNELS  # == 5
LABELS = {i: name for i, name in enumerate(CHANNEL_NAMES)}
LABELS[NO_CHANNEL_LABEL] = "NO_CHANNEL"

# The four QoS metrics measured per channel (order matters for column naming).
QOS_METRICS = ["rssi_dbm", "sinr_db", "per", "throughput_kbps"]

# ----------------------------------------------------------------------------- #
#  Per-channel physical parameters.
#
#  [CALIBRATE] Every numeric value below is a modelling choice that should be
#  calibrated against and cited to the maritime-communications literature in the
#  paper's Methods section. The values here are physically plausible defaults,
#  chosen to produce realistic RSSI/SINR/PER/throughput ranges and a non-trivial
#  channel-selection problem — not authoritative measurements.
# ----------------------------------------------------------------------------- #
@dataclass
class ChannelSpec:
    name: str
    freq_mhz: float          # carrier frequency (drives free-space path loss)
    tx_power_dbm: float      # effective transmit power
    tx_gain_db: float        # transmitter antenna gain
    rx_gain_db: float        # receiver antenna gain
    bandwidth_hz: float      # channel bandwidth (drives noise floor & Shannon rate)
    noise_figure_db: float   # receiver noise figure
    path_loss_exp: float     # log-distance path-loss exponent n (over-sea LOS ~2)
    coverage_km: float       # nominal max range; beyond this the channel is unavailable
    max_rate_kbps: float     # MCS / protocol cap on throughput
    latency_ms: float        # STATIC one-way latency (used by oracle, NOT a feature)
    cost_norm: float         # normalized monetary/operational cost in [0,1]
    weather_sensitivity: float  # extra dB loss per unit weather severity (rain fade)
    is_satellite: bool = False  # satellite is modelled specially (see qos below)


# The channel roster. See [CALIBRATE] note above.
CHANNELS: dict[str, ChannelSpec] = {
    "VHF_DSC": ChannelSpec(
        name="VHF_DSC", freq_mhz=156.0, tx_power_dbm=44.0,   # ~25 W
        tx_gain_db=6.0, rx_gain_db=3.0, bandwidth_hz=25e3,
        noise_figure_db=6.0, path_loss_exp=2.2, coverage_km=40.0,
        # [CALIBRATE] rate represents an enhanced VHF/VDE-class data mode (VDES era),
        # not the ~1.2 kbps of legacy DSC signalling. Adjust to your literature.
        max_rate_kbps=9.6, latency_ms=100.0, cost_norm=0.05,
        weather_sensitivity=0.3,
    ),
    "dPMR": ChannelSpec(
        name="dPMR", freq_mhz=450.0, tx_power_dbm=37.0,      # ~5 W
        tx_gain_db=3.0, rx_gain_db=2.0, bandwidth_hz=6.25e3,
        noise_figure_db=7.0, path_loss_exp=2.5, coverage_km=20.0,
        # dPMR niche: short-range port/harbour radio with the LOWEST latency, so it
        # wins latency-sensitive small traffic near shore. [CALIBRATE]
        max_rate_kbps=16.0, latency_ms=55.0, cost_norm=0.08,
        weather_sensitivity=0.4,
    ),
    "AIS_VDES": ChannelSpec(
        name="AIS_VDES", freq_mhz=162.0, tx_power_dbm=41.0,  # ~12.5 W
        tx_gain_db=6.0, rx_gain_db=3.0, bandwidth_hz=100e3,  # VDES wider than legacy AIS
        noise_figure_db=6.0, path_loss_exp=2.2, coverage_km=45.0,
        max_rate_kbps=300.0, latency_ms=150.0, cost_norm=0.10,
        weather_sensitivity=0.3,
    ),
    "LTE_5G": ChannelSpec(
        name="LTE_5G", freq_mhz=1800.0, tx_power_dbm=46.0,   # base-station EIRP-ish
        tx_gain_db=12.0, rx_gain_db=2.0, bandwidth_hz=10e6,
        noise_figure_db=7.0, path_loss_exp=3.2, coverage_km=30.0,  # limited offshore
        max_rate_kbps=50000.0, latency_ms=40.0, cost_norm=0.30,
        weather_sensitivity=0.8,
    ),
    "SATELLITE": ChannelSpec(
        name="SATELLITE", freq_mhz=12000.0, tx_power_dbm=50.0,  # Ku-band VSAT-ish
        tx_gain_db=35.0, rx_gain_db=35.0, bandwidth_hz=1e6,
        noise_figure_db=3.0, path_loss_exp=2.0, coverage_km=1e9,  # effectively global
        max_rate_kbps=2000.0, latency_ms=600.0, cost_norm=0.90,   # GEO-like latency & cost
        weather_sensitivity=4.0,  # strong rain fade at Ku-band
        is_satellite=True,
    ),
}

# ----------------------------------------------------------------------------- #
#  Message-type requirements used by the oracle for feasibility + utility.
#  Priority is one of the 7 scenario attributes (0=routine, 1=safety, 2=emergency).
#  [CALIBRATE] Requirement thresholds are modelling choices to be justified in text.
# ----------------------------------------------------------------------------- #
#  The required throughput is computed dynamically as:
#      req_throughput = base_throughput_kbps + throughput_per_kb * msg_size_kb
#  so that small messages (e.g., tiny distress alerts) have low bandwidth needs and
#  can be served by narrowband channels (VHF-DSC, dPMR), giving those channels a
#  realistic operational niche instead of being dominated out of every decision.
THROUGHPUT_PER_KB = 0.8  # kbps of required rate per kB of message payload
MSG_REQUIREMENTS = {
    0: {  # routine (e.g., logistics, non-urgent telemetry) — can be large payloads
        "base_throughput_kbps": 1.0, "max_per": 0.10, "max_latency_ms": 3000.0,
        "size_mean_kb": 8.0, "size_max_kb": 50.0,
        "w_reliability": 1.0, "w_throughput": 0.5, "w_latency": 0.5, "w_cost": 1.0,
    },
    1: {  # safety (e.g., navigational safety information) — typically small alerts
        "base_throughput_kbps": 2.5, "max_per": 0.05, "max_latency_ms": 1500.0,
        "size_mean_kb": 3.0, "size_max_kb": 15.0,
        "w_reliability": 2.0, "w_throughput": 0.4, "w_latency": 1.0, "w_cost": 0.3,
    },
    2: {  # emergency (e.g., distress) — tiny, reliability & reachability dominate; cost ignored
        "base_throughput_kbps": 0.5, "max_per": 0.02, "max_latency_ms": 2000.0,
        "size_mean_kb": 1.0, "size_max_kb": 5.0,
        "w_reliability": 4.0, "w_throughput": 0.2, "w_latency": 1.0, "w_cost": 0.0,
    },
}


# ----------------------------------------------------------------------------- #
#  1. PHYSICAL LINK-BUDGET MODEL
#     distance -> path loss -> RSSI -> SINR -> PER, throughput
# ----------------------------------------------------------------------------- #
def free_space_path_loss_db(distance_km: float, freq_mhz: float) -> float:
    """Free-space path loss (FSPL) in dB for a reference distance.

    FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44   (d in km, f in MHz)

    We use FSPL only to anchor the path loss at the 1 km reference distance; the
    distance-dependent growth is then governed by the log-distance exponent below.
    """
    d = max(distance_km, 1e-3)  # guard against log(0)
    return 20.0 * np.log10(d) + 20.0 * np.log10(freq_mhz) + 32.44


def log_distance_path_loss_db(
    distance_km: float, spec: ChannelSpec, sea_state: float, rng: np.random.Generator
) -> float:
    """Log-distance path loss with log-normal shadowing, over-sea flavour.

    PL(d) = FSPL(d0=1km) + 10*n*log10(d/d0) + X_sigma

    - n is the channel's path-loss exponent (over-sea LOS ~2, more with obstruction).
    - X_sigma ~ N(0, sigma) is log-normal shadowing; sigma grows with sea state,
      capturing rougher-sea multipath/ducting variability.
    """
    d0 = 1.0  # reference distance in km
    pl_d0 = free_space_path_loss_db(d0, spec.freq_mhz)
    d = max(distance_km, d0)
    mean_pl = pl_d0 + 10.0 * spec.path_loss_exp * np.log10(d / d0)
    # Shadowing standard deviation: 4 dB baseline, up to ~8 dB in high sea state.
    sigma = 4.0 + 0.45 * sea_state
    shadowing = rng.normal(0.0, sigma)
    return mean_pl + shadowing


def noise_floor_dbm(spec: ChannelSpec) -> float:
    """Thermal noise floor including receiver noise figure.

    N(dBm) = -174 + 10*log10(B_Hz) + NF
    """
    return PHYS["thermal_noise_dbm_per_hz"] + 10.0 * np.log10(spec.bandwidth_hz) + spec.noise_figure_db


def interference_dbm(spec: ChannelSpec, traffic_density: float, rng: np.random.Generator) -> float:
    """Aggregate co-channel interference power in dBm.

    Modelled as a floor that rises with local traffic density (more vessels ->
    more co-channel emissions). Narrowband maritime VHF channels are more
    congestion-limited than wideband cellular/satellite, so we scale the effect
    by bandwidth (wider band -> interference is relatively smaller per Hz).
    """
    # Base interference sits a bit above the noise floor; density pushes it up.
    base = noise_floor_dbm(spec) + 3.0
    # Congestion penalty (dB) grows with density; damped for wideband channels.
    band_factor = np.clip(1.0 - np.log10(spec.bandwidth_hz / 25e3) * 0.15, 0.2, 1.0)
    congestion_db = 12.0 * traffic_density * band_factor
    jitter = rng.normal(0.0, 1.5)
    return base + congestion_db + jitter


def per_from_sinr(sinr_db: float, sinr_50: float, steepness: float = 0.7) -> float:
    """Packet error rate via a logistic 'waterfall' curve.

    PER = 1 / (1 + exp(steepness * (SINR - SINR_50)))

    - At SINR == SINR_50, PER = 0.5 (the waterfall midpoint).
    - As SINR grows, PER -> ~0; as it drops, PER -> ~1.
    A small floor keeps PER strictly positive (residual errors always possible).
    """
    per = 1.0 / (1.0 + np.exp(steepness * (sinr_db - sinr_50)))
    return float(np.clip(per, 1e-4, 1.0))


def throughput_kbps_from_sinr(sinr_db: float, per: float, spec: ChannelSpec) -> float:
    """Effective throughput via Shannon capacity, MCS-capped and PER-derated.

    C = B * log2(1 + SNR_linear)          (bits/s)
    throughput = min(C, MCS cap) * (1 - PER)

    SNR_linear is derived from SINR (dB). The MCS cap reflects the protocol/modem
    ceiling; the (1-PER) factor accounts for goodput lost to retransmission/errors.
    """
    snr_linear = 10.0 ** (sinr_db / 10.0)
    capacity_kbps = spec.bandwidth_hz * np.log2(1.0 + max(snr_linear, 1e-6)) / 1e3
    capped = min(capacity_kbps, spec.max_rate_kbps)
    return float(max(capped * (1.0 - per), 0.0))


def channel_qos(
    spec: ChannelSpec,
    distance_km: float,
    sea_state: float,
    weather: float,
    traffic_density: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compute the full (complete-information) QoS tuple for one channel.

    Returns a dict with rssi_dbm, sinr_db, per, throughput_kbps.

    Satellite is modelled specially: because a GEO/VSAT link budget is essentially
    independent of vessel-to-shore distance, its RSSI/SINR are near-constant and
    dominated by rain fade (weather), rather than by the log-distance model. This
    is more defensible than pretending vessel-shore distance drives a satellite link.
    """
    if spec.is_satellite:
        # Near-constant received level; degraded primarily by weather (rain fade).
        nominal_rssi = -95.0
        rain_fade_db = spec.weather_sensitivity * weather
        rssi = nominal_rssi - rain_fade_db + rng.normal(0.0, 1.0)
        noise = noise_floor_dbm(spec)
        sinr = rssi - noise  # satellite interference is negligible in this model
        sinr_50 = 4.0        # [CALIBRATE] waterfall midpoint for the satellite modem
    else:
        pl = log_distance_path_loss_db(distance_km, spec, sea_state, rng)
        # Extra weather-driven attenuation (small for VHF, larger for high-freq links).
        pl += spec.weather_sensitivity * weather
        rssi = spec.tx_power_dbm + spec.tx_gain_db + spec.rx_gain_db - pl
        noise = noise_floor_dbm(spec)
        interf = interference_dbm(spec, traffic_density, rng)
        # Combine noise and interference in the LINEAR domain, then back to dB.
        n_lin = 10.0 ** (noise / 10.0)
        i_lin = 10.0 ** (interf / 10.0)
        ni_dbm = 10.0 * np.log10(n_lin + i_lin)
        sinr = rssi - ni_dbm
        sinr_50 = 6.0  # [CALIBRATE] waterfall midpoint for terrestrial modems

    # Clip to physically realistic ranges: real receivers saturate and cannot
    # report unbounded SINR/RSSI. These caps also prevent the Shannon term from
    # exploding. [CALIBRATE] the caps to the receiver dynamic range in the paper.
    rssi = float(np.clip(rssi, -140.0, -20.0))
    sinr = float(np.clip(sinr, -20.0, 40.0))

    per = per_from_sinr(sinr, sinr_50)
    tput = throughput_kbps_from_sinr(sinr, per, spec)
    return {
        "rssi_dbm": float(rssi),
        "sinr_db": float(sinr),
        "per": per,
        "throughput_kbps": tput,
    }


def channel_available(spec: ChannelSpec, distance_km: float, weather: float, rng: np.random.Generator) -> int:
    """Physical reachability flag (1/0), independent of measurement missingness.

    - Terrestrial channels are available within their coverage radius, with a soft
      random edge (fade near the boundary) to avoid a hard step.
    - Satellite is globally available except during rare severe-weather blackouts.

    IMPORTANT: availability != missingness. Availability answers "can this channel
    physically reach the network here?" (always observed). Missingness answers
    "did we manage to measure this channel's QoS?" (injected separately).
    """
    if spec.is_satellite:
        # Rare blackout probability rising with extreme weather (Ku-band rain outage).
        blackout_p = 0.03 + 0.35 * max(weather - 0.7, 0.0)
        return int(rng.random() > blackout_p)
    # Soft coverage edge: probability of availability decays around the boundary.
    edge = (distance_km - spec.coverage_km) / (0.15 * spec.coverage_km)
    p_available = 1.0 / (1.0 + np.exp(edge))  # ~1 well inside, ~0 well outside
    return int(rng.random() < p_available)


# ----------------------------------------------------------------------------- #
#  2. ORACLE POLICY (label generator, uses COMPLETE information)
# ----------------------------------------------------------------------------- #
def oracle_label(
    qos_by_channel: dict[str, dict[str, float]],
    availability: dict[str, int],
    priority: int,
    msg_size_kb: float,
) -> int:
    """Select the best channel from COMPLETE QoS, or NO_CHANNEL if none is feasible.

    Steps:
      1. Feasibility filter: a channel is feasible if it is available AND meets the
         message's PER / throughput / latency requirements.
      2. Utility ranking: among feasible channels, maximise a weighted utility that
         trades off reliability (1-PER), throughput adequacy, latency, and cost.
         Weights depend on message priority (emergency ignores cost, weights
         reliability heavily; routine is cost-sensitive).
      3. If no channel is feasible, return NO_CHANNEL (the safety-relevant fallback).

    This deterministic policy is the ground-truth the learners must approximate
    from *incomplete* observations at inference time.
    """
    req = MSG_REQUIREMENTS[priority]
    # Size-dependent throughput requirement (see THROUGHPUT_PER_KB note above).
    req_throughput = req["base_throughput_kbps"] + THROUGHPUT_PER_KB * msg_size_kb
    best_label = NO_CHANNEL_LABEL
    best_score = -np.inf

    for idx, name in enumerate(CHANNEL_NAMES):
        if not availability[name]:
            continue
        q = qos_by_channel[name]
        spec = CHANNELS[name]

        # --- (1) feasibility filter -------------------------------------------
        if q["per"] > req["max_per"]:
            continue
        if q["throughput_kbps"] < req_throughput:
            continue
        if spec.latency_ms > req["max_latency_ms"]:
            continue

        # --- (2) utility score -------------------------------------------------
        reliability_term = req["w_reliability"] * (1.0 - q["per"])
        # Throughput adequacy saturates at 1 (extra headroom beyond requirement
        # gives diminishing returns).
        tput_ratio = min(q["throughput_kbps"] / req_throughput, 3.0) / 3.0
        throughput_term = req["w_throughput"] * tput_ratio
        latency_term = req["w_latency"] * (spec.latency_ms / req["max_latency_ms"])
        cost_term = req["w_cost"] * spec.cost_norm
        score = reliability_term + throughput_term - latency_term - cost_term

        if score > best_score:
            best_score = score
            best_label = idx

    return best_label


# ----------------------------------------------------------------------------- #
#  3. TRAJECTORY GENERATION
# ----------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    """Top-level generation configuration."""
    n_trajectories: int = 200          # number of vessel trajectories
    steps_min: int = 10                # min steps per trajectory
    steps_max: int = 30                # max steps per trajectory
    max_distance_km: float = 80.0      # farthest offshore distance considered
    # Priority sampling weights (routine, safety, emergency). Emergency is rare,
    # which deliberately induces class imbalance (motivating RUSBoost etc.).
    priority_weights: tuple = (0.75, 0.20, 0.05)
    seed: int = 0


def _random_walk_clip(value: float, step_sigma: float, lo: float, hi: float,
                      rng: np.random.Generator) -> float:
    """One step of a bounded Gaussian random walk (for smoothly-evolving context)."""
    return float(np.clip(value + rng.normal(0.0, step_sigma), lo, hi))


def generate_trajectory(traj_id: int, cfg: GenConfig, rng: np.random.Generator) -> list[dict]:
    """Generate one vessel trajectory as a list of per-step feature/label dicts.

    Environmental context (distance, sea state, weather, traffic density) evolves
    smoothly via bounded random walks, so that consecutive steps are correlated —
    the property that makes LOCF meaningful. A fresh message (with its priority) is
    drawn at every step.
    """
    n_steps = int(rng.integers(cfg.steps_min, cfg.steps_max + 1))

    # --- initialise slowly-varying context --------------------------------------
    # Start distance is biased toward near-shore (mode ~20 km), reflecting that most
    # VTS-relevant traffic operates near the coast where terrestrial channels reach.
    distance = float(rng.triangular(2.0, 20.0, cfg.max_distance_km))
    # Movement trend: inbound (-), outbound (+), or loitering (~0).
    trend = float(rng.choice([-1.0, 0.0, 1.0])) * rng.uniform(1.0, 4.0)
    sea_state = float(rng.uniform(0.0, 6.0))       # WMO-like 0..9 scale
    weather = float(rng.uniform(0.0, 0.9))         # normalized severity 0..1
    traffic = float(rng.uniform(0.0, 1.0))         # normalized local density 0..1
    hour = int(rng.integers(0, 24))

    rows = []
    for t in range(n_steps):
        # --- advance context one step -------------------------------------------
        distance = float(np.clip(distance + trend + rng.normal(0.0, 1.0), 1.0, cfg.max_distance_km))
        sea_state = _random_walk_clip(sea_state, 0.4, 0.0, 9.0, rng)
        weather = _random_walk_clip(weather, 0.05, 0.0, 1.0, rng)
        traffic = _random_walk_clip(traffic, 0.08, 0.0, 1.0, rng)
        hour = (hour + 1) % 24

        # --- draw the message for this step -------------------------------------
        # Priority determines the (right-skewed) size distribution: distress/safety
        # traffic is small (alerts), routine traffic can be large (data transfer).
        priority = int(rng.choice([0, 1, 2], p=cfg.priority_weights))
        _req = MSG_REQUIREMENTS[priority]
        msg_size_kb = float(np.round(
            np.clip(rng.exponential(_req["size_mean_kb"]), 0.1, _req["size_max_kb"]), 2))

        # --- compute COMPLETE QoS + availability for every channel --------------
        qos_by_channel: dict[str, dict[str, float]] = {}
        availability: dict[str, int] = {}
        for name, spec in CHANNELS.items():
            availability[name] = channel_available(spec, distance, weather, rng)
            qos_by_channel[name] = channel_qos(spec, distance, sea_state, weather, traffic, rng)
            # If physically unavailable, force QoS to an 'unusable' but OBSERVED state
            # (very low RSSI/SINR, PER~1, throughput~0). This is a known observation,
            # distinct from a *missing* measurement injected later.
            if not availability[name]:
                qos_by_channel[name] = {
                    "rssi_dbm": float(noise_floor_dbm(spec) - 10.0),
                    "sinr_db": -20.0,
                    "per": 1.0,
                    "throughput_kbps": 0.0,
                }

        # --- oracle label (from complete information) ---------------------------
        label = oracle_label(qos_by_channel, availability, priority, msg_size_kb)

        # --- assemble the flat feature row --------------------------------------
        row: dict[str, float] = {
            "traj_id": traj_id,
            "step": t,
            # 7 scenario attributes (ALWAYS OBSERVED)
            "distance_to_shore_km": distance,
            "sea_state": sea_state,
            "traffic_density": traffic,
            "weather_severity": weather,
            "msg_priority": priority,
            "msg_size_kb": msg_size_kb,
            "hour_of_day": hour,
        }
        # 20 per-channel QoS values (SUBJECT TO MISSINGNESS later)
        for name in CHANNEL_NAMES:
            for metric in QOS_METRICS:
                row[f"{name}__{metric}"] = qos_by_channel[name][metric]
        # 5 availability flags (ALWAYS OBSERVED)
        for name in CHANNEL_NAMES:
            row[f"{name}__available"] = availability[name]
        # target
        row["label"] = label
        rows.append(row)

    return rows


def generate_dataset(cfg: GenConfig) -> pd.DataFrame:
    """Generate a full dataset (many trajectories) for a single seed.

    The returned DataFrame is CLEAN (no missingness). Missingness is injected
    downstream by inject_mcar_missingness so that the same clean labels can be
    reused across every missingness rate in the robustness sweep.
    """
    rng = np.random.default_rng(cfg.seed)
    all_rows: list[dict] = []
    for traj_id in range(cfg.n_trajectories):
        all_rows.extend(generate_trajectory(traj_id, cfg, rng))
    df = pd.DataFrame(all_rows)
    df["seed"] = cfg.seed
    return df


# ----------------------------------------------------------------------------- #
#  4. MISSINGNESS (MCAR) AND LOCF IMPUTATION
# ----------------------------------------------------------------------------- #
def qos_columns() -> list[str]:
    """Return the 20 QoS column names (the ONLY columns subject to missingness)."""
    return [f"{name}__{metric}" for name in CHANNEL_NAMES for metric in QOS_METRICS]


def inject_mcar_missingness(df: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    """Inject Missing-Completely-At-Random (MCAR) missingness into the 20 QoS values.

    Each QoS cell is independently set to NaN with probability p. Scenario
    attributes and availability flags are NEVER masked (they are always observed).

    Returns a COPY; the input is left untouched so the clean labels remain intact.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("missingness rate p must be in [0, 1]")
    out = df.copy()
    if p == 0.0:
        return out
    rng = np.random.default_rng(seed)
    cols = qos_columns()
    mask = rng.random(size=(len(out), len(cols))) < p
    block = out[cols].to_numpy(dtype=float, copy=True)  # copy: pandas 3.0 may return a read-only view
    block[mask] = np.nan
    out[cols] = block
    return out


def inject_mar_missingness(df: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    """Inject Missing-At-Random (MAR) missingness into the 20 QoS values.

    Unlike MCAR, the per-cell missing probability depends on observed attributes,
    modelling the real maritime measurement process:
      - RSSI is a *passive* reading and is rarely missing; SINR is often derived and
        moderately missing; PER and throughput require *active* probing of the channel
        and are frequently missing or stale.
      - A channel that is currently unavailable (a_c = 0) is less likely to be probed,
        so its QoS is more often missing.
    The per-cell weights are normalized so that the overall expected missingness rate
    matches the nominal p, making MAR directly comparable to MCAR at the same p.
    Only the 20 QoS features are affected; scenario attributes and availability flags
    remain fully observed. Returns a COPY.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("missingness rate p must be in [0, 1]")
    out = df.copy()
    if p == 0.0:
        return out
    rng = np.random.default_rng(seed)

    # metric-dependent base weights (higher => more often missing)
    metric_weight = {"rssi_dbm": 0.3, "sinr_db": 0.8, "per": 1.6, "throughput_kbps": 1.6}
    cols = qos_columns()

    # Build the per-cell weight matrix from metric type and channel availability.
    W = np.zeros((len(out), len(cols)), dtype=float)
    for j, name in enumerate(CHANNEL_NAMES):
        avail = out[f"{name}__available"].to_numpy(dtype=float)  # 1 if reachable
        avail_mult = np.where(avail > 0.5, 1.0, 1.8)  # unavailable -> more often missing
        for m_idx, metric in enumerate(QOS_METRICS):
            col_idx = j * len(QOS_METRICS) + m_idx
            W[:, col_idx] = metric_weight[metric] * avail_mult

    # Normalize so E[missing] ~= p, then clip probabilities into [0, 1].
    prob = np.clip(p * W / W.mean(), 0.0, 1.0)
    mask = rng.random(size=W.shape) < prob
    block = out[cols].to_numpy(dtype=float, copy=True)
    block[mask] = np.nan
    out[cols] = block
    return out


def locf_impute(df_missing: pd.DataFrame) -> pd.DataFrame:
    """Last-Observation-Carried-Forward imputation, applied WITHIN each trajectory.

    This is the imputation used by the POLICY baseline: when a channel's QoS is
    missing, the most recent non-missing value along the same trajectory is carried
    forward. Trajectories are treated independently (no carry-over across vessels),
    and rows are ordered by 'step' before filling.

    Any cells still missing after LOCF are 'cold-start' cases (missing from the very
    first step, with no past value to borrow). These are LEFT AS NaN here; the
    policy baseline treats a channel with cold-start-missing QoS as non-selectable
    (the conservative, safety-oriented rule agreed in the design).
    """
    out = df_missing.sort_values(["traj_id", "step"]).copy()
    cols = qos_columns()
    out[cols] = out.groupby("traj_id")[cols].ffill()
    return out


def add_missingness_indicators(df_missing: pd.DataFrame, impute_values: Optional[dict] = None
                               ) -> tuple[pd.DataFrame, dict]:
    """Missingness-aware featurization for the LEARNERS (fair across all 5 models).

    Produces a 52-dimensional representation:
        - each of the 20 QoS columns is mean-imputed (using training statistics), and
        - a 0/1 indicator column '<qos>__isnan' is appended for each.
    All five classifiers therefore receive the SAME apples-to-apples input, which is
    required for a fair comparison (only the XGBoost-native-missing ABLATION deviates).

    Parameters
    ----------
    impute_values : dict or None
        Column -> fill value (compute this ONCE on the training split and reuse on
        the test split to avoid leakage). If None, means are computed from df_missing
        itself (use only for the training split).

    Returns (featurized_df, impute_values).
    """
    out = df_missing.copy()
    cols = qos_columns()
    if impute_values is None:
        impute_values = {c: float(out[c].mean(skipna=True)) for c in cols}
    for c in cols:
        out[f"{c}__isnan"] = out[c].isna().astype(int)
        out[c] = out[c].fillna(impute_values[c])
    return out, impute_values


# ----------------------------------------------------------------------------- #
#  5. TRAJECTORY-LEVEL TRAIN/TEST SPLIT (no temporal leakage)
# ----------------------------------------------------------------------------- #
def train_test_split_by_trajectory(df: pd.DataFrame, test_size: float = 0.3, seed: int = 0
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by trajectory id so that steps from one trajectory never straddle the
    train/test boundary (prevents leakage from temporally adjacent, correlated steps).
    """
    rng = np.random.default_rng(seed)
    traj_ids = df["traj_id"].unique()
    rng.shuffle(traj_ids)
    n_test = int(round(len(traj_ids) * test_size))
    test_ids = set(traj_ids[:n_test].tolist())
    test_mask = df["traj_id"].isin(test_ids)
    return df[~test_mask].copy(), df[test_mask].copy()


# ----------------------------------------------------------------------------- #
#  6. METADATA / FEATURE-SCHEMA EXPORT (for reproducibility & the paper)
# ----------------------------------------------------------------------------- #
def feature_schema() -> dict:
    """Return a machine-readable description of the 32-dim feature layout + labels."""
    scenario = [
        "distance_to_shore_km", "sea_state", "traffic_density",
        "weather_severity", "msg_priority", "msg_size_kb", "hour_of_day",
    ]
    qos = qos_columns()
    availability = [f"{name}__available" for name in CHANNEL_NAMES]
    return {
        "n_features": len(scenario) + len(qos) + len(availability),
        "scenario_attributes": scenario,
        "qos_features_missable": qos,
        "availability_flags": availability,
        "label_map": LABELS,
        "channel_static_latency_ms": {n: CHANNELS[n].latency_ms for n in CHANNEL_NAMES},
    }


# ----------------------------------------------------------------------------- #
#  7. SCRIPT ENTRY POINT — generate one dataset and print sanity diagnostics
# ----------------------------------------------------------------------------- #
def _print_diagnostics(df: pd.DataFrame) -> None:
    """Print distribution sanity checks — the physically-motivated values should
    fall in realistic ranges, and every class (including NO_CHANNEL & emergency)
    should be represented."""
    print("=" * 70)
    print(f"Rows: {len(df)}   Trajectories: {df['traj_id'].nunique()}")
    print("-" * 70)
    print("Label distribution (channel selected by oracle):")
    counts = df["label"].value_counts().sort_index()
    for lbl, cnt in counts.items():
        print(f"  {lbl} {LABELS[int(lbl)]:>10}: {cnt:5d} ({100*cnt/len(df):5.1f}%)")
    print("-" * 70)
    print("Priority mix:")
    for p, cnt in df["msg_priority"].value_counts().sort_index().items():
        pname = {0: "routine", 1: "safety", 2: "emergency"}[int(p)]
        print(f"  {pname:>9}: {cnt:5d} ({100*cnt/len(df):5.1f}%)")
    print("-" * 70)
    print("QoS ranges (min / median / max):")
    for name in CHANNEL_NAMES:
        for metric in QOS_METRICS:
            c = f"{name}__{metric}"
            s = df[c]
            print(f"  {c:28s}: {s.min():9.2f} / {s.median():9.2f} / {s.max():9.2f}")
    print("-" * 70)
    print("Availability rate per channel:")
    for name in CHANNEL_NAMES:
        c = f"{name}__available"
        print(f"  {name:>10}: {100*df[c].mean():5.1f}%")
    print("=" * 70)


def main() -> None:
    # ---- generate a single-seed dataset (loop over seeds for the paper's runs) ----
    cfg = GenConfig(n_trajectories=200, seed=0)
    df = generate_dataset(cfg)

    _print_diagnostics(df)

    # ---- demonstrate the downstream robustness pipeline on one missingness rate ---
    p_demo = 0.25
    df_missing = inject_mcar_missingness(df, p=p_demo, seed=123)
    n_missing = df_missing[qos_columns()].isna().sum().sum()
    total_qos_cells = len(df_missing) * len(qos_columns())
    print(f"\nInjected MCAR missingness p={p_demo}: "
          f"{n_missing}/{total_qos_cells} QoS cells masked "
          f"({100*n_missing/total_qos_cells:.1f}%).")

    df_locf = locf_impute(df_missing)
    residual = df_locf[qos_columns()].isna().sum().sum()
    print(f"After LOCF (policy baseline): {residual} residual cold-start NaNs remain "
          f"(these channels are treated as non-selectable by the policy).")

    train_df, test_df = train_test_split_by_trajectory(df_missing, test_size=0.3, seed=0)
    train_feat, fill = add_missingness_indicators(train_df)          # fit fills on train
    test_feat, _ = add_missingness_indicators(test_df, impute_values=fill)  # reuse on test
    print(f"Learner featurization: {train_feat.shape[1]} columns "
          f"(includes 20 missingness indicators). "
          f"Train rows={len(train_feat)}, Test rows={len(test_feat)}.")

    # ---- persist artefacts --------------------------------------------------------
    out_csv = "../data/romacs_dataset_seed0_clean.csv"
    df.to_csv(out_csv, index=False)
    with open("../data/romacs_feature_schema.json", "w") as f:
        json.dump(feature_schema(), f, indent=2)
    print(f"\nSaved clean dataset -> {out_csv}")
    print("Saved feature schema -> ../data/romacs_feature_schema.json")


if __name__ == "__main__":
    main()
