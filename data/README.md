# Benchmark data

The benchmark is *generated*, not stored in git: run `python src/romacs_datagen.py`
for a single-seed CSV, or `python src/romacs_experiment.py` / `romacs_v2_experiment.py`
which generate all 8 seeds in memory (600 trajectories per seed; 95,825 scenarios total).

Pre-generated CSVs for all 8 seeds are archived on **IEEE DataPort**: [DOI TBA].

## Feature schema (36 columns)

- `traj_id`, `step` — trajectory index and decision epoch
- 7 scenario attributes: `distance_to_shore_km`, `sea_state`, `traffic_density`,
  `weather_severity`, `msg_priority` (0 routine / 1 safety / 2 emergency),
  `msg_size_kb`, `hour_of_day`
- 20 QoS values: `{CH}__{rssi_dbm,sinr_db,per,throughput_kbps}` for
  CH ∈ {VHF_DSC, dPMR, AIS_VDES, LTE_5G, SATELLITE}
- 5 availability flags: `{CH}__available`
- `label` — oracle decision: 0–4 = channel index, 5 = `no_channel`

Degradation is applied downstream: MCAR/MAR masking (`inject_mcar_missingness`,
`romacs_mar_check.py`) or staleness via per-channel Bernoulli probing
(`inject_staleness` in `romacs_v2_experiment.py`), which adds `{CH}__age` AoI columns.

License: CC BY 4.0.
