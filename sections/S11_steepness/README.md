# S11 — Channel Steepness & Concavity

> Hack's SL index, normalised steepness ksn (θref=0.45), concavity index θ from S-A regression, and chi (χ) coordinate profiles for drainage divide analysis.

---

## Inputs
`gdf_so_sub, RASTERS['dem'], RASTERS['flow_acc']`

## Key outputs
`SL_index_per_basin.csv, concavity_steepness.csv, ksn_per_basin.csv, 11a_SL_anomaly_map.png, 11b_ksn_map.png, 11c_chi_elevation_plot.html`

## Key functions
- `compute_SL_for_segments()`
- `compute_Vf_at_outlet()`
- `chi integration loop`

---

## How to run

```python
# In Colab — ensure previous sections are in memory
exec(open('sections/S11_steepness/script.py').read())
```

## Output Preview

![S11 — Channel Steepness & Concavity](../../outputs/previews/11a_SL_anomaly_map.png)

---

← [Back to main README](../../README.md)
