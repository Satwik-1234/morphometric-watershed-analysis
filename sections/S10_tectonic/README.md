# S10 — Tectonic Activity Indices

> Computes IAT (Index of Active Tectonics) from AF, T, Vf, and Smf following El Hamdouni et al. (2008). Classifies each subbasin into Class 1–4 tectonic activity.

---

## Inputs
`gdf_sub, gdf_so, RASTERS['dem'], FACC_ARR`

## Key outputs
`tectonic_IAT.csv, 10a_tectonic_IAT_map.png, 10b_tectonic_radar.html`

## Key functions
- `compute_AF()`
- `compute_T()`
- `compute_Vf_at_outlet()`
- `compute_Smf()`
- `iat_class()`

---

## How to run

```python
# In Colab — ensure previous sections are in memory
exec(open('sections/S10_tectonic/script.py').read())
```

## Output Preview

![S10 — Tectonic Activity Indices](../../outputs/previews/10a_tectonic_IAT_map.png)

---

← [Back to main README](../../README.md)
