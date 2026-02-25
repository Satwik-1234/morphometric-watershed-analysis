# S12 — Geomorphic Anomaly & Lineament Analysis

> Sinuosity Index per segment, GAI composite raster (SL×0.5 + TRI×0.3 + TWI⁻¹×0.2), structural lineament detection via Sobel edge + Probabilistic Hough transform.

---

## Inputs
`gdf_SL, TRI_ARR, TWI_ARR (from S11)`

## Key outputs
`sinuosity_per_basin.csv, GAI_per_basin.csv, 12a_GAI_map.png, 12b_lineament_proxy_map.png, 12c_sinuosity_map.png, lineament_proxy.shp`

## Key functions
- `compute_sinuosity()`
- `compute_GAI()`
- `canny()`
- `probabilistic_hough_line()`

---

## How to run

```python
# In Colab — ensure previous sections are in memory
exec(open('sections/S12_anomaly/script.py').read())
```

## Output Preview

![S12 — Geomorphic Anomaly & Lineament Analysis](../../outputs/previews/12a_GAI_map.png)

---

← [Back to main README](../../README.md)
