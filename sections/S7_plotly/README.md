# S7 — Plotly Interactive Visualization Suite

> 12 fully interactive HTML charts: Horton's Laws, radar, scatter matrix, 3D scatter, histograms, box plots, hypsometric curves, correlation heatmap, PCA biplot, parallel coordinates, bubble plot, longitudinal profiles.

---

## Inputs
`df_master, LINEAR_PER_ORDER, HYPS, CHI_PROFILES (from earlier sections)`

## Key outputs
`12 HTML files in outputs/html/`

## Key functions
- `save_fig()`
- `go.Scatterpolar()`
- `go.Scatter3d()`
- `px.parallel_coordinates()`

---

## How to run

```python
# In Colab — ensure previous sections are in memory
exec(open('sections/S7_plotly/script.py').read())
```

---

← [Back to main README](../../README.md)
