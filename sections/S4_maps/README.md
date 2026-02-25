# S4 — Publication-Grade Map Generation

> Generates 9 publication maps, all with hillshade background, DMS (°′″) grid, north arrow, and scale bar.

---

## Inputs
`DEM_ARR, SLOPE_ARR, ASPECT_ARR, FDIR_ARR, FACC_ARR, gdf_so, gdf_sub, gdf_streams`

## Key outputs
`01_elevation.png, 02_slope.png, 03_aspect.png, 04_flow_direction.png, 05_flow_accumulation.png, 06_stream_order.png, 07_drainage_density.png, 08_contour.png, 09_pour_points.png`

## Key functions
- `base_axes()`
- `apply_dms_grid()`
- `add_north_arrow()`
- `add_scale_bar()`
- `finalize_and_save()`

---

## How to run

```python
# In Colab — ensure previous sections are in memory
exec(open('sections/S4_maps/script.py').read())
```

## Output Preview

![S4 — Publication-Grade Map Generation](../../outputs/previews/01_elevation.png)

---

← [Back to main README](../../README.md)
