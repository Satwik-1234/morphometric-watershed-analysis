# S3 — Morphometric Parameter Calculation

> **Runs on:** outputs of S2 (requires UTM-projected layers in memory)  
> **Produces:** `morphometric_master_table.csv` + section tables

---

## What this section computes

### A. Linear Aspects `[Horton 1945 · Strahler 1952]`

For each **stream order** within each **subbasin**:

```
Nu   = stream count per order
Lu   = total stream length (km)
Lsm  = Lu / Nu               (mean stream length)
RL   = Lsm(u) / Lsm(u-1)    (stream length ratio)
Rb   = Nu(u) / Nu(u+1)       (bifurcation ratio)
Rbm  = arithmetic mean of Rb
wRbm = weighted mean Rb (by stream count)
```

**Interpretation of Rbm:**
| Rbm | Geological Implication |
|-----|------------------------|
| 3–5 | Normal basin, no structural disturbance |
| < 3 | Permeable uniform lithology |
| > 5 | Strong structural control / faulting |

Horton's Law plots (log-scale stream number vs order, log-scale stream length vs order) are auto-generated with R² regression.

---

### B. Areal Aspects `[Schumm 1956 · Miller 1953]`

Per subbasin:

| Parameter | Symbol | Formula | Unit |
|-----------|--------|---------|------|
| Drainage Density | Dd | ΣL / A | km/km² |
| Stream Frequency | Fs | N / A | /km² |
| Texture Ratio | T | N / P | — |
| Form Factor | Ff | A / Lb² | — |
| Elongation Ratio | Re | (2/Lb)√(A/π) | — |
| Circularity Ratio | Rc | 4πA / P² | — |
| Compactness Coeff | Cc | P / (2√πA) | — |
| Overland Flow | Lg | 1/2Dd | km |
| Channel Maintenance | C | 1/Dd | km²/km |

**Shape classification (Re):**
- Re ≥ 0.9 → Circular
- Re 0.8–0.9 → Oval
- Re 0.7–0.8 → Less Elongated
- Re 0.5–0.7 → Elongated
- Re < 0.5 → More Elongated (flashy flood response)

---

### C. Relief Aspects `[Strahler 1964 · Melton 1965 · Riley 1999]`

Computed using zonal statistics on the clipped DEM per subbasin:

```
H    = Hmax - Hmin                           (Basin Relief, m)
Rh   = H / Lb                                (Relief Ratio)
Rn   = H × Dd                                (Ruggedness Number)
MRN  = H / √A                               (Melton Ruggedness)
HI   = (Hmean - Hmin) / H                   (Hypsometric Integral)
TRI  = √[Σ(xi - x0)²]    per 3×3 window     (Terrain Ruggedness)
```

**Hypsometric Integral interpretation:**
- HI > 0.60 → Monadnock/Young stage (active erosion)
- HI 0.35–0.60 → Mature/Equilibrium stage
- HI < 0.35 → Peneplain/Old stage

---

## Output files

| File | Path | Description |
|------|------|-------------|
| Master table | `outputs/tables/morphometric_master_table.csv` | All parameters, all basins |
| Stream order | `outputs/tables/stream_order_summary.csv` | Nu, Lu, Rb per order per basin |

---

## Sample output preview

After running, the master table looks like:

```
basin_id  Area_km2  Dd    Re    Rc    HI    Rn    Shape_Class   Hyps_Class
SB1       45.23     2.34  0.72  0.61  0.54  1.23  Less Elongated  Mature
SB2       31.87     3.12  0.58  0.45  0.68  2.01  Elongated       Monadnock
...
```

![Master Table Preview](../../outputs/previews/07_drainage_density.png)

---

## How to run (Colab)

```python
# Ensure S1 and S2 have been run first (variables in memory)
exec(open('sections/S3_morphometry/S3_morphometric_params.py').read())
```

## How to run (local)

```bash
python sections/S3_morphometry/S3_morphometric_params.py
```
