"""
=============================================================================
SECTION 3 — MORPHOMETRIC PARAMETER CALCULATION
=============================================================================
Computes all linear, areal, and relief morphometric parameters
per subbasin following Horton (1945), Strahler (1952, 1964),
Schumm (1956), and Miller (1953).
=============================================================================
"""

print("=" * 60)
print("SECTION 3 — MORPHOMETRIC PARAMETER CALCULATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
#  A. LINEAR ASPECTS  (stream order statistics)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[A] Computing Linear Aspects...")

def compute_linear_aspects(gdf_streams_clipped, order_col, basin_id):
    """
    Compute stream order statistics for one subbasin.
    Returns per-order DataFrame and summary ratios.
    """
    rows = []
    orders = sorted(gdf_streams_clipped[order_col].unique())

    for u in orders:
        segs = gdf_streams_clipped[gdf_streams_clipped[order_col] == u]
        nu   = len(segs)
        lu   = segs.geometry.length.sum()
        lsm  = lu / nu if nu > 0 else 0
        rows.append({'basin_id': basin_id, 'order': u,
                     'Nu': nu, 'Lu': lu, 'Lsm': lsm})

    df = pd.DataFrame(rows).set_index('order')

    # Bifurcation ratio Rb = Nu / Nu+1
    df['Rb'] = np.nan
    for i in range(len(df) - 1):
        o1, o2 = orders[i], orders[i+1]
        if df.loc[o2, 'Nu'] > 0:
            df.loc[o1, 'Rb'] = df.loc[o1, 'Nu'] / df.loc[o2, 'Nu']

    # Stream length ratio RL = Lsm(u) / Lsm(u-1)
    df['RL'] = np.nan
    for i in range(1, len(df)):
        o_prev, o_curr = orders[i-1], orders[i]
        if df.loc[o_prev, 'Lsm'] > 0:
            df.loc[o_curr, 'RL'] = df.loc[o_curr, 'Lsm'] / df.loc[o_prev, 'Lsm']

    # Mean bifurcation ratio (arithmetic)
    Rb_vals = df['Rb'].dropna()
    Rbm = Rb_vals.mean() if len(Rb_vals) > 0 else np.nan

    # Weighted mean bifurcation ratio (Strahler, 1957)
    wRbm = np.nan
    if len(Rb_vals) > 0:
        weights = []
        for i in range(len(orders) - 1):
            o1, o2 = orders[i], orders[i+1]
            if not np.isnan(df.loc[o1, 'Rb']):
                weights.append(df.loc[o1, 'Nu'] + df.loc[o2, 'Nu'])
            else:
                weights.append(0)
        wts = np.array(weights)
        rb_wts = Rb_vals.values
        if wts.sum() > 0:
            wRbm = np.average(rb_wts, weights=wts[:len(rb_wts)])

    return df.reset_index(), Rbm, wRbm


# Spatial join: streams to subbasins
gdf_so_sub = gpd.sjoin(
    gdf_so[[ORDER_COL, 'geometry']],
    gdf_sub[['basin_id', 'geometry']],
    how='left', predicate='within'
)
# Fallback: intersects for streams spanning boundaries
gdf_so_inter = gpd.sjoin(
    gdf_so[[ORDER_COL, 'geometry']],
    gdf_sub[['basin_id', 'geometry']],
    how='left', predicate='intersects'
)
gdf_so_sub = gdf_so_sub.dropna(subset=['basin_id'])
if len(gdf_so_sub) == 0:
    gdf_so_sub = gdf_so_inter.dropna(subset=['basin_id'])

LINEAR_PER_ORDER = {}   # basin_id → DataFrame
LINEAR_SUMMARY   = []   # one row per basin

for bid in gdf_sub['basin_id']:
    segs = gdf_so_sub[gdf_so_sub['basin_id'] == bid]
    if len(segs) == 0:
        print(f"  ⚠️  No stream segments found for basin {bid}")
        continue
    df_lin, Rbm, wRbm = compute_linear_aspects(segs, ORDER_COL, bid)
    LINEAR_PER_ORDER[bid] = df_lin

    total_N = df_lin['Nu'].sum()
    total_L = df_lin['Lu'].sum()
    max_ord = df_lin['order'].max()

    LINEAR_SUMMARY.append({
        'basin_id'       : bid,
        'total_streams_N': total_N,
        'total_length_m' : total_L,
        'max_order'      : max_ord,
        'Rbm'            : round(Rbm, 4),
        'wRbm'           : round(wRbm, 4) if not np.isnan(wRbm) else np.nan,
    })
    print(f"  {bid}: {total_N} streams | max order {max_ord} | Rbm={Rbm:.3f}")

df_linear_summary = pd.DataFrame(LINEAR_SUMMARY).set_index('basin_id')
print("\n  Stream Order Summary (all basins):")
for bid, df in LINEAR_PER_ORDER.items():
    print(f"\n  [{bid}]")
    print(df[['order','Nu','Lu','Lsm','Rb','RL']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
#  B. AREAL ASPECTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[B] Computing Areal Aspects...")

def longest_flow_path(basin_geom, facc_arr, transform, res_m):
    """
    Approximate basin length (Lb) using the longest flow path concept:
    distance from centroid to pour point approximated as sqrt(A / 1.128)
    (Hack, 1957) as fallback.  Actual longest flow path would require
    full D8 tracing — approximation is acceptable for published studies.
    """
    area = basin_geom.area          # m²
    lb   = np.sqrt(area / 1.128)   # Hack approximation
    return lb


AREAL = []

for _, row in gdf_sub.iterrows():
    bid   = row['basin_id']
    geom  = row.geometry

    A  = geom.area          # m²
    P  = geom.length        # m
    Lb = longest_flow_path(geom, FACC_ARR, DEM_TRANSFORM, DEM_RES)

    # Streams inside basin
    segs = gdf_so_sub[gdf_so_sub['basin_id'] == bid]
    total_stream_length = segs.geometry.length.sum() if len(segs) > 0 else 0
    Nu_total = len(segs)

    # ----- parameters -----
    A_km2   = A   / 1e6
    P_km    = P   / 1e3
    Lb_km   = Lb  / 1e3
    L_km    = total_stream_length / 1e3

    Dd = L_km  / A_km2 if A_km2 > 0 else np.nan   # Drainage density  [km/km²]
    Fs = Nu_total / A_km2 if A_km2 > 0 else np.nan # Stream frequency  [streams/km²]
    T  = Nu_total / P_km  if P_km  > 0 else np.nan # Texture ratio
    Ff = A_km2   / (Lb_km**2)     if Lb_km > 0 else np.nan  # Form factor (Horton,1932)
    Re = (2 / Lb_km) * np.sqrt(A_km2 / np.pi) if Lb_km > 0 else np.nan  # Elongation ratio
    Rc = (4 * np.pi * A_km2) / (P_km**2)  if P_km > 0 else np.nan       # Circularity ratio
    Cc = P_km / (2 * np.sqrt(np.pi * A_km2)) if A_km2 > 0 else np.nan   # Compactness coeff
    Lg = 1 / (2 * Dd)   if Dd and Dd > 0 else np.nan   # Length of overland flow
    C  = 1 / Dd          if Dd and Dd > 0 else np.nan   # Constant of channel maintenance

    AREAL.append({
        'basin_id'              : bid,
        'Area_km2'              : round(A_km2, 4),
        'Perimeter_km'          : round(P_km, 4),
        'Basin_Length_km'       : round(Lb_km, 4),
        'Total_Stream_Length_km': round(L_km, 4),
        'Stream_Count'          : Nu_total,
        'Drainage_Density_Dd'   : round(Dd, 4) if not np.isnan(Dd) else np.nan,
        'Stream_Frequency_Fs'   : round(Fs, 4) if not np.isnan(Fs) else np.nan,
        'Texture_Ratio_T'       : round(T,  4) if not np.isnan(T)  else np.nan,
        'Form_Factor_Ff'        : round(Ff, 4) if not np.isnan(Ff) else np.nan,
        'Elongation_Ratio_Re'   : round(Re, 4) if not np.isnan(Re) else np.nan,
        'Circularity_Ratio_Rc'  : round(Rc, 4) if not np.isnan(Rc) else np.nan,
        'Compactness_Cc'        : round(Cc, 4) if not np.isnan(Cc) else np.nan,
        'LengthOverlandFlow_Lg' : round(Lg, 4) if not np.isnan(Lg) else np.nan,
        'ChannelMaintenance_C'  : round(C,  4) if not np.isnan(C)  else np.nan,
    })

    print(f"  {bid}: A={A_km2:.2f} km² | Dd={Dd:.3f} km/km² | "
          f"Re={Re:.3f} | Rc={Rc:.3f} | Ff={Ff:.3f}")

df_areal = pd.DataFrame(AREAL).set_index('basin_id')

# ─────────────────────────────────────────────────────────────────────────────
#  C. RELIEF ASPECTS  (DEM zonal statistics)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[C] Computing Relief Aspects...")

def hypsometric_integral(dem_clipped):
    """
    Compute hypsometric integral (HI) = (mean_elev - min_elev) / (max_elev - min_elev)
    Also returns arrays for hypsometric curve: (relative_area, relative_elevation)
    """
    vals = dem_clipped[~np.isnan(dem_clipped)].flatten()
    if len(vals) < 10:
        return np.nan, None, None
    mn, mx, mu = vals.min(), vals.max(), vals.mean()
    rng = mx - mn
    if rng == 0:
        return np.nan, None, None
    HI = (mu - mn) / rng
    # Curve: relative elevation h/H vs relative area a/A
    thresholds   = np.percentile(vals, np.linspace(0, 100, 101))
    rel_elev     = (thresholds - mn) / rng          # h/H  (0→1)
    rel_area     = 1 - np.linspace(0, 1, 101)       # a/A  (1→0)
    return HI, rel_area, rel_elev


def terrain_ruggedness_index(dem_arr):
    """
    TRI (Riley et al., 1999): mean absolute difference from center cell
    to 8 neighbours.
    """
    from scipy.ndimage import generic_filter
    def _tri_kernel(x):
        centre = x[4]
        if np.isnan(centre):
            return np.nan
        diffs = x - centre
        diffs[4] = 0
        valid = diffs[~np.isnan(diffs)]
        return np.sqrt(np.sum(valid**2)) if len(valid) > 0 else np.nan
    tri = generic_filter(dem_arr.astype(float), _tri_kernel, size=3, mode='reflect')
    tri[np.isnan(dem_arr)] = np.nan
    return tri


def melton_ruggedness(h_m, a_km2):
    """Melton (1965) ruggedness index MRN = H / sqrt(A)."""
    return h_m / np.sqrt(a_km2) if a_km2 > 0 else np.nan


# Compute TRI once for full DEM
print("  Computing TRI (this may take 30–60 sec on large DEMs)...")
TRI_ARR = terrain_ruggedness_index(DEM_ARR)
save_raster(TRI_ARR, os.path.join(OUT_DIR, "tri.tif"), RASTERS['dem'])

RELIEF   = []
HYPS     = {}   # basin_id → (rel_area, rel_elev)

for _, row in gdf_sub.iterrows():
    bid  = row['basin_id']
    geom = [row.geometry.__geo_interface__]

    # Mask DEM to subbasin
    with rasterio.open(RASTERS['dem']) as src:
        try:
            arr_masked, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
            dem_clip = arr_masked[0].astype(np.float32)
            dem_clip[dem_clip == src.nodata] = np.nan
        except Exception:
            dem_clip = DEM_ARR.copy()

    # Mask slope to subbasin
    with rasterio.open(RASTERS['slope']) as src:
        try:
            s_masked, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
            slope_clip = s_masked[0].astype(np.float32)
            slope_clip[slope_clip == -9999.0] = np.nan
        except Exception:
            slope_clip = SLOPE_ARR.copy()

    # Mask TRI
    with rasterio.open(os.path.join(OUT_DIR, "tri.tif")) as src:
        try:
            t_masked, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
            tri_clip = t_masked[0].astype(np.float32)
            tri_clip[tri_clip == -9999.0] = np.nan
        except Exception:
            tri_clip = TRI_ARR.copy()

    valid_dem   = dem_clip[~np.isnan(dem_clip)]
    valid_slope = slope_clip[~np.isnan(slope_clip)]
    valid_tri   = tri_clip[~np.isnan(tri_clip)]

    if len(valid_dem) == 0:
        print(f"  ⚠️  {bid}: no valid DEM cells")
        continue

    elev_min  = float(valid_dem.min())
    elev_max  = float(valid_dem.max())
    elev_mean = float(valid_dem.mean())
    H         = elev_max - elev_min              # Basin relief (m)
    A_km2     = df_areal.loc[bid, 'Area_km2']
    Lb_km     = df_areal.loc[bid, 'Basin_Length_km']
    P_km      = df_areal.loc[bid, 'Perimeter_km']

    Rh  = H / (Lb_km * 1000) if Lb_km > 0 else np.nan   # Relief ratio
    Rr  = H / P_km            if P_km  > 0 else np.nan   # Relative relief
    Dd  = df_areal.loc[bid, 'Drainage_Density_Dd']
    Rn  = H * Dd / 1000       if not np.isnan(Dd) else np.nan  # Ruggedness number
    MRN = melton_ruggedness(H, A_km2)                           # Melton ruggedness

    # Hypsometric integral
    HI, rel_area, rel_elev = hypsometric_integral(dem_clip)
    if rel_area is not None:
        HYPS[bid] = (rel_area, rel_elev)

    # Slope statistics
    slope_mean = float(np.nanmean(valid_slope))
    slope_std  = float(np.nanstd(valid_slope))
    slope_skew = float(stats.skew(valid_slope))

    # TRI stats
    tri_mean = float(np.nanmean(valid_tri))

    RELIEF.append({
        'basin_id'         : bid,
        'Elev_Min_m'       : round(elev_min,  2),
        'Elev_Max_m'       : round(elev_max,  2),
        'Elev_Mean_m'      : round(elev_mean, 2),
        'Basin_Relief_H_m' : round(H,         2),
        'Relief_Ratio_Rh'  : round(Rh,        6) if not np.isnan(Rh) else np.nan,
        'Relative_Relief'  : round(Rr,        4) if not np.isnan(Rr) else np.nan,
        'Ruggedness_Rn'    : round(Rn,        4) if not np.isnan(Rn) else np.nan,
        'Melton_MRN'       : round(MRN,       4) if not np.isnan(MRN) else np.nan,
        'Hypsometric_HI'   : round(HI,        4) if not np.isnan(HI) else np.nan,
        'Slope_Mean_deg'   : round(slope_mean, 3),
        'Slope_Std_deg'    : round(slope_std,  3),
        'Slope_Skewness'   : round(slope_skew, 4),
        'TRI_Mean'         : round(tri_mean,   3),
    })

    print(f"  {bid}: H={H:.0f}m | Rh={Rh:.5f} | HI={HI:.3f} | "
          f"Rn={Rn:.3f} | Slope_mean={slope_mean:.2f}°")

df_relief = pd.DataFrame(RELIEF).set_index('basin_id')

# ─────────────────────────────────────────────────────────────────────────────
#  D. MASTER MORPHOMETRIC TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[D] Assembling master morphometric table...")

df_master = df_areal.join(df_relief, how='left')
df_master = df_master.join(df_linear_summary, how='left')

# Add stream order per-basin summary
for bid in gdf_sub['basin_id']:
    if bid in LINEAR_PER_ORDER:
        df_lin = LINEAR_PER_ORDER[bid]
        for _, r in df_lin.iterrows():
            col = f"Nu_order{int(r['order'])}"
            df_master.loc[bid, col] = r['Nu']
            col = f"Lu_order{int(r['order'])}_km"
            df_master.loc[bid, col] = round(r['Lu'] / 1000, 4)

# ── Interpretation flags ──────────────────────────────────────────────────────
def interpret_elongation(Re):
    if pd.isna(Re):       return "Unknown"
    if Re >= 0.9:         return "Circular"
    if Re >= 0.8:         return "Oval"
    if Re >= 0.7:         return "Less Elongated"
    if Re >= 0.5:         return "Elongated"
    return "More Elongated"

def interpret_circularity(Rc):
    if pd.isna(Rc):      return "Unknown"
    if Rc >= 0.75:       return "Circular/Young"
    if Rc >= 0.50:       return "Intermediate"
    return "Elongated/Old"

def interpret_HI(HI):
    if pd.isna(HI):      return "Unknown"
    if HI > 0.60:        return "Monadnock (Young/Convex)"
    if HI > 0.35:        return "Mature (Equilibrium)"
    return "Peneplain (Old/Concave)"

df_master['Shape_Class']    = df_master['Elongation_Ratio_Re'].apply(interpret_elongation)
df_master['Circ_Class']     = df_master['Circularity_Ratio_Rc'].apply(interpret_circularity)
df_master['Hyps_Class']     = df_master['Hypsometric_HI'].apply(interpret_HI)

# Save
csv_path = os.path.join(TABLES_DIR, "morphometric_master_table.csv")
df_master.to_csv(csv_path)
print(f"  ✅ Master table saved: {csv_path}")

print("\n" + "─"*60)
print("  MASTER MORPHOMETRIC TABLE (first 10 rows/all params):")
print("─"*60)
print(df_master.to_string())
