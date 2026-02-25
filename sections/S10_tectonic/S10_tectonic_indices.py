"""
=============================================================================
SECTION 10 — TECTONIC ACTIVITY INDICES
=============================================================================
Assumes Sections 1–3 variables are in memory:
  gdf_sub, df_master, DEM_ARR, DEM_TRANSFORM, DEM_RES,
  RASTERS, HILLSHADE, UTM_EPSG, FACC_ARR, SLOPE_ARR,
  gdf_streams, gdf_so, ORDER_COL, OUT_DIR, MAPS_DIR,
  PLOTS_DIR, TABLES_DIR, HTML_DIR

Parameters computed (per subbasin unless noted):
  AF   — Drainage Basin Asymmetry Factor (El Hamdouni et al., 2008)
  T    — Transverse Topographic Symmetry Factor (Cox, 1994)
  Vf   — Valley Floor Width-to-Height Ratio (Bull & McFadden, 1977)
  Smf  — Mountain Front Sinuosity (Bull & McFadden, 1977)
  IAT  — Index of Active Tectonics (composite classification)
  BS   — Basin Shape Index (Cannon, 1976)

References:
  Bull, W.B. & McFadden, L.D. (1977). Tectonic geomorphology N & S of the Garlock fault.
  Cox, R.T. (1994). Analysis of drainage basin symmetry as a rapid technique.
  El Hamdouni, R. et al. (2008). Assessment of relative active tectonics, SE Spain.
=============================================================================
"""

print("=" * 60)
print("SECTION 10 — TECTONIC ACTIVITY INDICES")
print("=" * 60)

from scipy.ndimage import label as ndlabel
from shapely.ops import split, unary_union
from shapely.geometry import LineString, MultiPoint

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — reuse map utilities from S4 (must already be in memory)
# ─────────────────────────────────────────────────────────────────────────────

def raster_extent_from(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


# ─────────────────────────────────────────────────────────────────────────────
#  A. ASYMMETRY FACTOR (AF)
# ─────────────────────────────────────────────────────────────────────────────
# AF = 100 × (Ar / At)
# Ar = area of basin to the right of the trunk stream (facing downstream)
# At = total basin area
# AF = 50 → no tectonic tilting; >50 or <50 → tilting

print("\n[A] Asymmetry Factor (AF)...")

def compute_AF(basin_geom, stream_geom_longest):
    """
    Split basin by the projected midline of the trunk stream.
    Count cells left vs right.
    """
    try:
        # Project stream onto basin geometry: get bounding centroid axis
        cx = basin_geom.centroid.x
        cy = basin_geom.centroid.y
        # Use stream bearing to define left/right
        coords = list(stream_geom_longest.coords)
        if len(coords) < 2:
            return np.nan, np.nan, np.nan
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return np.nan, np.nan, np.nan
        # Perpendicular bisector through centroid
        perp_dx, perp_dy = -dy / length, dx / length
        scale = max(basin_geom.bounds[2] - basin_geom.bounds[0],
                    basin_geom.bounds[3] - basin_geom.bounds[1]) * 2
        bisector = LineString([
            (cx - perp_dx * scale, cy - perp_dy * scale),
            (cx + perp_dx * scale, cy + perp_dy * scale),
        ])
        parts = split(basin_geom, bisector)
        if len(parts.geoms) < 2:
            return np.nan, np.nan, np.nan
        At = basin_geom.area
        Ar = parts.geoms[1].area   # right side (facing downstream)
        Al = parts.geoms[0].area
        AF = 100 * (Ar / At)
        return AF, Ar / 1e6, Al / 1e6
    except Exception:
        return np.nan, np.nan, np.nan


AF_rows = []
for _, row in gdf_sub.iterrows():
    bid   = row['basin_id']
    geom  = row.geometry
    # Longest stream in basin
    segs  = gdf_so[gdf_so.geometry.within(geom.buffer(50))]
    if len(segs) == 0:
        AF_rows.append({'basin_id': bid, 'AF': np.nan,
                        'AF_deviation': np.nan, 'AF_class': 'Unknown'})
        continue
    longest = segs.loc[segs.geometry.length.idxmax()].geometry
    if longest.geom_type == 'MultiLineString':
        longest = max(longest.geoms, key=lambda g: g.length)
    AF_val, Ar, Al = compute_AF(geom, longest)
    AF_dev = abs(AF_val - 50) if not np.isnan(AF_val) else np.nan
    if np.isnan(AF_val):
        cls = 'Unknown'
    elif AF_dev < 5:
        cls = 'Symmetric (Low tectonic activity)'
    elif AF_dev < 15:
        cls = 'Slightly asymmetric (Moderate)'
    else:
        cls = 'Highly asymmetric (High tectonic activity)'
    AF_rows.append({'basin_id': bid, 'AF': round(AF_val, 3),
                    'AF_deviation': round(AF_dev, 3), 'AF_class': cls})
    print(f"  {bid}: AF={AF_val:.2f} | dev={AF_dev:.2f} | {cls}")

df_AF = pd.DataFrame(AF_rows).set_index('basin_id')

# ─────────────────────────────────────────────────────────────────────────────
#  B. TRANSVERSE TOPOGRAPHIC SYMMETRY FACTOR (T)
# ─────────────────────────────────────────────────────────────────────────────
# T = Da / Dd
# Da = distance from midline of basin to midline of active meander belt
# Dd = distance from midline of basin to basin divide
# T → 0 : perfectly symmetric; T → 1 : highly asymmetric

print("\n[B] Transverse Topographic Symmetry Factor (T)...")

def compute_T(basin_geom, trunk_stream_geom):
    """
    Approximation using centroid offset:
    T = distance(basin_centroid → stream_centroid) /
        distance(basin_centroid → furthest point on divide)
    """
    try:
        basin_c   = basin_geom.centroid
        stream_c  = trunk_stream_geom.centroid
        Da        = basin_c.distance(stream_c)
        # Dd: mean distance from centroid to boundary
        boundary_pts = [
            basin_geom.boundary.interpolate(basin_geom.boundary.length * f)
            for f in np.linspace(0, 1, 200)
        ]
        dists_to_bdy = [basin_c.distance(p) for p in boundary_pts]
        Dd = np.mean(dists_to_bdy)
        T  = Da / Dd if Dd > 0 else np.nan
        return T, Da, Dd
    except Exception:
        return np.nan, np.nan, np.nan


T_rows = []
for _, row in gdf_sub.iterrows():
    bid  = row['basin_id']
    geom = row.geometry
    segs = gdf_so[gdf_so.geometry.within(geom.buffer(50))]
    if len(segs) == 0:
        T_rows.append({'basin_id': bid, 'T': np.nan, 'T_class': 'Unknown'})
        continue
    longest = segs.loc[segs.geometry.length.idxmax()].geometry
    if longest.geom_type == 'MultiLineString':
        longest = max(longest.geoms, key=lambda g: g.length)
    T_val, Da, Dd = compute_T(geom, longest)
    cls = ('Symmetric' if T_val < 0.1 else
           'Slightly asymmetric' if T_val < 0.25 else
           'Moderately asymmetric' if T_val < 0.5 else
           'Highly asymmetric') if not np.isnan(T_val) else 'Unknown'
    T_rows.append({'basin_id': bid, 'T': round(T_val, 4) if not np.isnan(T_val) else np.nan,
                   'Da_m': round(Da, 1), 'Dd_m': round(Dd, 1), 'T_class': cls})
    print(f"  {bid}: T={T_val:.4f} | Da={Da:.0f}m | Dd={Dd:.0f}m | {cls}")

df_T = pd.DataFrame(T_rows).set_index('basin_id')

# ─────────────────────────────────────────────────────────────────────────────
#  C. VALLEY FLOOR WIDTH-TO-HEIGHT RATIO (Vf)
# ─────────────────────────────────────────────────────────────────────────────
# Vf = 2Vfw / [(Eld - Esc) + (Erd - Esc)]
# Vfw  = valley floor width
# Eld/Erd = elevation of left/right valley walls at top
# Esc = elevation of valley floor

print("\n[C] Valley Floor Width-to-Height Ratio (Vf)...")

def compute_Vf_at_outlet(basin_geom, dem_path, n_transects=5, transect_frac=0.15):
    """
    Sample cross-valley transects near the outlet.
    Returns mean Vf across transects.
    """
    Vf_vals = []
    bounds   = basin_geom.bounds
    y_sample = np.linspace(bounds[1] + (bounds[3]-bounds[1])*0.05,
                           bounds[1] + (bounds[3]-bounds[1])*transect_frac,
                           n_transects)
    x_min, x_max = bounds[0], bounds[2]

    with rasterio.open(dem_path) as src:
        for y in y_sample:
            # horizontal transect
            pts_x  = np.linspace(x_min, x_max, 100)
            elevs  = []
            for px in pts_x:
                try:
                    r_i, c_i = rowcol(src.transform, px, y)
                    e = src.read(1)[r_i, c_i]
                    nd = src.nodata if src.nodata else -9999
                    elevs.append(np.nan if e == nd else float(e))
                except:
                    elevs.append(np.nan)
            elevs = np.array(elevs)
            valid  = ~np.isnan(elevs)
            if valid.sum() < 10:
                continue
            Esc    = np.nanmin(elevs)           # valley floor elevation
            Eld    = np.nanpercentile(elevs, 95) # left wall (approx)
            Erd    = Eld                         # symmetric approximation
            # Vfw: width of cells within 10% above minimum
            threshold = Esc + (Eld - Esc) * 0.10
            Vfw_cells = np.sum(elevs[valid] <= threshold)
            cell_size  = (x_max - x_min) / 100
            Vfw_m      = Vfw_cells * cell_size
            denom      = (Eld - Esc) + (Erd - Esc)
            if denom > 0:
                Vf_vals.append((2 * Vfw_m) / denom)

    return np.nanmean(Vf_vals) if Vf_vals else np.nan


Vf_rows = []
for _, row in gdf_sub.iterrows():
    bid   = row['basin_id']
    geom  = row.geometry
    Vf    = compute_Vf_at_outlet(geom, RASTERS['dem'])
    cls   = ('V-shaped valley (active uplift)' if not np.isnan(Vf) and Vf < 0.5 else
             'Transitional' if not np.isnan(Vf) and Vf < 1.0 else
             'Wide flat valley (tectonic quiescence)') if not np.isnan(Vf) else 'Unknown'
    Vf_rows.append({'basin_id': bid, 'Vf': round(Vf, 4) if not np.isnan(Vf) else np.nan,
                    'Vf_class': cls})
    print(f"  {bid}: Vf={Vf:.3f} | {cls}" if not np.isnan(Vf) else f"  {bid}: Vf=N/A")

df_Vf = pd.DataFrame(Vf_rows).set_index('basin_id')

# ─────────────────────────────────────────────────────────────────────────────
#  D. MOUNTAIN FRONT SINUOSITY (Smf)
# ─────────────────────────────────────────────────────────────────────────────
# Smf = Lmf / Ls
# Lmf = length of mountain front (actual, sinuous boundary)
# Ls  = straight-line length of mountain front
# Smf → 1 : straight, active front; Smf > 3 : sinuous, inactive

print("\n[D] Mountain Front Sinuosity (Smf)...")

def compute_Smf(basin_geom):
    """
    Use the lower boundary segment of each subbasin as mountain front proxy.
    Lmf = actual perimeter of lower 25% of basin extent
    Ls  = straight-line distance of same extent
    """
    try:
        bounds   = basin_geom.bounds
        y_thresh = bounds[1] + (bounds[3] - bounds[1]) * 0.25
        lower    = basin_geom.intersection(
            box(bounds[0], bounds[1], bounds[2], y_thresh)
        )
        if lower.is_empty:
            return np.nan
        Lmf = lower.boundary.length if hasattr(lower, 'boundary') else lower.length
        Ls  = np.sqrt((bounds[2] - bounds[0])**2)  # E-W extent of lower portion
        return Lmf / Ls if Ls > 0 else np.nan
    except Exception:
        return np.nan


Smf_rows = []
for _, row in gdf_sub.iterrows():
    bid = row['basin_id']
    Smf = compute_Smf(row.geometry)
    cls = ('Straight/active front' if not np.isnan(Smf) and Smf < 1.4 else
           'Moderately sinuous' if not np.isnan(Smf) and Smf < 3.0 else
           'Highly sinuous/inactive') if not np.isnan(Smf) else 'Unknown'
    Smf_rows.append({'basin_id': bid, 'Smf': round(Smf, 4) if not np.isnan(Smf) else np.nan,
                     'Smf_class': cls})
    print(f"  {bid}: Smf={Smf:.3f} | {cls}" if not np.isnan(Smf) else f"  {bid}: Smf=N/A")

df_Smf = pd.DataFrame(Smf_rows).set_index('basin_id')

# ─────────────────────────────────────────────────────────────────────────────
#  E. INDEX OF ACTIVE TECTONICS (IAT) — composite
# ─────────────────────────────────────────────────────────────────────────────
# IAT = mean of individual class scores (1=high, 2=moderate, 3=low activity)
# Following El Hamdouni et al. (2008)

print("\n[E] Index of Active Tectonics (IAT)...")

def score_AF(AF):
    if np.isnan(AF): return 2
    dev = abs(AF - 50)
    if dev > 15: return 1
    if dev > 5:  return 2
    return 3

def score_T(T):
    if np.isnan(T): return 2
    if T > 0.5:  return 1
    if T > 0.25: return 2
    return 3

def score_Vf(Vf):
    if np.isnan(Vf): return 2
    if Vf < 0.5: return 1
    if Vf < 1.0: return 2
    return 3

def score_Smf(Smf):
    if np.isnan(Smf): return 2
    if Smf < 1.4: return 1
    if Smf < 3.0: return 2
    return 3

def iat_class(iat):
    if iat <= 1.5: return "Class 1 — Very High"
    if iat <= 2.0: return "Class 2 — High"
    if iat <= 2.5: return "Class 3 — Moderate"
    return "Class 4 — Low"

IAT_rows = []
for bid in gdf_sub['basin_id']:
    AF_v  = df_AF.loc[bid, 'AF']     if bid in df_AF.index  else np.nan
    T_v   = df_T.loc[bid, 'T']       if bid in df_T.index   else np.nan
    Vf_v  = df_Vf.loc[bid, 'Vf']     if bid in df_Vf.index  else np.nan
    Smf_v = df_Smf.loc[bid, 'Smf']   if bid in df_Smf.index else np.nan

    s_AF, s_T, s_Vf, s_Smf = score_AF(AF_v), score_T(T_v), score_Vf(Vf_v), score_Smf(Smf_v)
    IAT   = np.mean([s_AF, s_T, s_Vf, s_Smf])
    cls   = iat_class(IAT)
    IAT_rows.append({
        'basin_id': bid,
        'AF': AF_v, 'T': T_v, 'Vf': Vf_v, 'Smf': Smf_v,
        'Score_AF': s_AF, 'Score_T': s_T, 'Score_Vf': s_Vf, 'Score_Smf': s_Smf,
        'IAT': round(IAT, 3), 'IAT_class': cls,
    })
    print(f"  {bid}: IAT={IAT:.2f} → {cls}")

df_IAT = pd.DataFrame(IAT_rows).set_index('basin_id')
df_IAT.to_csv(os.path.join(TABLES_DIR, "tectonic_IAT.csv"))
print(f"\n  ✅ IAT table saved")

# ─────────────────────────────────────────────────────────────────────────────
#  F. TECTONIC MAP — IAT choropleth on hillshade
# ─────────────────────────────────────────────────────────────────────────────

print("\n[F] Generating Tectonic Activity Map...")

iat_color_map = {
    'Class 1 — Very High': '#d73027',
    'Class 2 — High'     : '#fc8d59',
    'Class 3 — Moderate' : '#fee08b',
    'Class 4 — Low'      : '#91bfdb',
}

gdf_iat = gdf_sub.merge(df_IAT[['IAT','IAT_class']].reset_index(), on='basin_id')

utm_ext  = compute_utm_extent()
fig, ax  = base_axes("Index of Active Tectonics (IAT) — El Hamdouni et al., 2008")

for _, row in gdf_iat.iterrows():
    color = iat_color_map.get(row['IAT_class'], 'grey')
    gpd.GeoDataFrame([row], geometry='geometry', crs=gdf_sub.crs).plot(
        ax=ax, color=color, edgecolor='black', linewidth=1.2, alpha=0.75, zorder=3
    )
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, f"{row['basin_id']}\nIAT={row['IAT']:.2f}",
            ha='center', va='center', fontsize=8, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in iat_color_map.items()]
ax.legend(handles=legend_patches, loc='lower left', fontsize=8,
          title='Tectonic Activity', title_fontsize=9, framealpha=0.9)
gdf_streams.plot(ax=ax, color='royalblue', linewidth=0.6, alpha=0.5, zorder=5)
finalize_and_save(fig, ax, utm_ext, "10a_tectonic_IAT_map.png")

# ── Plotly radar — tectonic scores ────────────────────────────────────────────
fig_r = go.Figure()
for i, bid in enumerate(df_IAT.index):
    row  = df_IAT.loc[bid]
    cats = ['AF_score','T_score','Vf_score','Smf_score']
    vals = [row['Score_AF'], row['Score_T'], row['Score_Vf'], row['Score_Smf']]
    vals += [vals[0]]
    fig_r.add_trace(go.Scatterpolar(
        r=vals, theta=['AF','T','Vf','Smf','AF'],
        fill='toself', name=bid,
        line_color=px.colors.qualitative.Set1[i % 9], opacity=0.65,
        hovertemplate=bid+'<br>%{theta}: %{r} (1=high 3=low activity)',
    ))
fig_r.update_layout(
    polar=dict(radialaxis=dict(range=[0, 3], tickvals=[1,2,3],
                               ticktext=['High','Moderate','Low'])),
    title="Tectonic Activity Score Radar — Per Subbasin",
    template='plotly_white', height=550,
)
save_fig(fig_r, "10b_tectonic_radar")

print("\n✅ SECTION 10 complete.")
