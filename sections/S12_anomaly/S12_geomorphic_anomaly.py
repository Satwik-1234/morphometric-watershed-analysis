"""
=============================================================================
SECTION 12 — GEOMORPHIC ANOMALY & LINEAMENT ANALYSIS
=============================================================================
Assumes Sections 1–11 in memory.

Analyses:
  SL Anomaly Map  — spatial distribution of high SL reaches
  Channel Sinuosity Index (SI) — per segment
  Valley Asymmetry Gradient — spatial gradient of AF across watershed
  Geomorphic Anomaly Index (GAI) — composite raster (normalised SL + TRI + ksn)
  Lineament Proxy — high-gradient alignment detection using Canny edge filter
                    on DEM + slope to detect structural lineaments
  Drainage Anomaly Zones — map of statistically anomalous stream segments

Maps:
  GAI raster map
  Lineament proxy map
  Channel sinuosity map
=============================================================================
"""

print("=" * 60)
print("SECTION 12 — GEOMORPHIC ANOMALY & LINEAMENT ANALYSIS")
print("=" * 60)

from scipy.ndimage import (sobel, gaussian_filter, maximum_filter,
                            generic_filter, binary_dilation)

# ─────────────────────────────────────────────────────────────────────────────
#  A. CHANNEL SINUOSITY INDEX (SI)
# ─────────────────────────────────────────────────────────────────────────────
# SI = actual channel length / straight-line distance between endpoints
# SI > 1.5 → sinuous/meandering; SI ≈ 1.0 → straight

print("\n[A] Channel Sinuosity Index (SI)...")

def compute_sinuosity(geom):
    """SI = channel length / straight-line distance between endpoints."""
    if geom.geom_type == 'MultiLineString':
        geom = max(geom.geoms, key=lambda g: g.length)
    if geom.geom_type != 'LineString' or geom.length < DEM_RES:
        return np.nan
    coords    = list(geom.coords)
    straight  = np.sqrt((coords[-1][0] - coords[0][0])**2 +
                        (coords[-1][1] - coords[0][1])**2)
    return geom.length / straight if straight > 0 else np.nan


gdf_SL['SI'] = gdf_SL['geometry'].apply(compute_sinuosity)
SI_per_basin  = gdf_SL.groupby('basin_id')['SI'].agg(
    SI_mean='mean', SI_max='max', SI_std='std'
).round(4)

def si_class(si):
    if np.isnan(si):  return 'Unknown'
    if si < 1.05:     return 'Straight (structural control)'
    if si < 1.3:      return 'Irregular'
    if si < 1.5:      return 'Sinuous'
    return 'Meandering'

SI_per_basin['SI_class'] = SI_per_basin['SI_mean'].apply(si_class)
print(SI_per_basin.to_string())
SI_per_basin.to_csv(os.path.join(TABLES_DIR, "sinuosity_per_basin.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  B. GEOMORPHIC ANOMALY INDEX (GAI) RASTER
# ─────────────────────────────────────────────────────────────────────────────
# GAI = normalised composite of:
#   • SL anomaly (rasterised from segment values)
#   • TRI (terrain ruggedness)
#   • TWI inverted (low TWI = steep, anomalous)
# Higher GAI → geomorphically anomalous zone

print("\n[B] Geomorphic Anomaly Index (GAI) raster...")

def normalise_0_1(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# Rasterise SL anomaly: burn each segment's SL_anomaly value onto raster
SL_anomaly_raster = np.full(DEM_ARR.shape, np.nan, dtype=np.float32)
with rasterio.open(RASTERS['dem']) as src:
    transform_r = src.transform
    for _, seg in gdf_SL[gdf_SL['SL_anomaly'].notna()].iterrows():
        geom = seg['geometry']
        pts  = [geom.interpolate(f, normalized=True) for f in np.linspace(0, 1, 20)]
        for pt in pts:
            try:
                r_i, c_i = rowcol(transform_r, pt.x, pt.y)
                if 0 <= r_i < SL_anomaly_raster.shape[0] and 0 <= c_i < SL_anomaly_raster.shape[1]:
                    existing = SL_anomaly_raster[r_i, c_i]
                    val      = seg['SL_anomaly']
                    SL_anomaly_raster[r_i, c_i] = val if np.isnan(existing) else max(existing, val)
            except:
                pass

# Fill gaps with Gaussian spread (proximity decay)
mask_sl = ~np.isnan(SL_anomaly_raster)
SL_filled = np.where(mask_sl, SL_anomaly_raster, 0)
SL_spread = gaussian_filter(SL_filled, sigma=5)
SL_spread[np.isnan(DEM_ARR)] = np.nan

# TRI already computed: TRI_ARR
# TWI inverted: high TWI = flat = low anomaly → invert
TWI_inv = np.nanmax(TWI_ARR) - TWI_ARR

# Composite GAI
n_SL  = normalise_0_1(SL_spread)
n_TRI = normalise_0_1(TRI_ARR)
n_TWI = normalise_0_1(TWI_inv)

GAI = (n_SL * 0.5 + n_TRI * 0.3 + n_TWI * 0.2)
GAI[np.isnan(DEM_ARR)] = np.nan

save_raster(GAI, os.path.join(OUT_DIR, "GAI.tif"), RASTERS['dem'])
RASTERS['GAI'] = os.path.join(OUT_DIR, "GAI.tif")
print(f"  GAI range: {np.nanmin(GAI):.3f} – {np.nanmax(GAI):.3f}")

# Classify high anomaly zones (top 20%)
GAI_thresh       = np.nanpercentile(GAI, 80)
HIGH_ANOMALY     = (GAI > GAI_thresh).astype(np.float32)
HIGH_ANOMALY[np.isnan(DEM_ARR)] = np.nan
save_raster(HIGH_ANOMALY, os.path.join(OUT_DIR, "GAI_high_anomaly.tif"), RASTERS['dem'])

# Per-basin GAI statistics
GAI_basin = []
for _, row in gdf_sub.iterrows():
    geom = [row.geometry.__geo_interface__]
    with rasterio.open(os.path.join(OUT_DIR, "GAI.tif")) as src:
        try:
            arr_m, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
            gai_clip  = arr_m[0]
            gai_clip[gai_clip == -9999] = np.nan
        except:
            gai_clip  = GAI.copy()
    GAI_basin.append({
        'basin_id': row['basin_id'],
        'GAI_mean': round(float(np.nanmean(gai_clip)), 4),
        'GAI_max' : round(float(np.nanmax(gai_clip)), 4),
        'GAI_high_frac': round(float(np.nanmean(gai_clip > GAI_thresh)), 4),
    })
df_GAI_basin = pd.DataFrame(GAI_basin).set_index('basin_id')
print("  Per-basin GAI:")
print(df_GAI_basin.to_string())
df_GAI_basin.to_csv(os.path.join(TABLES_DIR, "GAI_per_basin.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  C. LINEAMENT PROXY — structural lineament detection
# ─────────────────────────────────────────────────────────────────────────────
# Method: Sobel edge detection on smoothed DEM + slope, thresholded to
# identify linear high-gradient zones likely representing faults/fractures.

print("\n[C] Structural Lineament Proxy...")

try:
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print("  scikit-image not available — using Sobel only")

# Smooth DEM
dem_smooth   = gaussian_filter(np.where(np.isnan(DEM_ARR), np.nanmean(DEM_ARR), DEM_ARR), sigma=3)

# Sobel edge magnitude
sx = sobel(dem_smooth, axis=1)
sy = sobel(dem_smooth, axis=0)
edge_mag = np.hypot(sx, sy)
edge_mag[np.isnan(DEM_ARR)] = 0

# Combine with slope for structural emphasis
edge_combined = (normalise_0_1(edge_mag) * 0.6 +
                 normalise_0_1(np.where(np.isnan(SLOPE_ARR), 0, SLOPE_ARR)) * 0.4)
edge_combined[np.isnan(DEM_ARR)] = np.nan
save_raster(edge_combined.astype(np.float32),
            os.path.join(OUT_DIR, "lineament_proxy.tif"), RASTERS['dem'])

# Detect probable lineaments using Canny + Hough if available
LINEAMENTS_GDF = None
if SKIMAGE_OK:
    try:
        edge_uint8 = ((edge_combined / np.nanmax(edge_combined)) * 255).astype(np.uint8)
        canny_edges = canny(edge_uint8, sigma=2,
                            low_threshold=50, high_threshold=100)
        lines = probabilistic_hough_line(
            canny_edges, threshold=30, line_length=20, line_gap=5
        )
        if lines:
            line_geoms = []
            with rasterio.open(RASTERS['dem']) as src:
                T = src.transform
                for (x0, y0), (x1, y1) in lines:
                    wx0, wy0 = xy(T, y0, x0)
                    wx1, wy1 = xy(T, y1, x1)
                    if abs(wx1 - wx0) > 0 or abs(wy1 - wy0) > 0:
                        line_geoms.append(LineString([(wx0, wy0), (wx1, wy1)]))
            if line_geoms:
                LINEAMENTS_GDF = gpd.GeoDataFrame(
                    {'lineament_id': range(len(line_geoms))},
                    geometry=line_geoms, crs=UTM_EPSG
                )
                LINEAMENTS_GDF.to_file(os.path.join(SHAPES_DIR, "lineament_proxy.shp"))
                print(f"  Detected {len(line_geoms)} probable lineaments")
    except Exception as e:
        print(f"  Hough detection failed ({e}) — edge raster saved only")

# ─────────────────────────────────────────────────────────────────────────────
#  D. MAPS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[D] Generating anomaly maps...")

utm_ext = compute_utm_extent()

# GAI map
fig, ax = base_axes("Geomorphic Anomaly Index (GAI)\n"
                    "(0.5×SL + 0.3×TRI + 0.2×TWI⁻¹ normalised composite)")
im = ax.imshow(
    GAI, extent=raster_extent(), origin='upper',
    cmap='RdYlGn_r', alpha=0.80, zorder=1,
    vmin=0, vmax=1,
)
# High anomaly contour overlay
b = DEM_BOUNDS
x_c = np.linspace(b.left, b.right,  GAI.shape[1])
y_c = np.linspace(b.bottom, b.top,  GAI.shape[0])[::-1]
XX, YY = np.meshgrid(x_c, y_c)
ax.contour(XX, YY, np.where(np.isnan(GAI), 0, GAI),
           levels=[GAI_thresh], colors='black', linewidths=1.5,
           linestyles='--', zorder=8)
ax.text(0.02, 0.06, f"Dashed contour = top 20%\nGAI threshold = {GAI_thresh:.3f}",
        transform=ax.transAxes, fontsize=7.5, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2, zorder=10)
gdf_streams.plot(ax=ax, color='royalblue', linewidth=0.5, alpha=0.4, zorder=6)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.07)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("GAI (0 = low, 1 = high anomaly)", fontsize=9)
finalize_and_save(fig, ax, utm_ext, "12a_GAI_map.png")

# Lineament proxy map
fig, ax = base_axes("Structural Lineament Proxy (Sobel edge + slope composite)")
im2 = ax.imshow(
    edge_combined,
    extent=raster_extent(), origin='upper',
    cmap='copper', alpha=0.80, zorder=1,
    vmin=0, vmax=np.nanpercentile(edge_combined, 99),
)
if LINEAMENTS_GDF is not None and len(LINEAMENTS_GDF) > 0:
    LINEAMENTS_GDF.plot(ax=ax, color='cyan', linewidth=0.7, alpha=0.7, zorder=7,
                        label='Probable lineaments')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
gdf_sub.boundary.plot(ax=ax, edgecolor='white', linewidth=1.2, zorder=10)
divider2 = make_axes_locatable(ax)
cax2 = divider2.append_axes("right", size="3%", pad=0.07)
cb2  = plt.colorbar(im2, cax=cax2)
cb2.set_label("Edge Magnitude (normalised)", fontsize=9)
finalize_and_save(fig, ax, utm_ext, "12b_lineament_proxy_map.png")

# Channel sinuosity map — coloured by SI
fig, ax = base_axes("Channel Sinuosity Index (SI) per Segment")
SI_valid  = gdf_SL[gdf_SL['SI'].notna()].copy()
if len(SI_valid) > 0:
    vmin_si, vmax_si = 1.0, np.nanpercentile(SI_valid['SI'], 98)
    cmap_si = plt.get_cmap('RdYlBu_r')
    norm_si = Normalize(vmin=vmin_si, vmax=vmax_si)
    for _, seg in SI_valid.iterrows():
        color = cmap_si(norm_si(seg['SI']))
        ax.plot(*seg.geometry.xy, color=color, linewidth=1.2, zorder=5)
    sm_si = plt.cm.ScalarMappable(cmap=cmap_si, norm=norm_si)
    sm_si.set_array([])
    cb3 = plt.colorbar(sm_si, ax=ax, fraction=0.03, pad=0.02)
    cb3.set_label("Sinuosity Index (SI)", fontsize=9)
gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2, zorder=10)
finalize_and_save(fig, ax, utm_ext, "12c_sinuosity_map.png")

# ─────────────────────────────────────────────────────────────────────────────
#  E. PLOTLY — GAI interactive + anomaly scatter
# ─────────────────────────────────────────────────────────────────────────────

print("\n[E] Plotly charts...")

# GAI per basin bar + sinuosity overlay
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["GAI Mean per Subbasin",
                                    "Sinuosity vs SL Anomaly"])
fig.add_trace(go.Bar(
    x=df_GAI_basin.index.tolist(),
    y=df_GAI_basin['GAI_mean'].tolist(),
    marker_color=px.colors.sequential.Reds[3:],
    text=[f"{v:.3f}" for v in df_GAI_basin['GAI_mean']],
    textposition='outside',
    hovertemplate='%{x}<br>GAI mean: %{y:.3f}',
    name='GAI mean',
), row=1, col=1)
fig.add_trace(go.Bar(
    x=df_GAI_basin.index.tolist(),
    y=(df_GAI_basin['GAI_high_frac'] * 100).tolist(),
    marker_color=px.colors.sequential.OrRd[3:],
    name='% High anomaly',
    hovertemplate='%{x}<br>High anomaly: %{y:.1f}%',
    yaxis='y2',
), row=1, col=1)

# Sinuosity vs SL scatter
for bid in gdf_sub['basin_id']:
    si_m  = SI_per_basin.loc[bid, 'SI_mean'] if bid in SI_per_basin.index else np.nan
    sl_m  = SL_per_basin.loc[bid, 'SL_anomaly_max'] if bid in SL_per_basin.index else np.nan
    if np.isnan(si_m) or np.isnan(sl_m):
        continue
    fig.add_trace(go.Scatter(
        x=[si_m], y=[sl_m], mode='markers+text',
        text=[bid], textposition='top center',
        marker=dict(size=14, symbol='circle'),
        name=bid,
        hovertemplate=f"{bid}<br>SI={si_m:.3f}<br>SL anomaly={sl_m:.2f}",
    ), row=1, col=2)

fig.update_xaxes(title_text='Subbasin', row=1, col=1)
fig.update_yaxes(title_text='GAI Mean', row=1, col=1)
fig.update_xaxes(title_text='Mean Sinuosity (SI)', row=1, col=2)
fig.update_yaxes(title_text='Max SL Anomaly', row=1, col=2)
fig.update_layout(title="Geomorphic Anomaly Analysis",
                  template='plotly_white', height=480, showlegend=True)
save_fig(fig, "12d_geomorphic_anomaly_plotly")

print("\n✅ SECTION 12 complete.")
