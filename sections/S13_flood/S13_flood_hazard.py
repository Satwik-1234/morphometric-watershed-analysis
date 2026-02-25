"""
=============================================================================
SECTION 13 — FLOOD HAZARD INDICATORS
=============================================================================
Assumes Sections 1–12 in memory.
TWI, SPI, STI already computed in Section 11.

New parameters:
  FFPI — Flash Flood Potential Index (Smith, 2003)
         FFPI = f(slope, land cover proxy, drainage density, relief)
         Implemented as: FFPI = (norm_slope × 0.4 + norm_relief × 0.3
                                  + norm_Dd × 0.2 + norm_TWI_inv × 0.1)
  FHI  — Flood Hazard Index per subbasin (composite rank-based)

Maps:
  TWI map
  SPI map
  STI map
  FFPI map
  Composite flood hazard map (FHI choropleth)

Plotly:
  Multi-panel hazard index comparison
  Bubble plot: FFPI vs Dd vs Relief
  Flood susceptibility ranking bar
=============================================================================
"""

print("=" * 60)
print("SECTION 13 — FLOOD HAZARD INDICATORS")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
#  A. VERIFY TWI / SPI / STI (computed in S11)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[A] Loading TWI, SPI, STI arrays...")
assert 'TWI' in RASTERS, "TWI raster not found — ensure Section 11 ran first"

for key in ['TWI', 'SPI', 'STI']:
    with rasterio.open(RASTERS[key]) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == -9999.0] = np.nan
        mn, mx, mu = np.nanmin(arr), np.nanmax(arr), np.nanmean(arr)
        print(f"  {key:4s}: min={mn:.2f} max={mx:.2f} mean={mu:.2f}")

# Re-read into memory (may have been computed in S11)
with rasterio.open(RASTERS['TWI']) as src:
    TWI_ARR2 = src.read(1).astype(np.float32)
    TWI_ARR2[TWI_ARR2 == -9999.0] = np.nan

with rasterio.open(RASTERS['SPI']) as src:
    SPI_ARR2 = src.read(1).astype(np.float32)
    SPI_ARR2[SPI_ARR2 == -9999.0] = np.nan

with rasterio.open(RASTERS['STI']) as src:
    STI_ARR2 = src.read(1).astype(np.float32)
    STI_ARR2[STI_ARR2 == -9999.0] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
#  B. FLASH FLOOD POTENTIAL INDEX (FFPI)
# ─────────────────────────────────────────────────────────────────────────────
# FFPI raster derived from slope, relief proxy, and TWI
# Weights following Smith (2003) and Gregory & Walling (1973) concepts

print("\n[B] Flash Flood Potential Index (FFPI)...")

def normalise_raster(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# Component normalised rasters
norm_slope   = normalise_raster(np.where(np.isnan(SLOPE_ARR), 0, SLOPE_ARR))

# Relief proxy: local relief within 5×5 neighbourhood
from scipy.ndimage import maximum_filter, minimum_filter
dem_safe     = np.where(np.isnan(DEM_ARR), np.nanmean(DEM_ARR), DEM_ARR)
local_relief = maximum_filter(dem_safe, size=5) - minimum_filter(dem_safe, size=5)
local_relief[np.isnan(DEM_ARR)] = np.nan
norm_relief  = normalise_raster(np.where(np.isnan(local_relief), 0, local_relief))

# TWI inverted: high TWI = flat accumulation zone = high flood potential
TWI_safe     = np.where(np.isnan(TWI_ARR2), np.nanmin(TWI_ARR2), TWI_ARR2)
norm_twi     = normalise_raster(TWI_safe)

# SPI: high SPI = high stream power = high flood energy
SPI_safe     = np.where(np.isnan(SPI_ARR2), 0, SPI_ARR2)
norm_spi     = normalise_raster(np.log1p(SPI_safe))  # log-transform

# Weighted FFPI
FFPI = (norm_slope  * 0.35 +
        norm_relief * 0.25 +
        norm_twi    * 0.25 +
        norm_spi    * 0.15)
FFPI[np.isnan(DEM_ARR)] = np.nan

save_raster(FFPI.astype(np.float32), os.path.join(OUT_DIR, "FFPI.tif"), RASTERS['dem'])
RASTERS['FFPI'] = os.path.join(OUT_DIR, "FFPI.tif")
print(f"  FFPI range: {np.nanmin(FFPI):.3f} – {np.nanmax(FFPI):.3f}")

# Classify FFPI
def classify_ffpi(val):
    if np.isnan(val): return "Unknown"
    if val > 0.75:    return "Very High"
    if val > 0.55:    return "High"
    if val > 0.35:    return "Moderate"
    if val > 0.20:    return "Low"
    return "Very Low"

# ─────────────────────────────────────────────────────────────────────────────
#  C. PER-BASIN HAZARD STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[C] Per-basin hazard statistics...")

HAZARD_ROWS = []
for _, row in gdf_sub.iterrows():
    bid  = row['basin_id']
    geom = [row.geometry.__geo_interface__]

    def mask_raster(path):
        with rasterio.open(path) as src:
            try:
                arr_m, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
                arr = arr_m[0].astype(np.float32)
                arr[arr == -9999.0] = np.nan
                return arr
            except:
                return np.array([np.nan])

    twi_clip  = mask_raster(RASTERS['TWI'])
    spi_clip  = mask_raster(RASTERS['SPI'])
    sti_clip  = mask_raster(RASTERS['STI'])
    ffpi_clip = mask_raster(RASTERS['FFPI'])

    ffpi_mean = float(np.nanmean(ffpi_clip))
    HAZARD_ROWS.append({
        'basin_id'      : bid,
        'TWI_mean'      : round(float(np.nanmean(twi_clip)),  3),
        'TWI_max'       : round(float(np.nanmax(twi_clip)),   3),
        'SPI_mean'      : round(float(np.nanmean(spi_clip)),  3),
        'SPI_max'       : round(float(np.nanmax(spi_clip)),   3),
        'STI_mean'      : round(float(np.nanmean(sti_clip)),  3),
        'STI_max'       : round(float(np.nanmax(sti_clip)),   3),
        'FFPI_mean'     : round(ffpi_mean, 4),
        'FFPI_max'      : round(float(np.nanmax(ffpi_clip)),  4),
        'FFPI_high_frac': round(float(np.nanmean(ffpi_clip > 0.55)), 4),
        'FFPI_class'    : classify_ffpi(ffpi_mean),
    })
    print(f"  {bid}: TWI_mean={np.nanmean(twi_clip):.2f} | "
          f"SPI_mean={np.nanmean(spi_clip):.2f} | "
          f"FFPI_mean={ffpi_mean:.3f} → {classify_ffpi(ffpi_mean)}")

df_hazard = pd.DataFrame(HAZARD_ROWS).set_index('basin_id')

# Composite Flood Hazard Rank
rank_cols = ['TWI_mean', 'SPI_mean', 'STI_mean', 'FFPI_mean']
df_hazard_r = df_hazard[rank_cols].copy()
for col in rank_cols:
    df_hazard[f'rank_{col}'] = df_hazard_r[col].rank(ascending=False, method='min')
df_hazard['FHI_rank'] = df_hazard[[f'rank_{c}' for c in rank_cols]].mean(axis=1)
df_hazard['FHI_priority'] = pd.qcut(
    df_hazard['FHI_rank'], q=3, labels=['High','Moderate','Low'], duplicates='drop'
)

df_hazard.to_csv(os.path.join(TABLES_DIR, "flood_hazard_indices.csv"))
print(f"\n  ✅ Flood hazard table saved")
print(df_hazard[['TWI_mean','SPI_mean','STI_mean','FFPI_mean','FFPI_class','FHI_priority']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  D. MAPS — all 5 hazard maps
# ─────────────────────────────────────────────────────────────────────────────

print("\n[D] Generating hazard maps...")

utm_ext = compute_utm_extent()

MAP_CONFIGS = [
    ('TWI',  TWI_ARR2, 'Topographic Wetness Index (TWI)', 'Blues',   "13a_TWI_map.png"),
    ('SPI',  SPI_ARR2, 'Stream Power Index (SPI)',        'YlOrRd',  "13b_SPI_map.png"),
    ('STI',  STI_ARR2, 'Sediment Transport Index (STI)',  'RdPu',    "13c_STI_map.png"),
    ('FFPI', FFPI,     'Flash Flood Potential Index (FFPI)\n(Slope×0.35 + Relief×0.25 + TWI×0.25 + SPI×0.15)',
                                                           'OrRd',    "13d_FFPI_map.png"),
]

for key, arr_map, title, cmap_name, fname in MAP_CONFIGS:
    fig, ax = base_axes(title)
    vmax_map = np.nanpercentile(arr_map, 98)
    im = ax.imshow(
        arr_map,
        extent=raster_extent(), origin='upper',
        cmap=cmap_name, alpha=0.78, zorder=1,
        vmin=np.nanpercentile(arr_map, 2), vmax=vmax_map,
    )
    gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2, zorder=10)
    gdf_streams.plot(ax=ax, color='royalblue', linewidth=0.6, alpha=0.5, zorder=8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.07)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(key, fontsize=10)
    finalize_and_save(fig, ax, utm_ext, fname)

# Composite flood hazard choropleth
ffpi_class_colors = {
    'Very High': '#7f0000', 'High': '#d73027', 'Moderate': '#fc8d59',
    'Low': '#fee090',       'Very Low': '#91bfdb', 'Unknown': 'grey',
}
fig, ax = base_axes("Composite Flood Hazard Priority Map\n"
                    "(TWI + SPI + STI + FFPI composite ranking)")
gdf_fhaz = gdf_sub.merge(
    df_hazard[['FFPI_class','FHI_priority','FFPI_mean']].reset_index(),
    on='basin_id', how='left',
)
for _, row in gdf_fhaz.iterrows():
    col = ffpi_class_colors.get(row['FFPI_class'], 'grey')
    gpd.GeoDataFrame([row], geometry='geometry', crs=gdf_sub.crs).plot(
        ax=ax, color=col, edgecolor='black', linewidth=1.2, alpha=0.80, zorder=3
    )
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, f"{row['basin_id']}\n{row['FFPI_class']}\nFFPI={row['FFPI_mean']:.3f}",
            ha='center', va='center', fontsize=7.5, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

gdf_streams.plot(ax=ax, color='royalblue', linewidth=0.7, alpha=0.5, zorder=7)
legend_patches = [mpatches.Patch(color=v, label=k)
                  for k, v in ffpi_class_colors.items() if k != 'Unknown']
ax.legend(handles=legend_patches, loc='lower left', fontsize=8,
          title='Flood Hazard Class', title_fontsize=9, framealpha=0.9)
finalize_and_save(fig, ax, utm_ext, "13e_flood_hazard_composite_map.png")

# ─────────────────────────────────────────────────────────────────────────────
#  E. PLOTLY — multi-panel hazard comparison
# ─────────────────────────────────────────────────────────────────────────────

print("\n[E] Plotly flood hazard charts...")

basins = df_hazard.index.tolist()

# Multi-panel bar comparison
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=['TWI Mean', 'SPI Mean', 'STI Mean', 'FFPI Mean'])
colors_p = px.colors.qualitative.Set1

for panel_i, (col, r, c_idx) in enumerate([
    ('TWI_mean',  1, 1), ('SPI_mean',  1, 2),
    ('STI_mean',  2, 1), ('FFPI_mean', 2, 2),
]):
    fig.add_trace(go.Bar(
        x=basins, y=df_hazard[col].tolist(),
        name=col, marker_color=colors_p[panel_i % 9],
        text=[f"{v:.3f}" for v in df_hazard[col]],
        textposition='outside',
        hovertemplate='%{x}<br>' + col + ': %{y:.3f}',
    ), row=r, col=c_idx)

fig.update_layout(
    title="Flood Hazard Indices — All Subbasins",
    template='plotly_white', height=600, showlegend=False,
)
save_fig(fig, "13f_flood_indices_bar")

# Bubble plot: FFPI vs Drainage Density vs Basin Relief
df_bubble_fh = df_hazard[['FFPI_mean']].join(
    df_master[['Drainage_Density_Dd', 'Basin_Relief_H_m']]
).reset_index()
fig = px.scatter(
    df_bubble_fh, x='Drainage_Density_Dd', y='FFPI_mean',
    size='Basin_Relief_H_m', color='basin_id', text='basin_id',
    title="Flash Flood Potential vs Drainage Density<br>"
          "<sup>Bubble size = Basin Relief (m)</sup>",
    labels={
        'Drainage_Density_Dd': 'Drainage Density (km/km²)',
        'FFPI_mean': 'FFPI Mean',
        'Basin_Relief_H_m': 'Relief (m)',
    },
    template='plotly_white', size_max=55,
)
fig.add_hline(y=0.55, line_dash='dash', line_color='red',
              annotation_text='High flood hazard threshold (FFPI=0.55)')
save_fig(fig, "13g_flood_bubble_plot")

# Susceptibility ranking bar
fig = go.Figure()
fig.add_trace(go.Bar(
    x=basins,
    y=df_hazard['FHI_rank'].tolist(),
    marker_color=[
        {'High': '#d73027', 'Moderate': '#fc8d59', 'Low': '#4575b4'}.get(
            str(df_hazard.loc[b, 'FHI_priority']), 'grey'
        ) for b in basins
    ],
    text=[str(df_hazard.loc[b, 'FHI_priority']) for b in basins],
    textposition='outside',
    hovertemplate='%{x}<br>FHI Rank: %{y:.2f}<br>Priority: %{text}',
))
fig.update_layout(
    title="Flood Hazard Priority Ranking<br>"
          "<sup>Lower rank = higher flood susceptibility</sup>",
    xaxis_title="Subbasin", yaxis_title="FHI Composite Rank",
    template='plotly_white', height=430,
    yaxis=dict(autorange='reversed'),
)
save_fig(fig, "13h_flood_susceptibility_ranking")

print("\n✅ SECTION 13 complete.")

# ─────────────────────────────────────────────────────────────────────────────
#  ADVANCED INTERPRETATION PARAGRAPHS (appended to report)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[F] Writing advanced interpretation to report...")

ADVANCED_REPORT_PATH = os.path.join(REPORT_DIR, "advanced_analysis_interpretation.txt")

with open(ADVANCED_REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ADVANCED MORPHOMETRIC ANALYSIS — SUPPLEMENTARY INTERPRETATIONS\n")
    f.write("=" * 80 + "\n\n")

    f.write("10. TECTONIC ACTIVITY ANALYSIS\n" + "-"*40 + "\n")
    f.write(
        "The Index of Active Tectonics (IAT) integrates four geomorphic proxies: "
        "Asymmetry Factor (AF), Transverse Symmetry (T), Valley Floor Width-to-Height "
        "Ratio (Vf), and Mountain Front Sinuosity (Smf), following El Hamdouni et al. "
        "(2008). AF values deviating substantially from 50 indicate basin tilting "
        "driven by differential uplift or lithological asymmetry. Vf < 0.5 is "
        "diagnostic of active incision associated with tectonic uplift, producing "
        "V-shaped valleys, whereas Vf > 1.0 reflects reduced tectonic activity and "
        "lateral widening. Low Smf (< 1.4) indicates a tectonically active, "
        "straight mountain front.\n\n"
    )
    for bid in gdf_sub['basin_id']:
        if bid in df_IAT.index:
            row = df_IAT.loc[bid]
            f.write(
                f"  {bid}: IAT={row['IAT']:.2f} ({row['IAT_class']}). "
                f"AF={row['AF']:.2f}, T={row['T']:.4f}, "
                f"Vf={row['Vf']:.3f}, Smf={row['Smf']:.3f}.\n"
            )
    f.write("\n")

    f.write("11. CHANNEL STEEPNESS & CONCAVITY\n" + "-"*40 + "\n")
    f.write(
        "Channel steepness indices (ksn) and concavity (θ) were derived from the "
        "slope-area relationship following Hack (1973) and Flint (1974). High ksn "
        "values indicate either strong lithological resistance, active rock uplift, "
        "or transient adjustment to base-level change. The chi (χ) coordinate plot "
        "(Perron & Royden, 2012) allows comparison of drainage networks independent "
        "of their spatial position, where non-collinear χ-elevation relationships "
        "between adjacent basins signal ongoing divide migration or stream capture. "
        "SL anomaly hotspots correspond to knickpoints or reaches crossing resistant "
        "lithological boundaries.\n\n"
    )
    for bid, tres in THETA_RESULTS.items():
        f.write(
            f"  {bid}: θ={tres['theta_concavity']:.3f} "
            f"({'Concave (normal)' if tres['theta_concavity'] > 0.3 else 'Low concavity (active uplift or hard substrate)'}) "
            f"| ksn mean={ksn_stats.loc[bid,'ksn_mean'] if bid in ksn_stats.index else 'N/A'} "
            f"| R²={tres['R2_SA']:.3f}\n"
        )
    f.write("\n")

    f.write("12. GEOMORPHIC ANOMALY & LINEAMENT ANALYSIS\n" + "-"*40 + "\n")
    f.write(
        "The Geomorphic Anomaly Index (GAI) integrates SL anomaly, TRI, and inverse "
        "TWI to identify geomorphically active zones where structural or lithological "
        "controls modulate landscape evolution. High GAI zones (top 20th percentile) "
        "are spatially coincident with anomalously high SL reaches, implying "
        "knickpoint clusters, fault zones, or resistant bedrock outcrops. Structural "
        "lineaments were identified as a proxy using Sobel edge detection combined "
        "with Probabilistic Hough Line Transform, targeting linear high-gradient "
        "alignments in the DEM and slope rasters.\n\n"
    )
    for bid in gdf_sub['basin_id']:
        if bid in df_GAI_basin.index:
            g = df_GAI_basin.loc[bid]
            si_m = SI_per_basin.loc[bid, 'SI_mean'] if bid in SI_per_basin.index else np.nan
            f.write(
                f"  {bid}: GAI_mean={g['GAI_mean']:.3f} | "
                f"High anomaly fraction={g['GAI_high_frac']*100:.1f}% | "
                f"Mean SI={si_m:.3f} "
                f"({'Straight — possible structural control' if si_m < 1.05 else 'Sinuous/meandering'})\n"
            )
    f.write("\n")

    f.write("13. FLOOD HAZARD ANALYSIS\n" + "-"*40 + "\n")
    f.write(
        "Topographic Wetness Index (TWI), Stream Power Index (SPI), Sediment Transport "
        "Index (STI), and Flash Flood Potential Index (FFPI) were computed to characterise "
        "the hydrological response and hazard potential of each subbasin. TWI identifies "
        "zones of moisture accumulation and potential saturation-excess overland flow. "
        "High SPI zones correspond to areas of concentrated flow energy capable of "
        "significant geomorphic work. STI quantifies sediment detachment and transport "
        "potential. FFPI synthesises these signals as a weighted composite.\n\n"
    )
    for bid in df_hazard.index:
        row = df_hazard.loc[bid]
        f.write(
            f"  {bid}: FFPI={row['FFPI_mean']:.3f} ({row['FFPI_class']}) | "
            f"TWI_mean={row['TWI_mean']:.2f} | SPI_mean={row['SPI_mean']:.2f} | "
            f"Flood priority: {row['FHI_priority']}\n"
        )
    f.write("\n")

    f.write("REFERENCES (Advanced Sections)\n" + "-"*40 + "\n")
    refs = [
        "Bull, W.B. & McFadden, L.D. (1977). Tectonic geomorphology N & S of the Garlock fault. Geomorphology in arid regions, 115–138.",
        "Cox, R.T. (1994). Analysis of drainage basin symmetry. Geology, 22(9), 813–816.",
        "El Hamdouni, R. et al. (2008). Assessment of relative active tectonics, SE Spain. Geomorphology, 96(1–2), 150–173.",
        "Flint, J.J. (1974). Stream gradient as a function of order, magnitude, and discharge. Water Resources Research, 10(5), 969–973.",
        "Gregory, K.J. & Walling, D.E. (1973). Drainage Basin Form and Process. Edward Arnold.",
        "Hack, J.T. (1973). Stream-profile analysis and stream-gradient index. USGS Journal of Research, 1(4), 421–429.",
        "Moore, I.D., Grayson, R.B. & Ladson, A.R. (1991). Digital terrain modelling. Hydrological Processes, 5(1), 3–30.",
        "Moore, I.D. & Burch, G.J. (1986). Sediment transport capacity of sheet and rill flow. Water Resources Research, 22(13), 1350–1360.",
        "Perron, J.T. & Royden, L. (2012). An integral approach to bedrock river profile analysis. Earth Surface Processes and Landforms, 38(6), 570–576.",
        "Smith, G.H. (2003). The morphometry of drainage basins. Annals of the Association of American Geographers.",
    ]
    for ref in refs:
        f.write(f"  {ref}\n")

print(f"  ✅ Advanced interpretation saved: {ADVANCED_REPORT_PATH}")
print("\n✅ ALL ADVANCED SECTIONS COMPLETE (10–13).")
print(f"\n  Total new maps  : 5 tectonic + 3 channel + 3 anomaly + 5 flood = 16 maps")
print(f"  Total new tables: IAT, SL, ksn, theta, sinuosity, GAI, flood hazard = 7 CSVs")
print(f"  Total new plots : 14 interactive Plotly HTML files")
