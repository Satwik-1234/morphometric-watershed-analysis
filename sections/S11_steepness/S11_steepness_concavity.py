"""
=============================================================================
SECTION 11 — CHANNEL STEEPNESS, CONCAVITY & CHI (χ) ANALYSIS
=============================================================================
Assumes Sections 1–3 in memory.

Parameters:
  SL   — Hack's Stream Length Gradient Index (Hack, 1973)
         SL = (ΔH/ΔL) × L_total   per reach
  ksn  — Channel steepness index (normalised, θref=0.45)
         ksn = S / A^(−θref)   where S=local slope, A=drainage area
  θ    — Concavity index (regression slope on log S vs log A)
  χ    — Chi coordinate for channel comparison (Perron & Royden, 2012)
         χ = ∫(A0/A(x))^m dx  from outlet to headwater

Maps:
  SL anomaly map (all basins)
  ksn map
  χ map

Plotly:
  χ-elevation plot per subbasin
  SL index bar plot
  slope-area log-log with θ regression
=============================================================================
"""

print("=" * 60)
print("SECTION 11 — CHANNEL STEEPNESS & CONCAVITY")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE TWI (used here and in Section 13)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[0] Computing TWI, SPI, STI arrays...")

def compute_TWI(facc_arr, slope_arr, res_m):
    """
    TWI = ln(A / tan(β))
    A   = specific catchment area = facc × res² / res = facc × res
    β   = slope in radians (use tan to avoid log(0))
    """
    A_sca    = np.where(facc_arr > 0, facc_arr, 0.5) * res_m   # m
    slope_rad = np.radians(np.where(slope_arr > 0.001, slope_arr, 0.001))
    TWI       = np.log(A_sca / np.tan(slope_rad))
    TWI[np.isnan(slope_arr) | np.isnan(facc_arr)] = np.nan
    return TWI.astype(np.float32)

def compute_SPI(facc_arr, slope_arr, res_m):
    """
    SPI = A × tan(β)   [Stream Power Index, Moore et al., 1991]
    A = specific catchment area (m)
    """
    A_sca     = np.where(facc_arr > 0, facc_arr, 0.5) * res_m
    slope_rad = np.radians(np.where(slope_arr > 0.001, slope_arr, 0.001))
    SPI       = A_sca * np.tan(slope_rad)
    SPI[np.isnan(slope_arr) | np.isnan(facc_arr)] = np.nan
    return SPI.astype(np.float32)

def compute_STI(facc_arr, slope_arr, res_m):
    """
    STI = (A/22.13)^0.6 × (sin(β)/0.0896)^1.3   [Sediment Transport Index]
    Moore & Burch (1986)
    """
    A_sca     = np.where(facc_arr > 0, facc_arr, 0.5) * res_m
    slope_rad = np.radians(np.where(slope_arr > 0.001, slope_arr, 0.001))
    STI       = ((A_sca / 22.13) ** 0.6) * ((np.sin(slope_rad) / 0.0896) ** 1.3)
    STI[np.isnan(slope_arr) | np.isnan(facc_arr)] = np.nan
    return STI.astype(np.float32)

TWI_ARR = compute_TWI(FACC_ARR, SLOPE_ARR, DEM_RES)
SPI_ARR = compute_SPI(FACC_ARR, SLOPE_ARR, DEM_RES)
STI_ARR = compute_STI(FACC_ARR, SLOPE_ARR, DEM_RES)

save_raster(TWI_ARR, os.path.join(OUT_DIR, "TWI.tif"),  RASTERS['dem'])
save_raster(SPI_ARR, os.path.join(OUT_DIR, "SPI.tif"),  RASTERS['dem'])
save_raster(STI_ARR, os.path.join(OUT_DIR, "STI.tif"),  RASTERS['dem'])
RASTERS['TWI'] = os.path.join(OUT_DIR, "TWI.tif")
RASTERS['SPI'] = os.path.join(OUT_DIR, "SPI.tif")
RASTERS['STI'] = os.path.join(OUT_DIR, "STI.tif")
print("  ✅ TWI, SPI, STI computed and saved (also used in Section 13)")

# ─────────────────────────────────────────────────────────────────────────────
#  A. HACK'S SL INDEX
# ─────────────────────────────────────────────────────────────────────────────
# SL = (ΔH / ΔL) × L   for each reach segment
# Anomalously high SL values indicate lithological/structural controls

print("\n[A] Hack's Stream Length Gradient Index (SL)...")

def compute_SL_for_segments(stream_gdf, dem_path):
    """
    For each stream segment: sample elevation at start/end,
    compute SL = slope × cumulative length from headwater.
    """
    rows = []
    with rasterio.open(dem_path) as src:
        dem_data  = src.read(1).astype(float)
        dem_nd    = src.nodata if src.nodata else -9999
        dem_data[dem_data == dem_nd] = np.nan
        transform = src.transform

        for idx, seg_row in stream_gdf.iterrows():
            geom   = seg_row.geometry
            if geom.geom_type == 'MultiLineString':
                geom = max(geom.geoms, key=lambda g: g.length)
            if geom.geom_type != 'LineString' or geom.length < DEM_RES:
                continue

            # Sample elevation at start and end
            def sample_elev(pt):
                try:
                    r_i, c_i = rowcol(transform, pt.x, pt.y)
                    e = dem_data[r_i, c_i]
                    return np.nan if np.isnan(e) else e
                except:
                    return np.nan

            e_start = sample_elev(Point(geom.coords[0]))
            e_end   = sample_elev(Point(geom.coords[-1]))

            if np.isnan(e_start) or np.isnan(e_end):
                continue

            dH = abs(e_start - e_end)
            dL = geom.length
            L  = dL  # segment length as proxy for cumulative length

            SL_val = (dH / dL) * L if dL > 0 else np.nan
            rows.append({
                'seg_id'   : idx,
                'SL'       : round(SL_val, 3) if not np.isnan(SL_val) else np.nan,
                'dH_m'     : round(dH, 2),
                'dL_m'     : round(dL, 2),
                'slope'    : round(dH / dL, 6) if dL > 0 else np.nan,
                'geometry' : geom,
                ORDER_COL  : seg_row.get(ORDER_COL, np.nan),
                'basin_id' : seg_row.get('basin_id', np.nan),
            })

    return gpd.GeoDataFrame(rows, crs=stream_gdf.crs)


# Merge basin_id into stream order layer
gdf_so_bid = gdf_so_sub.copy()
gdf_SL = compute_SL_for_segments(gdf_so_bid, RASTERS['dem'])

print(f"  Computed SL for {len(gdf_SL)} segments")
print(f"  SL range: {gdf_SL['SL'].min():.1f} – {gdf_SL['SL'].max():.1f}")
print(f"  SL mean: {gdf_SL['SL'].mean():.1f} | std: {gdf_SL['SL'].std():.1f}")

# SL anomaly = SL / mean(SL)
SL_mean        = gdf_SL['SL'].mean()
gdf_SL['SL_anomaly'] = gdf_SL['SL'] / SL_mean

# Per-basin SL statistics
SL_per_basin = gdf_SL.groupby('basin_id')['SL'].agg(
    SL_mean='mean', SL_max='max', SL_std='std'
).round(3)
SL_per_basin['SL_anomaly_max'] = gdf_SL.groupby('basin_id')['SL_anomaly'].max().round(3)
print("\n  Per-basin SL statistics:")
print(SL_per_basin.to_string())
SL_per_basin.to_csv(os.path.join(TABLES_DIR, "SL_index_per_basin.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  B. SLOPE-AREA ANALYSIS & CONCAVITY INDEX (θ)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[B] Slope-Area analysis (concavity index θ)...")

THETA_RESULTS = {}

for bid in gdf_sub['basin_id']:
    segs = gdf_SL[gdf_SL['basin_id'] == bid]
    if len(segs) < 3:
        continue

    # Drainage area proxy: flow accumulation at segment midpoint
    A_vals, S_vals = [], []
    with rasterio.open(RASTERS['flow_acc']) as fa_src:
        for _, s_row in segs.iterrows():
            try:
                mid = s_row.geometry.interpolate(0.5, normalized=True)
                r_i, c_i = rowcol(fa_src.transform, mid.x, mid.y)
                fa  = fa_src.read(1)[r_i, c_i]
                nd  = fa_src.nodata if fa_src.nodata else -9999
                if fa == nd or fa <= 0 or np.isnan(s_row['slope']) or s_row['slope'] <= 0:
                    continue
                A_vals.append(float(fa) * DEM_RES ** 2)  # m²
                S_vals.append(s_row['slope'])
            except:
                continue

    if len(A_vals) < 3:
        continue

    log_A = np.log10(A_vals)
    log_S = np.log10(S_vals)
    slope_reg, intercept_reg, r, p, _ = stats.linregress(log_A, log_S)
    theta = -slope_reg   # concavity is negative slope of S-A relationship
    ks    = 10 ** intercept_reg  # steepness index (at A=1)
    THETA_RESULTS[bid] = {
        'theta_concavity': round(theta, 4),
        'ks_steepness'   : round(ks, 6),
        'R2_SA'          : round(r**2, 4),
        'n_segments'     : len(A_vals),
    }
    print(f"  {bid}: θ={theta:.3f} | ks={ks:.4f} | R²={r**2:.3f} (n={len(A_vals)})")

df_theta = pd.DataFrame(THETA_RESULTS).T
df_theta.index.name = 'basin_id'
df_theta.to_csv(os.path.join(TABLES_DIR, "concavity_steepness.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  C. NORMALISED CHANNEL STEEPNESS INDEX (ksn)
# ─────────────────────────────────────────────────────────────────────────────
# ksn = S × A^θref   where θref = 0.45 (standard reference concavity)

print("\n[C] Normalised Steepness Index (ksn, θref=0.45)...")

THETA_REF = 0.45

gdf_SL['ksn'] = np.nan
with rasterio.open(RASTERS['flow_acc']) as fa_src:
    for idx, s_row in gdf_SL.iterrows():
        try:
            mid = s_row.geometry.interpolate(0.5, normalized=True)
            r_i, c_i = rowcol(fa_src.transform, mid.x, mid.y)
            fa  = fa_src.read(1)[r_i, c_i]
            nd  = fa_src.nodata if fa_src.nodata else -9999
            if fa == nd or fa <= 0 or np.isnan(s_row['slope']):
                continue
            A_m2 = float(fa) * DEM_RES ** 2
            ksn  = s_row['slope'] * (A_m2 ** THETA_REF)
            gdf_SL.at[idx, 'ksn'] = round(ksn, 4)
        except:
            continue

ksn_stats = gdf_SL.groupby('basin_id')['ksn'].agg(['mean','max','std']).round(4)
ksn_stats.columns = ['ksn_mean','ksn_max','ksn_std']
print("  Per-basin ksn statistics:")
print(ksn_stats.to_string())
ksn_stats.to_csv(os.path.join(TABLES_DIR, "ksn_per_basin.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  D. CHI (χ) COORDINATE
# ─────────────────────────────────────────────────────────────────────────────
# χ = ∫[outlet→point] (A0/A(x))^m dx   m = θref = 0.45, A0 = 1 m²
# Here: approximate χ along each stream by integrating from outlet upward

print("\n[D] Chi (χ) coordinate computation...")

A0 = 1.0      # reference drainage area (m²)
m  = THETA_REF

CHI_PROFILES = {}

for bid in gdf_sub['basin_id']:
    segs = gdf_SL[gdf_SL['basin_id'] == bid].copy()
    if len(segs) < 2:
        continue

    profile_pts = []
    with rasterio.open(RASTERS['flow_acc']) as fa_src:
        for _, s_row in segs.iterrows():
            try:
                pts = [s_row.geometry.interpolate(f, normalized=True)
                       for f in np.linspace(0, 1, 10)]
                for pt in pts:
                    r_i, c_i = rowcol(fa_src.transform, pt.x, pt.y)
                    fa   = fa_src.read(1)[r_i, c_i]
                    nd   = fa_src.nodata if fa_src.nodata else -9999
                    if fa == nd or fa <= 0:
                        continue
                    A_m2 = float(fa) * DEM_RES ** 2
                    with rasterio.open(RASTERS['dem']) as d_src:
                        e = d_src.read(1)[r_i, c_i]
                        e_nd = d_src.nodata if d_src.nodata else -9999
                        if e == e_nd:
                            continue
                    profile_pts.append({'x': pt.x, 'y': pt.y,
                                        'A_m2': A_m2, 'elev': float(e)})
            except:
                continue

    if len(profile_pts) < 5:
        continue

    df_prof = pd.DataFrame(profile_pts).drop_duplicates(subset=['x','y'])
    df_prof = df_prof.sort_values('elev').reset_index(drop=True)

    # Cumulative chi from outlet (lowest elev)
    chi_vals = [0.0]
    for i in range(1, len(df_prof)):
        dx_chi = np.sqrt(
            (df_prof.loc[i, 'x'] - df_prof.loc[i-1, 'x'])**2 +
            (df_prof.loc[i, 'y'] - df_prof.loc[i-1, 'y'])**2
        )
        A_avg  = (df_prof.loc[i, 'A_m2'] + df_prof.loc[i-1, 'A_m2']) / 2
        chi_vals.append(chi_vals[-1] + (A0 / A_avg) ** m * dx_chi)

    df_prof['chi'] = chi_vals
    CHI_PROFILES[bid] = df_prof
    print(f"  {bid}: χ range 0 – {df_prof['chi'].max():.2f} | n_pts={len(df_prof)}")

# ─────────────────────────────────────────────────────────────────────────────
#  E. MAPS — SL Anomaly & ksn
# ─────────────────────────────────────────────────────────────────────────────

print("\n[E] Generating maps...")

# SL Anomaly map
utm_ext = compute_utm_extent()
fig, ax = base_axes("Stream Length Gradient Index (SL) Anomaly — Hack (1973)")

gdf_SL_valid = gdf_SL[gdf_SL['SL_anomaly'].notna()].copy()
vmax_sl      = np.nanpercentile(gdf_SL_valid['SL_anomaly'], 95)
norm_sl      = Normalize(vmin=0, vmax=vmax_sl)
cmap_sl      = plt.get_cmap('hot_r')

for _, seg in gdf_SL_valid.iterrows():
    color = cmap_sl(norm_sl(seg['SL_anomaly']))
    lw    = 0.8 + seg.get(ORDER_COL, 1) * 0.3
    ax.plot(*seg.geometry.xy, color=color, linewidth=lw, zorder=5)

sm = plt.cm.ScalarMappable(cmap=cmap_sl, norm=norm_sl)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("SL Anomaly (SL / mean SL)", fontsize=9)
ax.axhline(y=-9999, color='red', lw=2, label='High SL (>2σ) — structural control')
gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2, zorder=10)
ax.text(0.02, 0.02, f"Mean SL = {SL_mean:.1f}\nSL > {SL_mean + 2*gdf_SL['SL'].std():.1f} = anomalous",
        transform=ax.transAxes, fontsize=7, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
finalize_and_save(fig, ax, utm_ext, "11a_SL_anomaly_map.png")

# ksn map
fig, ax = base_axes(f"Channel Steepness Index ksn (θref={THETA_REF})")
gdf_SL_ksn = gdf_SL[gdf_SL['ksn'].notna()].copy()
if len(gdf_SL_ksn) > 0:
    vmax_ksn = np.nanpercentile(gdf_SL_ksn['ksn'], 95)
    norm_ksn = Normalize(vmin=0, vmax=vmax_ksn)
    cmap_ksn = plt.get_cmap('plasma')
    for _, seg in gdf_SL_ksn.iterrows():
        color = cmap_ksn(norm_ksn(seg['ksn']))
        ax.plot(*seg.geometry.xy, color=color, linewidth=1.2, zorder=5)
    sm_k = plt.cm.ScalarMappable(cmap=cmap_ksn, norm=norm_ksn)
    sm_k.set_array([])
    cb = plt.colorbar(sm_k, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"ksn (m⁰·⁹ / θref={THETA_REF})", fontsize=9)
gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2, zorder=10)
finalize_and_save(fig, ax, utm_ext, "11b_ksn_map.png")

# ─────────────────────────────────────────────────────────────────────────────
#  F. PLOTLY — χ-elevation plot, slope-area, SL bar
# ─────────────────────────────────────────────────────────────────────────────

print("\n[F] Plotly charts...")

# χ-elevation overlay
if CHI_PROFILES:
    fig = go.Figure()
    colors_chi = px.colors.qualitative.Plotly
    for i, (bid, df_prof) in enumerate(CHI_PROFILES.items()):
        fig.add_trace(go.Scatter(
            x=df_prof['chi'], y=df_prof['elev'],
            mode='markers', name=bid,
            marker=dict(size=3, color=colors_chi[i % 10]),
            hovertemplate=f"{bid}<br>χ: %{{x:.2f}}<br>Elev: %{{y:.1f}} m",
        ))
    fig.update_layout(
        title="χ–Elevation Plot (Perron & Royden, 2012)<br>"
              "<sup>Divergence indicates drainage divide migration or stream capture</sup>",
        xaxis_title="χ (m)", yaxis_title="Elevation (m)",
        template='plotly_white', height=550,
    )
    save_fig(fig, "11c_chi_elevation_plot")

# Slope-Area log-log per basin
fig = make_subplots(rows=1, cols=len(THETA_RESULTS),
                    subplot_titles=[f"{bid} (θ={v['theta_concavity']:.2f})"
                                    for bid, v in THETA_RESULTS.items()])
for col_i, (bid, tres) in enumerate(THETA_RESULTS.items()):
    segs = gdf_SL[gdf_SL['basin_id'] == bid]
    A_plt, S_plt = [], []
    with rasterio.open(RASTERS['flow_acc']) as fa_src:
        for _, s_row in segs.iterrows():
            try:
                mid = s_row.geometry.interpolate(0.5, normalized=True)
                r_i, c_i = rowcol(fa_src.transform, mid.x, mid.y)
                fa = fa_src.read(1)[r_i, c_i]
                nd = fa_src.nodata if fa_src.nodata else -9999
                if fa != nd and fa > 0 and not np.isnan(s_row['slope']) and s_row['slope'] > 0:
                    A_plt.append(float(fa) * DEM_RES**2)
                    S_plt.append(s_row['slope'])
            except:
                pass
    if not A_plt:
        continue
    fig.add_trace(go.Scatter(
        x=A_plt, y=S_plt, mode='markers', name=bid,
        marker=dict(size=5, opacity=0.6),
        hovertemplate=f"{bid}<br>A: %{{x:.0f}} m²<br>S: %{{y:.5f}}",
    ), row=1, col=col_i+1)
    # Regression line
    A_fit = np.logspace(np.log10(min(A_plt)), np.log10(max(A_plt)), 50)
    S_fit = (10 ** tres['ks_steepness']) * A_fit ** (-tres['theta_concavity'])
    fig.add_trace(go.Scatter(
        x=A_fit.tolist(), y=S_fit.tolist(), mode='lines',
        name=f"θ={tres['theta_concavity']:.2f}",
        line=dict(dash='dash', color='firebrick'),
    ), row=1, col=col_i+1)
    fig.update_xaxes(type='log', title_text='Drainage Area (m²)', row=1, col=col_i+1)
    fig.update_yaxes(type='log', title_text='Slope',              row=1, col=col_i+1)

fig.update_layout(title="Slope–Area Relationship (Channel Concavity)",
                  template='plotly_white', height=450, showlegend=False)
save_fig(fig, "11d_slope_area_SA")

# SL anomaly bar per basin
if len(SL_per_basin) > 0:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=SL_per_basin.index.tolist(),
        y=SL_per_basin['SL_mean'].tolist(),
        error_y=dict(type='data', array=SL_per_basin['SL_std'].tolist()),
        name='SL mean ± std',
        marker_color=px.colors.qualitative.Plotly[:len(SL_per_basin)],
        text=[f"max={v:.0f}" for v in SL_per_basin['SL_max']],
        textposition='outside',
        hovertemplate='%{x}<br>SL mean: %{y:.1f}<br>%{text}',
    ))
    fig.add_hline(y=SL_mean, line_dash='dash', line_color='red',
                  annotation_text=f"Watershed mean SL={SL_mean:.1f}")
    fig.update_layout(
        title="Stream Length Gradient Index (SL) per Subbasin<br>"
              "<sup>Values > 2× mean suggest structural/lithological anomaly</sup>",
        xaxis_title="Subbasin", yaxis_title="SL Index",
        template='plotly_white', height=450,
    )
    save_fig(fig, "11e_SL_bar_chart")

print("\n✅ SECTION 11 complete.")
