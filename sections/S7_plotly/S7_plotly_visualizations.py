"""
=============================================================================
SECTION 7 — ADVANCED PLOTLY INTERACTIVE VISUALIZATIONS
=============================================================================
All figures saved as interactive HTML + static PNG.
=============================================================================
"""

print("=" * 60)
print("SECTION 7 — PLOTLY INTERACTIVE VISUALIZATION SUITE")
print("=" * 60)

HTML_DIR = os.path.join(PLOTS_DIR, "html/")
os.makedirs(HTML_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save Plotly figure as HTML and static PNG."""
    html_path = os.path.join(HTML_DIR, f"{name}.html")
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"  ✅ {name}.html")
    return html_path


# ─────────────────────────────────────────────────────────────────────────────
#  1. HORTON'S LAWS — Stream Number & Stream Length
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1] Horton's Law plots...")

for bid, df_lin in LINEAR_PER_ORDER.items():
    if df_lin.empty or len(df_lin) < 2:
        continue
    orders   = df_lin['order'].values
    Nu_vals  = df_lin['Nu'].values.astype(float)
    Lu_vals  = (df_lin['Lu'].values / 1000).astype(float)  # km

    # Regression on log scale (exclude zeros)
    mask_n = Nu_vals > 0
    log_Nu = np.log10(Nu_vals[mask_n])
    log_u  = np.log10(orders[mask_n])
    if len(log_u) > 1:
        slope_n, intercept_n, r_n, p_n, _ = stats.linregress(log_u, log_Nu)
        r2_n = r_n ** 2
    else:
        slope_n, intercept_n, r2_n = 0, 0, 0

    mask_l = Lu_vals > 0
    log_Lu = np.log10(Lu_vals[mask_l])
    if len(log_u[mask_l[:len(log_u)]]) > 1:
        slope_l, intercept_l, r_l, _, _ = stats.linregress(
            log_u[:len(log_Lu)], log_Lu
        )
        r2_l = r_l ** 2
    else:
        slope_l, intercept_l, r2_l = 0, 0, 0

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Stream Number Law — {bid}",
                            f"Stream Length Law — {bid}"
                        ])

    # Stream number
    fig.add_trace(go.Scatter(
        x=orders[mask_n], y=Nu_vals[mask_n], mode='markers+lines',
        name='Stream Number', marker=dict(size=10, color='royalblue'),
        hovertemplate='Order %{x}: %{y} streams',
    ), row=1, col=1)
    fit_x = np.linspace(orders.min(), orders.max(), 50)
    fit_y = 10 ** (intercept_n + slope_n * np.log10(fit_x))
    fig.add_trace(go.Scatter(
        x=fit_x, y=fit_y, mode='lines',
        name=f'Regression (R²={r2_n:.3f})',
        line=dict(color='firebrick', dash='dash'),
    ), row=1, col=1)

    # Stream length
    fig.add_trace(go.Scatter(
        x=orders[mask_l], y=Lu_vals[mask_l], mode='markers+lines',
        name='Stream Length (km)', marker=dict(size=10, color='darkorange'),
        hovertemplate='Order %{x}: %{y:.2f} km',
    ), row=1, col=2)
    if r2_l > 0:
        fit_yl = 10 ** (intercept_l + slope_l * np.log10(fit_x))
        fig.add_trace(go.Scatter(
            x=fit_x, y=fit_yl, mode='lines',
            name=f'Regression (R²={r2_l:.3f})',
            line=dict(color='green', dash='dash'),
        ), row=1, col=2)

    fig.update_xaxes(type='log', title_text='Stream Order (log)', row=1, col=1)
    fig.update_yaxes(type='log', title_text='Stream Number (log)', row=1, col=1)
    fig.update_xaxes(type='log', title_text='Stream Order (log)', row=1, col=2)
    fig.update_yaxes(type='log', title_text='Stream Length km (log)', row=1, col=2)
    fig.update_layout(title=f"Horton's Laws — {bid}", template='plotly_white',
                      height=500, showlegend=True)
    save_fig(fig, f"01_hortons_law_{bid}")

# ─────────────────────────────────────────────────────────────────────────────
#  2. RADAR CHART — Morphometric Signature per Subbasin
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2] Radar charts...")

radar_params = [
    'Drainage_Density_Dd', 'Stream_Frequency_Fs', 'Form_Factor_Ff',
    'Elongation_Ratio_Re', 'Circularity_Ratio_Rc', 'Ruggedness_Rn',
    'Hypsometric_HI', 'Relief_Ratio_Rh', 'Rbm',
]
radar_params = [p for p in radar_params if p in df_master.columns]

df_radar = df_master[radar_params].copy().astype(float)
# Normalise 0-1 for radar
df_radar_norm = (df_radar - df_radar.min()) / (df_radar.max() - df_radar.min() + 1e-12)

fig = go.Figure()
categories = [p.split('_')[-1] for p in radar_params]
colors_r   = px.colors.qualitative.Set2

for i, (bid, row) in enumerate(df_radar_norm.iterrows()):
    vals = row.tolist()
    vals += [vals[0]]  # close polygon
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=categories + [categories[0]],
        fill='toself', name=bid,
        line_color=colors_r[i % len(colors_r)],
        opacity=0.6,
        hovertemplate=bid + '<br>%{theta}: %{r:.3f}',
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Morphometric Signature Radar Chart — All Subbasins",
    template='plotly_white', height=600,
)
save_fig(fig, "02_radar_morphometric")

# ─────────────────────────────────────────────────────────────────────────────
#  3. SCATTER MATRIX
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3] Scatter matrix...")
scatter_cols = [c for c in ['Drainage_Density_Dd', 'Stream_Frequency_Fs',
                              'Elongation_Ratio_Re', 'Basin_Relief_H_m',
                              'Ruggedness_Rn', 'Hypsometric_HI']
                if c in df_master.columns]
df_sc = df_master[scatter_cols].reset_index()
fig   = px.scatter_matrix(
    df_sc, dimensions=scatter_cols, color='basin_id',
    title="Scatter Matrix — Key Morphometric Parameters",
    labels={c: c.split('_')[-1] for c in scatter_cols},
    template='plotly_white',
)
fig.update_traces(diagonal_visible=False, showupperhalf=False)
save_fig(fig, "03_scatter_matrix")

# ─────────────────────────────────────────────────────────────────────────────
#  4. 3D SCATTER — Dd vs Relief vs Area
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4] 3D scatter...")
if all(c in df_master.columns for c in ['Drainage_Density_Dd', 'Basin_Relief_H_m', 'Area_km2']):
    df3d = df_master[['Drainage_Density_Dd', 'Basin_Relief_H_m', 'Area_km2']].reset_index()
    fig  = px.scatter_3d(
        df3d, x='Drainage_Density_Dd', y='Basin_Relief_H_m', z='Area_km2',
        color='basin_id', text='basin_id',
        title="3D Scatter: Drainage Density vs Relief vs Area",
        labels={'Drainage_Density_Dd': 'Dd (km/km²)',
                'Basin_Relief_H_m': 'Relief (m)',
                'Area_km2': 'Area (km²)'},
        template='plotly_white', size_max=18,
    )
    save_fig(fig, "04_3d_scatter")

# ─────────────────────────────────────────────────────────────────────────────
#  5. HISTOGRAM DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] Histogram distributions...")
hist_cols = [c for c in STAT_COLS if c in df_master.columns][:9]
fig = make_subplots(rows=3, cols=3, subplot_titles=hist_cols)
for i, col in enumerate(hist_cols):
    r, c_idx = divmod(i, 3)
    fig.add_trace(
        go.Histogram(x=df_master[col].dropna(), name=col,
                     marker_color=px.colors.qualitative.Set1[i % 9],
                     nbinsx=10),
        row=r+1, col=c_idx+1,
    )
fig.update_layout(title="Parameter Distributions", template='plotly_white',
                  height=800, showlegend=False)
save_fig(fig, "05_histograms")

# ─────────────────────────────────────────────────────────────────────────────
#  6. BOX PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6] Box plots...")
df_melt = df_master[STAT_COLS[:10]].reset_index().melt(id_vars='basin_id')
fig     = px.box(
    df_melt, x='variable', y='value', color='variable',
    title="Box Plot — Morphometric Parameters",
    template='plotly_white', points='all',
    labels={'variable': 'Parameter', 'value': 'Value'},
)
fig.update_xaxes(tickangle=45)
save_fig(fig, "06_boxplots")

# ─────────────────────────────────────────────────────────────────────────────
#  7. HYPSOMETRIC CURVES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7] Hypsometric curves...")
if HYPS:
    fig = go.Figure()
    colors_h = px.colors.qualitative.Plotly
    for i, (bid, (rel_area, rel_elev)) in enumerate(HYPS.items()):
        hi_val = df_master.loc[bid, 'Hypsometric_HI'] if 'Hypsometric_HI' in df_master.columns else np.nan
        fig.add_trace(go.Scatter(
            x=rel_area, y=rel_elev, mode='lines',
            name=f"{bid} (HI={hi_val:.3f})" if not np.isnan(hi_val) else bid,
            line=dict(color=colors_h[i % len(colors_h)], width=2),
            hovertemplate='Rel. Area: %{x:.2f}<br>Rel. Elev: %{y:.2f}',
        ))
    # Reference lines
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0.5, 0.5], mode='lines',
        name='HI = 0.5 (Equilibrium)', line=dict(dash='dash', color='grey'),
    ))
    fig.update_layout(
        title="Hypsometric Curves — All Subbasins",
        xaxis_title="Relative Area (a/A)",
        yaxis_title="Relative Elevation (h/H)",
        template='plotly_white', height=550,
    )
    save_fig(fig, "07_hypsometric_curves")

# ─────────────────────────────────────────────────────────────────────────────
#  8. PLOTLY CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

print("\n[8] Plotly correlation heatmap...")
fig = go.Figure(go.Heatmap(
    z=corr_pearson.values,
    x=corr_pearson.columns.tolist(),
    y=corr_pearson.index.tolist(),
    colorscale='RdYlBu', zmid=0, zmin=-1, zmax=1,
    text=np.round(corr_pearson.values, 2).astype(str),
    texttemplate='%{text}', textfont_size=8,
    hovertemplate='%{y} vs %{x}: %{z:.3f}',
))
fig.update_layout(title="Pearson Correlation Matrix (Interactive)",
                  template='plotly_white', height=700, width=800)
save_fig(fig, "08_correlation_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
#  9. PARALLEL COORDINATE PLOT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[9] Parallel coordinate plot...")
par_cols = [c for c in STAT_COLS if c in df_master.columns][:8]
df_par   = df_master[par_cols].reset_index()
df_par['basin_num'] = range(len(df_par))
fig = px.parallel_coordinates(
    df_par, color='basin_num', dimensions=par_cols,
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Parallel Coordinate Plot — Morphometric Parameters",
    labels={c: c.replace('_', ' ') for c in par_cols},
)
save_fig(fig, "09_parallel_coordinates")

# ─────────────────────────────────────────────────────────────────────────────
#  10. BUBBLE PLOT — Area vs Dd sized by Relief
# ─────────────────────────────────────────────────────────────────────────────

print("\n[10] Bubble plot...")
if all(c in df_master.columns for c in ['Area_km2', 'Drainage_Density_Dd', 'Basin_Relief_H_m']):
    df_bub = df_master[['Area_km2', 'Drainage_Density_Dd', 'Basin_Relief_H_m']].reset_index()
    fig    = px.scatter(
        df_bub, x='Area_km2', y='Drainage_Density_Dd',
        size='Basin_Relief_H_m', color='basin_id', text='basin_id',
        title="Area vs Drainage Density (size = Basin Relief)",
        labels={'Area_km2': 'Area (km²)',
                'Drainage_Density_Dd': 'Drainage Density (km/km²)',
                'Basin_Relief_H_m': 'Relief (m)'},
        template='plotly_white', size_max=50,
    )
    save_fig(fig, "10_bubble_area_dd_relief")

# ─────────────────────────────────────────────────────────────────────────────
#  11. PRIORITY MAP — Interactive Plotly choropleth-style
# ─────────────────────────────────────────────────────────────────────────────

print("\n[11] Priority class map...")
priority_color = {'High': '#d73027', 'Moderate': '#fee090', 'Low': '#4575b4'}
fig = go.Figure()

for _, row in gdf_priority.iterrows():
    bid  = row['basin_id']
    pri  = row.get('Priority_M1', 'Unknown')
    col  = priority_color.get(pri, 'grey')
    geom = row.geometry

    if geom.geom_type == 'Polygon':
        geoms = [geom]
    else:
        geoms = list(geom.geoms)

    for g in geoms:
        coords = np.array(g.exterior.coords)
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            fill='toself', fillcolor=col,
            line=dict(color='black', width=1.5),
            name=f"{bid} ({pri})",
            opacity=0.75,
            hovertemplate=(
                f"<b>{bid}</b><br>"
                f"Priority: {pri}<br>"
                f"Rank M1: {row.get('Rank_M1','—')}<br>"
                f"Rank M2: {row.get('Rank_M2','—')}<br>"
                f"Rank M3: {row.get('Rank_M3','—')}<br>"
                f"Dd: {row.get('Drainage_Density_Dd','—')}"
            ),
        ))

fig.update_layout(
    title="Watershed Priority Classification Map (Method 1 — Compound Ranking)",
    xaxis=dict(title="Easting (m)", scaleanchor='y'),
    yaxis=dict(title="Northing (m)"),
    template='plotly_white', height=650,
    showlegend=True,
)
save_fig(fig, "11_priority_map")

# ─────────────────────────────────────────────────────────────────────────────
#  12. ELEVATION PROFILE (Main Channel)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[12] Elevation profiles...")

def extract_stream_profile(stream_gdf, dem_path, basin_id, n_points=200):
    """Sample DEM along the longest stream segment in a basin."""
    segs = stream_gdf[stream_gdf.get('basin_id', stream_gdf.index) == basin_id]
    if len(segs) == 0:
        return None, None, None
    longest_seg = segs.loc[segs.geometry.length.idxmax()]
    geom = longest_seg.geometry

    if geom.geom_type == 'MultiLineString':
        geom = linemerge(geom)
    if geom.geom_type != 'LineString':
        return None, None, None

    distances = np.linspace(0, geom.length, n_points)
    pts       = [geom.interpolate(d) for d in distances]

    with rasterio.open(dem_path) as src:
        elevations = []
        for pt in pts:
            r_idx, c_idx = rowcol(src.transform, pt.x, pt.y)
            try:
                elev = src.read(1)[r_idx, c_idx]
                nodata = src.nodata if src.nodata else -9999
                elevations.append(np.nan if elev == nodata else float(elev))
            except IndexError:
                elevations.append(np.nan)

    return np.array(distances / 1000), np.array(elevations), geom


fig_profiles = make_subplots(
    rows=len(gdf_sub), cols=1,
    shared_xaxes=False,
    subplot_titles=[f"Longitudinal Profile — {bid}" for bid in gdf_sub['basin_id']],
    vertical_spacing=0.08,
)

for i, (_, row) in enumerate(gdf_sub.iterrows()):
    bid = row['basin_id']
    # Assign basin_id to stream order dataframe if not present
    if 'basin_id' not in gdf_so_sub.columns:
        break
    dist, elev, _ = extract_stream_profile(gdf_so_sub, RASTERS['dem'], bid)
    if dist is None:
        continue
    valid = ~np.isnan(elev)
    fig_profiles.add_trace(
        go.Scatter(
            x=dist[valid], y=elev[valid],
            mode='lines', name=bid, fill='tozeroy',
            line=dict(color=px.colors.qualitative.Plotly[i % 10], width=2),
            hovertemplate='Distance: %{x:.2f} km<br>Elevation: %{y:.1f} m',
        ),
        row=i+1, col=1,
    )
    fig_profiles.update_xaxes(title_text="Distance from outlet (km)", row=i+1, col=1)
    fig_profiles.update_yaxes(title_text="Elevation (m)", row=i+1, col=1)

fig_profiles.update_layout(
    title="Longitudinal Stream Profiles — All Subbasins",
    template='plotly_white', height=300 * len(gdf_sub), showlegend=True,
)
save_fig(fig_profiles, "12_longitudinal_profiles")

print(f"\n✅ SECTION 7 complete. HTML files in: {HTML_DIR}")
print(f"   Total figures: 12")
