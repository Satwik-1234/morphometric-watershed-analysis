"""
=============================================================================
SECTION 4 — PUBLICATION-GRADE MAPS
=============================================================================
Generates 9 maps, all with:
  • Hillshade background
  • DMS (°′″) grid
  • North arrow
  • Scale bar
  • Subbasin boundaries overlay
  • Stream network overlay
  • Colourbar / legend
  • Title

Maps produced:
  1. Elevation (DEM)
  2. Slope
  3. Aspect
  4. Flow Direction
  5. Flow Accumulation
  6. Stream Order (Strahler)
  7. Drainage Density
  8. Contour
  9. Pour Points (snapped) on DEM
=============================================================================
"""

print("=" * 60)
print("SECTION 4 — MAP GENERATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED MAP UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

from pyproj import Transformer as PyTransformer

# Transform from UTM → WGS84 for grid labelling
_to_geo = PyTransformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)


def dd_to_dms(dd, is_lat=True):
    """Decimal degrees → DMS string."""
    hemi = ("N" if dd >= 0 else "S") if is_lat else ("E" if dd >= 0 else "W")
    dd = abs(dd)
    deg = int(dd)
    mins_full = (dd - deg) * 60
    mins = int(mins_full)
    secs = (mins_full - mins) * 60
    return f"{deg}°{mins:02d}′{secs:04.1f}″{hemi}"


def get_dms_ticks(utm_extent, n=5):
    """
    Return (x_utm_ticks, x_labels, y_utm_ticks, y_labels)
    for DMS-formatted grid lines.
    utm_extent = (xmin, xmax, ymin, ymax) in UTM metres
    """
    xmin, xmax, ymin, ymax = utm_extent
    # Sample grid corners in geographic
    corners_utm = [
        (xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax),
    ]
    lon_all, lat_all = [], []
    for xu, yu in corners_utm:
        lo, la = _to_geo.transform(xu, yu)
        lon_all.append(lo)
        lat_all.append(la)
    lon_min, lon_max = min(lon_all), max(lon_all)
    lat_min, lat_max = min(lat_all), max(lat_all)

    # Nicely spaced geographic ticks
    lon_ticks_geo = np.linspace(lon_min, lon_max, n)
    lat_ticks_geo = np.linspace(lat_min, lat_max, n)

    # Convert back to UTM for pyplot ticks
    from pyproj import Transformer as T2
    _to_utm = T2.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    x_ticks_utm = [_to_utm.transform(lo, (lat_min + lat_max) / 2)[0] for lo in lon_ticks_geo]
    y_ticks_utm = [_to_utm.transform((lon_min + lon_max) / 2, la)[1] for la in lat_ticks_geo]

    x_labels = [dd_to_dms(lo, is_lat=False) for lo in lon_ticks_geo]
    y_labels = [dd_to_dms(la, is_lat=True)  for la in lat_ticks_geo]

    return x_ticks_utm, x_labels, y_ticks_utm, y_labels


def compute_utm_extent():
    """Return (xmin, xmax, ymin, ymax) from DEM bounds."""
    b = DEM_BOUNDS
    return b.left, b.right, b.bottom, b.top


def add_north_arrow(ax, x=0.96, y=0.94, size=0.045):
    """Add a north arrow to axes using annotation."""
    ax.annotate(
        '', xy=(x, y), xycoords='axes fraction',
        xytext=(x, y - size * 2),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='black', lw=2),
        annotation_clip=False,
    )
    ax.text(x, y + 0.005, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=13,
            fontweight='bold', color='black',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])


def add_scale_bar(ax, extent_m, frac=0.2, y_pos=0.04, x_pos=0.05):
    """
    Add a scale bar. extent_m = (xmin, xmax, ymin, ymax) in metres.
    """
    xmin, xmax = extent_m[0], extent_m[1]
    width_m = (xmax - xmin) * frac

    # Round to nice number
    magnitude = 10 ** np.floor(np.log10(width_m))
    width_m   = round(width_m / magnitude) * magnitude

    # In axes fraction
    total_m = xmax - xmin
    bar_frac = width_m / total_m

    label_km = f"{width_m/1000:.0f} km" if width_m >= 1000 else f"{width_m:.0f} m"

    ax.annotate(
        '', xy=(x_pos + bar_frac, y_pos), xycoords='axes fraction',
        xytext=(x_pos, y_pos), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='<->', color='black', lw=2),
        annotation_clip=False,
    )
    ax.text(x_pos + bar_frac / 2, y_pos + 0.02,
            label_km, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, color='black',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def apply_dms_grid(ax, utm_extent, n_ticks=5):
    """Apply DMS-labelled grid to axes."""
    x_ticks, x_labels, y_ticks, y_labels = get_dms_ticks(utm_extent, n=n_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=7.5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.grid(True, linestyle='--', linewidth=0.4, color='grey', alpha=0.6)
    ax.tick_params(direction='in', top=True, right=True, length=4)


def base_axes(title, figsize=(11, 9)):
    """Create figure/axes with hillshade background."""
    fig, ax = plt.subplots(figsize=figsize)
    utm_extent = compute_utm_extent()
    # Hillshade background
    ax.imshow(
        HILLSHADE,
        extent=[utm_extent[0], utm_extent[1], utm_extent[2], utm_extent[3]],
        origin='upper', cmap='Greys', alpha=0.45,
        aspect='auto', zorder=0,
    )
    ax.set_xlim(utm_extent[0], utm_extent[1])
    ax.set_ylim(utm_extent[2], utm_extent[3])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    return fig, ax, utm_extent


def overlay_boundaries(ax, alpha_sub=0.9, alpha_str=0.5):
    """Overlay subbasin boundaries and stream network."""
    gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2,
                          zorder=10, label='Subbasin boundary')
    if len(gdf_streams) > 0:
        gdf_streams.plot(ax=ax, color='royalblue', linewidth=0.8,
                         alpha=alpha_str, zorder=9, label='Stream network')


def finalize_and_save(fig, ax, utm_extent, filename, n_ticks=5):
    """Apply grid, north arrow, scale bar, tight layout, save."""
    apply_dms_grid(ax, utm_extent, n_ticks)
    add_north_arrow(ax)
    add_scale_bar(ax, utm_extent)
    plt.tight_layout()
    out_path = os.path.join(MAPS_DIR, filename)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  MAP HELPER — raster_to_plot array
# ─────────────────────────────────────────────────────────────────────────────

def raster_extent():
    b = DEM_BOUNDS
    return [b.left, b.right, b.bottom, b.top]


# ─────────────────────────────────────────────────────────────────────────────
#  1. ELEVATION MAP
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/9] Elevation map...")
fig, ax, utm_ext = base_axes("Elevation Map — SRTM 30 m DEM")
cmap_elev = plt.get_cmap('terrain')
im = ax.imshow(
    DEM_ARR,
    extent=raster_extent(), origin='upper',
    cmap=cmap_elev, alpha=0.75, zorder=1,
    vmin=np.nanpercentile(DEM_ARR, 2), vmax=np.nanpercentile(DEM_ARR, 98),
)
overlay_boundaries(ax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.07)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("Elevation (m)", fontsize=10)
ax.legend(loc='lower left', fontsize=8, framealpha=0.8)
finalize_and_save(fig, ax, utm_ext, "01_elevation.png")

# ─────────────────────────────────────────────────────────────────────────────
#  2. SLOPE MAP
# ─────────────────────────────────────────────────────────────────────────────

print("[2/9] Slope map...")
fig, ax, utm_ext = base_axes("Slope Map (degrees)")
im = ax.imshow(
    SLOPE_ARR,
    extent=raster_extent(), origin='upper',
    cmap='YlOrRd', alpha=0.75, zorder=1,
    vmin=0, vmax=np.nanpercentile(SLOPE_ARR, 98),
)
overlay_boundaries(ax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.07)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("Slope (°)", fontsize=10)
finalize_and_save(fig, ax, utm_ext, "02_slope.png")

# ─────────────────────────────────────────────────────────────────────────────
#  3. ASPECT MAP
# ─────────────────────────────────────────────────────────────────────────────

print("[3/9] Aspect map...")
fig, ax, utm_ext = base_axes("Aspect Map (degrees from North)")
cmap_aspect = plt.get_cmap('hsv')
im = ax.imshow(
    ASPECT_ARR,
    extent=raster_extent(), origin='upper',
    cmap=cmap_aspect, alpha=0.75, zorder=1,
    vmin=0, vmax=360,
)
overlay_boundaries(ax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.07)
cb  = plt.colorbar(im, cax=cax)
cb.set_ticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
cb.set_ticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
cb.set_label("Aspect", fontsize=10)
finalize_and_save(fig, ax, utm_ext, "03_aspect.png")

# ─────────────────────────────────────────────────────────────────────────────
#  4. FLOW DIRECTION MAP
# ─────────────────────────────────────────────────────────────────────────────

print("[4/9] Flow direction map...")
fig, ax, utm_ext = base_axes("Flow Direction Map (D8 encoding)")
# D8: 1=E,2=SE,4=S,8=SW,16=W,32=NW,64=N,128=NE
d8_labels = {1:'E',2:'SE',4:'S',8:'SW',16:'W',32:'NW',64:'N',128:'NE'}
unique_d8 = [v for v in sorted(d8_labels.keys()) if v in np.unique(FDIR_ARR[~np.isnan(FDIR_ARR)])]
colors_d8  = plt.cm.tab10(np.linspace(0, 1, 8))
d8_cmap    = mcolors.ListedColormap(colors_d8[:len(unique_d8)])
d8_bounds  = [unique_d8[0] - 0.5] + [v + 0.5 for v in unique_d8]
d8_norm    = mcolors.BoundaryNorm(d8_bounds, d8_cmap.N)

im = ax.imshow(
    FDIR_ARR,
    extent=raster_extent(), origin='upper',
    cmap=d8_cmap, norm=d8_norm, alpha=0.70, zorder=1,
)
overlay_boundaries(ax)
patches_d8 = [mpatches.Patch(color=colors_d8[i], label=d8_labels.get(unique_d8[i], str(unique_d8[i])))
              for i in range(len(unique_d8))]
ax.legend(handles=patches_d8, loc='lower left', fontsize=7,
          title='Flow Dir.', title_fontsize=8, framealpha=0.8, ncol=2)
finalize_and_save(fig, ax, utm_ext, "04_flow_direction.png")

# ─────────────────────────────────────────────────────────────────────────────
#  5. FLOW ACCUMULATION MAP
# ─────────────────────────────────────────────────────────────────────────────

print("[5/9] Flow accumulation map...")
fig, ax, utm_ext = base_axes("Flow Accumulation Map (log₁₀ scale)")
fa_log = np.log10(np.where(FACC_ARR > 0, FACC_ARR, np.nan))
im = ax.imshow(
    fa_log,
    extent=raster_extent(), origin='upper',
    cmap='Blues', alpha=0.80, zorder=1,
)
overlay_boundaries(ax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.07)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("log₁₀(Flow Accum.)", fontsize=10)
finalize_and_save(fig, ax, utm_ext, "05_flow_accumulation.png")

# ─────────────────────────────────────────────────────────────────────────────
#  6. STREAM ORDER MAP (Strahler)
# ─────────────────────────────────────────────────────────────────────────────

print("[6/9] Stream order map...")
fig, ax, utm_ext = base_axes("Strahler Stream Order Map")
# Hillshade already in base_axes
overlay_boundaries(ax, alpha_str=0)   # suppress default streams

orders_list = sorted(gdf_so[ORDER_COL].unique())
order_cmap  = plt.cm.get_cmap('plasma_r', len(orders_list))
order_colors = {o: order_cmap(i) for i, o in enumerate(orders_list)}
lw_map       = {o: 0.5 + (o - 1) * 0.6 for o in orders_list}

for o in orders_list:
    segs = gdf_so[gdf_so[ORDER_COL] == o]
    segs.plot(ax=ax, color=order_colors[o], linewidth=lw_map[o],
              zorder=5 + o, label=f"Order {o}")

gdf_sub.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2, zorder=15)
ax.legend(loc='lower left', fontsize=8, framealpha=0.85, title='Strahler Order')
finalize_and_save(fig, ax, utm_ext, "06_stream_order.png")

# ─────────────────────────────────────────────────────────────────────────────
#  7. DRAINAGE DENSITY MAP
# ─────────────────────────────────────────────────────────────────────────────

print("[7/9] Drainage density map...")
fig, ax, utm_ext = base_axes("Drainage Density Map (km/km²)")
gdf_dd = gdf_sub.merge(
    df_master[['Drainage_Density_Dd']].reset_index(),
    on='basin_id', how='left'
)
gdf_dd.plot(
    column='Drainage_Density_Dd', ax=ax,
    cmap='YlGnBu', legend=True, alpha=0.75, zorder=2,
    legend_kwds={'label': 'Drainage Density (km/km²)', 'shrink': 0.7},
    edgecolor='black', linewidth=1.0,
)
# Basin labels
for _, r in gdf_dd.iterrows():
    cx, cy = r.geometry.centroid.x, r.geometry.centroid.y
    ax.text(cx, cy, f"{r['basin_id']}\n{r['Drainage_Density_Dd']:.2f}",
            ha='center', va='center', fontsize=8, fontweight='bold',
            color='white',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

gdf_streams.plot(ax=ax, color='royalblue', linewidth=0.7, alpha=0.6, zorder=5)
finalize_and_save(fig, ax, utm_ext, "07_drainage_density.png")

# ─────────────────────────────────────────────────────────────────────────────
#  8. CONTOUR MAP
# ─────────────────────────────────────────────────────────────────────────────

print("[8/9] Contour map...")
fig, ax, utm_ext = base_axes("Topographic Contour Map")

b = DEM_BOUNDS
dem_range = np.nanmax(DEM_ARR) - np.nanmin(DEM_ARR)
interval  = max(10, round(dem_range / 20, -1))   # smart interval

x_c = np.linspace(b.left,   b.right,  DEM_ARR.shape[1])
y_c = np.linspace(b.bottom, b.top,    DEM_ARR.shape[0])[::-1]  # origin='upper'
XX, YY = np.meshgrid(x_c, y_c)

contour_levels = np.arange(
    round(np.nanmin(DEM_ARR) / interval) * interval,
    np.nanmax(DEM_ARR) + interval,
    interval,
)
major_levels = contour_levels[::4]

dem_filled_c = np.where(np.isnan(DEM_ARR), np.nanmean(DEM_ARR), DEM_ARR)
cs_minor = ax.contour(XX, YY, dem_filled_c, levels=contour_levels,
                       colors='saddlebrown', linewidths=0.4, alpha=0.5, zorder=3)
cs_major = ax.contour(XX, YY, dem_filled_c, levels=major_levels,
                       colors='saddlebrown', linewidths=1.0, alpha=0.85, zorder=4)
ax.clabel(cs_major, inline=True, fontsize=6.5, fmt='%d m')

overlay_boundaries(ax)
ax.text(0.02, 0.02, f"Contour interval: {interval:.0f} m",
        transform=ax.transAxes, fontsize=8, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
finalize_and_save(fig, ax, utm_ext, "08_contour.png")

# ─────────────────────────────────────────────────────────────────────────────
#  9. POUR POINTS ON DEM
# ─────────────────────────────────────────────────────────────────────────────

print("[9/9] Pour points map...")
fig, ax, utm_ext = base_axes("Pour Points (Snapped) on DEM")
im = ax.imshow(
    DEM_ARR,
    extent=raster_extent(), origin='upper',
    cmap='terrain', alpha=0.65, zorder=1,
    vmin=np.nanpercentile(DEM_ARR, 2), vmax=np.nanpercentile(DEM_ARR, 98),
)
overlay_boundaries(ax)

if POUR_POINTS_OK and gdf_pp is not None:
    gdf_pp.plot(ax=ax, color='red', markersize=80, zorder=20,
                label='Snapped pour points', marker='v', edgecolor='white', linewidth=0.8)
    for idx, r in gdf_pp.iterrows():
        label = str(r.get('basin_id', idx))
        ax.annotate(
            label,
            xy=(r.geometry.x, r.geometry.y),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='red', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
        )

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.07)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("Elevation (m)", fontsize=10)
ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
finalize_and_save(fig, ax, utm_ext, "09_pour_points.png")

print(f"\n✅ All 9 maps saved to: {MAPS_DIR}")
