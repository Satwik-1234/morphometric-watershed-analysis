"""
=============================================================================
SECTION 2 — DATA LOADING, CRS VALIDATION & PREPROCESSING
=============================================================================
Loads all GIS layers, validates CRS, reprojects to UTM, cleans geometry.
EDIT the DATA_PATHS dict below with your actual file paths from Section 0.
=============================================================================
"""

# ── ▼▼▼  EDIT THESE PATHS  ▼▼▼ ────────────────────────────────────────────────
DATA_PATHS = {
    "dem"              : r"/content/watershed_data/dem.tif",
    "subbasins"        : r"/content/watershed_data/subbasins.shp",
    "streams"          : r"/content/watershed_data/streams.shp",
    "stream_order_shp" : r"/content/watershed_data/stream_order.shp",
    "flow_dir"         : r"/content/watershed_data/flow_direction.tif",
    "flow_acc"         : r"/content/watershed_data/flow_accumulation.tif",
    "pour_points"      : r"/content/watershed_data/pour_points.shp",  # optional
}
# ── ▲▲▲  EDIT ABOVE  ▲▲▲ ──────────────────────────────────────────────────────

N_SUBBASINS = 5   # Expected number of subbasins

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_utm_epsg(lon, lat):
    """Return appropriate UTM EPSG code for a given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"


def get_raster_info(path):
    """Return dict of raster metadata."""
    with rasterio.open(path) as src:
        return {
            "crs"        : src.crs,
            "res"        : src.res,
            "nodata"     : src.nodata,
            "shape"      : (src.height, src.width),
            "bounds"     : src.bounds,
            "dtype"      : src.dtypes[0],
            "count"      : src.count,
            "transform"  : src.transform,
        }


def reproject_raster(src_path, dst_path, target_crs):
    """Reproject a raster to target CRS and save."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs'       : target_crs,
            'transform' : transform,
            'width'     : width,
            'height'    : height,
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
    return dst_path


def fix_geometries(gdf, layer_name="layer"):
    """Fix invalid geometries and remove nulls."""
    before = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf['geometry'] = gdf['geometry'].apply(
        lambda g: g.buffer(0) if not g.is_valid else g
    )
    gdf = gdf[gdf.geometry.is_valid].copy()
    print(f"  {layer_name}: {before} → {len(gdf)} features (after geometry fix)")
    return gdf.reset_index(drop=True)


def explode_multipart(gdf, layer_name="layer"):
    """Explode multipart geometries to single-part."""
    before = len(gdf)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    if len(gdf) != before:
        print(f"  {layer_name}: Exploded multipart → {len(gdf)} parts")
    return gdf


def snap_pour_points(pour_pts_gdf, flow_acc_path, snap_distance_m=300):
    """
    Snap pour points to the highest flow accumulation cell
    within snap_distance_m (in metres, projected CRS assumed).
    Returns GeoDataFrame with snapped geometries.
    """
    with rasterio.open(flow_acc_path) as src:
        fa_data  = src.read(1).astype(float)
        nodata   = src.nodata if src.nodata is not None else -9999
        fa_data[fa_data == nodata] = np.nan
        transform = src.transform
        res        = src.res[0]  # metres per pixel

    snap_cells = int(snap_distance_m / res)
    snapped_pts = []

    for idx, row in pour_pts_gdf.iterrows():
        px_c, px_r = ~transform * (row.geometry.x, row.geometry.y)
        px_c, px_r = int(px_c), int(px_r)

        r0 = max(0, px_r - snap_cells)
        r1 = min(fa_data.shape[0], px_r + snap_cells + 1)
        c0 = max(0, px_c - snap_cells)
        c1 = min(fa_data.shape[1], px_c + snap_cells + 1)

        window = fa_data[r0:r1, c0:c1]
        if np.all(np.isnan(window)):
            snapped_pts.append(row.geometry)
            continue

        local_max = np.nanargmax(window)
        local_r, local_c = np.unravel_index(local_max, window.shape)
        global_r = r0 + local_r
        global_c = c0 + local_c

        snap_x, snap_y = xy(transform, global_r, global_c)
        snapped_pts.append(Point(snap_x, snap_y))

    result = pour_pts_gdf.copy()
    result['geometry']       = snapped_pts
    result['snap_distance_m'] = [
        row.geometry.distance(snapped_pts[i])
        for i, (_, row) in enumerate(pour_pts_gdf.iterrows())
    ]
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 2 — DATA LOADING & PREPROCESSING")
print("=" * 60)

# ── 1. Load DEM info first to determine UTM zone ──────────────────────────────
print("\n[1/6] Reading DEM metadata...")
assert os.path.exists(DATA_PATHS['dem']), f"DEM not found: {DATA_PATHS['dem']}"
dem_info = get_raster_info(DATA_PATHS['dem'])
print(f"  CRS      : {dem_info['crs']}")
print(f"  Res      : {dem_info['res']} m")
print(f"  Shape    : {dem_info['shape']}")
print(f"  Bounds   : {dem_info['bounds']}")
print(f"  No-data  : {dem_info['nodata']}")

# Determine if geographic or projected
src_crs = CRS.from_user_input(dem_info['crs'])
if src_crs.is_geographic:
    # Compute centroid lon/lat for UTM zone
    b = dem_info['bounds']
    cen_lon = (b.left + b.right) / 2
    cen_lat = (b.bottom + b.top) / 2
    UTM_EPSG = detect_utm_epsg(cen_lon, cen_lat)
    print(f"  DEM is geographic → will reproject to {UTM_EPSG}")
    NEEDS_REPROJECT = True
else:
    UTM_EPSG = str(dem_info['crs'])
    print(f"  DEM is already projected: {UTM_EPSG}")
    NEEDS_REPROJECT = False

TARGET_CRS = CRS.from_epsg(int(UTM_EPSG.split(":")[1]))

# ── 2. Reproject rasters if needed ───────────────────────────────────────────
print("\n[2/6] Reprojecting rasters...")
RASTER_KEYS = ['dem', 'flow_dir', 'flow_acc']
RASTERS = {}

for key in RASTER_KEYS:
    src_path = DATA_PATHS[key]
    assert os.path.exists(src_path), f"Missing: {src_path}"
    info = get_raster_info(src_path)
    if NEEDS_REPROJECT and CRS.from_user_input(info['crs']).is_geographic:
        dst_path = os.path.join(OUT_DIR, f"{key}_utm.tif")
        reproject_raster(src_path, dst_path, TARGET_CRS)
        RASTERS[key] = dst_path
        print(f"  ✅ Reprojected {key}")
    else:
        RASTERS[key] = src_path
        print(f"  ✅ {key} OK (already projected)")

# Optional stream order raster
if os.path.exists(DATA_PATHS.get('stream_order_raster', '')):
    so_path = DATA_PATHS['stream_order_raster']
    so_info = get_raster_info(so_path)
    if NEEDS_REPROJECT and CRS.from_user_input(so_info['crs']).is_geographic:
        dst = os.path.join(OUT_DIR, "stream_order_utm.tif")
        reproject_raster(so_path, dst, TARGET_CRS)
        RASTERS['stream_order_raster'] = dst
    else:
        RASTERS['stream_order_raster'] = so_path

# ── 3. Load & validate vector layers ─────────────────────────────────────────
print("\n[3/6] Loading vector layers...")

# Subbasins
gdf_sub = gpd.read_file(DATA_PATHS['subbasins'])
gdf_sub = fix_geometries(gdf_sub, "subbasins")
gdf_sub = gdf_sub.to_crs(UTM_EPSG)
assert len(gdf_sub) == N_SUBBASINS, (
    f"Expected {N_SUBBASINS} subbasins, got {len(gdf_sub)}. "
    "Update N_SUBBASINS or check your shapefile."
)
print(f"  ✅ Subbasins: {len(gdf_sub)} | CRS: {gdf_sub.crs}")

# Ensure unique basin ID
if 'basin_id' not in gdf_sub.columns:
    gdf_sub['basin_id'] = [f"SB{i+1}" for i in range(len(gdf_sub))]
print(f"  Basin IDs: {gdf_sub['basin_id'].tolist()}")

# Streams
gdf_streams = gpd.read_file(DATA_PATHS['streams'])
gdf_streams = fix_geometries(gdf_streams, "streams")
gdf_streams = explode_multipart(gdf_streams, "streams")
gdf_streams = gdf_streams.to_crs(UTM_EPSG)
print(f"  ✅ Streams: {len(gdf_streams)} segments | CRS: {gdf_streams.crs}")

# Stream order shapefile
gdf_so = gpd.read_file(DATA_PATHS['stream_order_shp'])
gdf_so = fix_geometries(gdf_so, "stream_order")
gdf_so = explode_multipart(gdf_so, "stream_order")
gdf_so = gdf_so.to_crs(UTM_EPSG)

# Detect stream order column
ORDER_COL = None
for col in gdf_so.columns:
    if any(k in col.lower() for k in ['order', 'strahler', 'storder', 'strord']):
        ORDER_COL = col
        break
if ORDER_COL is None:
    raise ValueError(
        f"Cannot detect stream order column. Columns: {gdf_so.columns.tolist()}\n"
        "Please set ORDER_COL manually below."
    )
print(f"  ✅ Stream order col detected: '{ORDER_COL}' "
      f"| Orders: {sorted(gdf_so[ORDER_COL].unique())}")

gdf_so[ORDER_COL] = gdf_so[ORDER_COL].astype(int)
MAX_ORDER = int(gdf_so[ORDER_COL].max())

# Pour points (optional but important for snapping)
POUR_POINTS_OK = False
if os.path.exists(DATA_PATHS.get('pour_points', '')):
    gdf_pp = gpd.read_file(DATA_PATHS['pour_points'])
    gdf_pp = gdf_pp.to_crs(UTM_EPSG)
    print(f"  ✅ Pour points: {len(gdf_pp)}")
    print("  Snapping pour points to max flow accumulation...")
    gdf_pp = snap_pour_points(gdf_pp, RASTERS['flow_acc'], snap_distance_m=300)
    print(f"  Snap distances (m): {gdf_pp['snap_distance_m'].round(1).tolist()}")
    gdf_pp.to_file(os.path.join(SHAPES_DIR, "pour_points_snapped.shp"))
    POUR_POINTS_OK = True
else:
    gdf_pp = None
    print("  ⚠️  Pour points file not found — skipping snap")

# ── 4. Validate DEM resolution ───────────────────────────────────────────────
print("\n[4/6] Validating DEM resolution...")
dem_info_utm = get_raster_info(RASTERS['dem'])
res_x, res_y = dem_info_utm['res']
assert 20 <= res_x <= 35, (
    f"DEM resolution {res_x:.1f}m — expected ~30m (SRTM). "
    "Check your DEM file."
)
print(f"  ✅ DEM resolution: {res_x:.1f} x {res_y:.1f} m ≈ 30 m SRTM ✓")

# ── 5. Read raster arrays into memory ────────────────────────────────────────
print("\n[5/6] Reading raster arrays...")

with rasterio.open(RASTERS['dem']) as src:
    DEM_ARR       = src.read(1).astype(np.float32)
    DEM_NODATA    = src.nodata if src.nodata is not None else -9999.0
    DEM_TRANSFORM = src.transform
    DEM_CRS       = src.crs
    DEM_BOUNDS    = src.bounds
    DEM_RES       = src.res[0]
    DEM_ARR[DEM_ARR == DEM_NODATA] = np.nan

with rasterio.open(RASTERS['flow_dir']) as src:
    FDIR_ARR    = src.read(1).astype(np.float32)
    FDIR_NODATA = src.nodata if src.nodata is not None else -9999.0
    FDIR_ARR[FDIR_ARR == FDIR_NODATA] = np.nan

with rasterio.open(RASTERS['flow_acc']) as src:
    FACC_ARR    = src.read(1).astype(np.float32)
    FACC_NODATA = src.nodata if src.nodata is not None else -9999.0
    FACC_ARR[FACC_ARR == FACC_NODATA] = np.nan

print(f"  DEM  shape: {DEM_ARR.shape} | min={np.nanmin(DEM_ARR):.1f} max={np.nanmax(DEM_ARR):.1f} m")
print(f"  FDIR shape: {FDIR_ARR.shape}")
print(f"  FACC shape: {FACC_ARR.shape}")

# ── 6. Compute slope & aspect if not provided ─────────────────────────────────
print("\n[6/6] Computing slope and aspect...")

def compute_slope_aspect_numpy(dem, res_m):
    """Compute slope (degrees) and aspect (degrees) using numpy gradient."""
    # Smooth first to reduce noise
    from scipy.ndimage import uniform_filter
    dem_sm = np.where(np.isnan(dem), 0, dem)
    dz_dy, dz_dx = np.gradient(dem_sm, res_m, res_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_deg = np.degrees(np.arctan2(-dz_dx, dz_dy)) % 360
    slope_deg[np.isnan(dem)] = np.nan
    aspect_deg[np.isnan(dem)] = np.nan
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


if RICHDEM_OK:
    try:
        rda = rd.rdarray(np.where(np.isnan(DEM_ARR), -9999, DEM_ARR), no_data=-9999)
        rda.projection = DEM_CRS.to_wkt()
        rda.geotransform = (DEM_TRANSFORM.c, DEM_TRANSFORM.a, 0,
                            DEM_TRANSFORM.f, 0, DEM_TRANSFORM.e)
        SLOPE_ARR  = np.array(rd.TerrainAttribute(rda, attrib='slope_degrees')).astype(np.float32)
        ASPECT_ARR = np.array(rd.TerrainAttribute(rda, attrib='aspect')).astype(np.float32)
        SLOPE_ARR[np.isnan(DEM_ARR)]  = np.nan
        ASPECT_ARR[np.isnan(DEM_ARR)] = np.nan
        print("  ✅ Slope & aspect from richdem")
    except Exception as e:
        print(f"  ⚠️  richdem failed ({e}) — using numpy")
        SLOPE_ARR, ASPECT_ARR = compute_slope_aspect_numpy(DEM_ARR, DEM_RES)
else:
    SLOPE_ARR, ASPECT_ARR = compute_slope_aspect_numpy(DEM_ARR, DEM_RES)
    print("  ✅ Slope & aspect from numpy gradient")

# Save slope & aspect to disk
def save_raster(arr, path, template_path):
    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
    meta.update({'dtype': 'float32', 'nodata': -9999.0, 'count': 1})
    arr_save = np.where(np.isnan(arr), -9999.0, arr)
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(arr_save.astype(np.float32), 1)

save_raster(SLOPE_ARR,  os.path.join(OUT_DIR, "slope.tif"),  RASTERS['dem'])
save_raster(ASPECT_ARR, os.path.join(OUT_DIR, "aspect.tif"), RASTERS['dem'])
RASTERS['slope']  = os.path.join(OUT_DIR, "slope.tif")
RASTERS['aspect'] = os.path.join(OUT_DIR, "aspect.tif")

# ── HILLSHADE (used as background in all maps) ────────────────────────────────
print("  Computing hillshade for map backgrounds...")
ls = LightSource(azdeg=315, altdeg=45)
dem_filled = np.where(np.isnan(DEM_ARR), np.nanmean(DEM_ARR), DEM_ARR)
HILLSHADE   = ls.hillshade(dem_filled, vert_exag=1.5, dx=DEM_RES, dy=DEM_RES)
HILLSHADE[np.isnan(DEM_ARR)] = np.nan
print("  ✅ Hillshade computed")

# ── SPATIAL INDEX (for fast spatial joins) ────────────────────────────────────
print("\n✅ SECTION 2 complete.")
print(f"  Subbasins    : {len(gdf_sub)}")
print(f"  Stream segs  : {len(gdf_streams)}")
print(f"  Stream orders: {sorted(gdf_so[ORDER_COL].unique())}")
print(f"  UTM CRS      : {UTM_EPSG}")
print(f"  DEM range    : {np.nanmin(DEM_ARR):.1f} – {np.nanmax(DEM_ARR):.1f} m")
print(f"  Slope range  : {np.nanmin(SLOPE_ARR):.1f}° – {np.nanmax(SLOPE_ARR):.1f}°")
