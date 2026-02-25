"""
scripts/validate_data.py
========================
Validates GIS input files before running the main pipeline.
Used by validate_data.yml GitHub Actions workflow.
"""

import os, sys, glob, json
from pathlib import Path

DATA_DIR   = os.environ.get('DATA_DIR', 'data')
REPORT_OUT = 'outputs/validation_report.txt'
os.makedirs('outputs', exist_ok=True)

issues, warnings, passed = [], [], []

def check(label, condition, msg_pass, msg_fail, is_warning=False):
    if condition:
        passed.append(f"  ✅ {label}: {msg_pass}")
    else:
        entry = f"  {'⚠️' if is_warning else '❌'} {label}: {msg_fail}"
        (warnings if is_warning else issues).append(entry)

# ── 1. DATA DIR EXISTS ───────────────────────────────────────────────────────
check("data/ directory", os.path.isdir(DATA_DIR),
      "exists", "data/ directory not found")

all_files = list(Path(DATA_DIR).rglob('*')) if os.path.isdir(DATA_DIR) else []
tifs  = [f for f in all_files if f.suffix.lower() in ('.tif', '.tiff')]
shps  = [f for f in all_files if f.suffix.lower() == '.shp']
zips  = [f for f in all_files if f.suffix.lower() == '.zip']

check("DEM file (.tif)",
      any('dem' in f.name.lower() or 'srtm' in f.name.lower() or 'elev' in f.name.lower() for f in tifs),
      "detected", "No DEM detected (expected: dem.tif or srtm*.tif)")

check("Subbasin shapefile",
      any('sub' in f.name.lower() or 'basin' in f.name.lower() or 'catch' in f.name.lower() for f in shps),
      "detected", "No subbasin shapefile detected")

check("Stream network shapefile",
      any('stream' in f.name.lower() or 'river' in f.name.lower() or 'channel' in f.name.lower() for f in shps),
      "detected", "No stream shapefile detected", is_warning=True)

check("Flow direction raster",
      any('dir' in f.name.lower() or 'fdir' in f.name.lower() for f in tifs),
      "detected", "No flow direction raster detected", is_warning=True)

check("Flow accumulation raster",
      any('acc' in f.name.lower() or 'facc' in f.name.lower() for f in tifs),
      "detected", "No flow accumulation raster detected", is_warning=True)

check("ZIP archive",
      len(zips) > 0,
      f"found {zips[0].name}", "No ZIP found (optional — individual files are fine)", is_warning=True)

# ── 2. CRS / METADATA CHECK ──────────────────────────────────────────────────
try:
    import rasterio
    for tif in tifs[:3]:
        with rasterio.open(tif) as src:
            crs   = src.crs
            res_x = src.res[0]
            check(f"DEM CRS ({tif.name})",
                  crs is not None,
                  f"EPSG:{crs.to_epsg()}", "No CRS found")
            check(f"DEM resolution ({tif.name})",
                  15 <= res_x <= 100,
                  f"{res_x:.1f} m", f"Unexpected resolution {res_x:.1f} m (expected ~30 m)", is_warning=True)
except ImportError:
    warnings.append("  ⚠️ rasterio not installed — skipping CRS checks")
except Exception as e:
    warnings.append(f"  ⚠️ CRS check failed: {e}")

try:
    import geopandas as gpd
    for shp in shps:
        gdf = gpd.read_file(shp)
        check(f"Geometry validity ({shp.name})",
              gdf.geometry.is_valid.all(),
              "all valid", f"{(~gdf.geometry.is_valid).sum()} invalid geometries", is_warning=True)
        check(f"CRS set ({shp.name})",
              gdf.crs is not None,
              str(gdf.crs), "No CRS", is_warning=True)
        if 'sub' in shp.name.lower() or 'basin' in shp.name.lower():
            check("Subbasin count",
                  len(gdf) == 5,
                  f"{len(gdf)} subbasins ✓", f"{len(gdf)} subbasins (expected 5)")
except ImportError:
    warnings.append("  ⚠️ geopandas not installed — skipping shapefile checks")
except Exception as e:
    warnings.append(f"  ⚠️ Shapefile check failed: {e}")

# ── REPORT ───────────────────────────────────────────────────────────────────
lines = [
    "=" * 60,
    "DATA VALIDATION REPORT",
    "=" * 60,
    "",
    f"Data directory : {os.path.abspath(DATA_DIR)}",
    f"Rasters found  : {len(tifs)}",
    f"Shapefiles     : {len(shps)}",
    f"ZIP archives   : {len(zips)}",
    "",
    f"✅ PASSED ({len(passed)})",
    *passed,
    "",
    f"⚠️  WARNINGS ({len(warnings)})",
    *warnings,
    "",
    f"❌ ERRORS ({len(issues)})",
    *issues,
    "",
    "=" * 60,
    "VERDICT: " + ("PASS — ready to run pipeline" if not issues else "FAIL — fix errors above"),
    "=" * 60,
]
report = "\n".join(lines)
print(report)

with open(REPORT_OUT, 'w') as f:
    f.write(report)

sys.exit(0 if not issues else 1)
