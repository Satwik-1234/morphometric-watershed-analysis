"""
=============================================================================
SECTION 0 — ZIP EXTRACTION & FILE DISCOVERY
=============================================================================
Upload your zip file to Google Colab and run this section first.
It will extract all files and auto-detect the required layers.
=============================================================================
"""

import os
import zipfile
import glob

# ── USER INPUT ────────────────────────────────────────────────────────────────
ZIP_PATH    = "/content/watershed_data.zip"   # ← Change to your zip filename
EXTRACT_DIR = "/content/watershed_data/"
# ─────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path, extract_dir):
    """Extract zip file and list contents."""
    os.makedirs(extract_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"ZIP not found at {zip_path}. "
            "Please upload your zip file to Colab first."
        )
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
        names = z.namelist()
    print(f"✅ Extracted {len(names)} files to {extract_dir}")
    return names


def discover_files(extract_dir):
    """
    Auto-detect required GIS layers from extracted directory.
    Returns a dict of file paths.
    """
    files = {}

    # Walk all subdirectories
    all_files = []
    for root, dirs, fnames in os.walk(extract_dir):
        for f in fnames:
            all_files.append(os.path.join(root, f))

    print("\n📂 All extracted files:")
    for f in all_files:
        print(f"   {f}")

    # ── RASTERS (.tif / .img / .asc) ─────────────────────────────────────────
    rasters = [f for f in all_files if f.lower().endswith(('.tif', '.tiff', '.img', '.asc'))]

    # Keyword-based auto-detection (case-insensitive)
    for r in rasters:
        base = os.path.basename(r).lower()
        if any(k in base for k in ['dem', 'srtm', 'elevation', 'filled', 'fill']):
            files['dem'] = r
        elif any(k in base for k in ['flowdir', 'flow_dir', 'fdir', 'direction']):
            files['flow_dir'] = r
        elif any(k in base for k in ['flowacc', 'flow_acc', 'facc', 'accumulation']):
            files['flow_acc'] = r
        elif any(k in base for k in ['strahler', 'streamorder', 'stream_order', 'order']):
            files['stream_order_raster'] = r
        elif any(k in base for k in ['slope']):
            files['slope'] = r
        elif any(k in base for k in ['aspect']):
            files['aspect'] = r

    # ── VECTORS (.shp) ────────────────────────────────────────────────────────
    shapefiles = [f for f in all_files if f.lower().endswith('.shp')]

    for s in shapefiles:
        base = os.path.basename(s).lower()
        if any(k in base for k in ['subbasin', 'sub_basin', 'watershed', 'basin', 'catchment']):
            files['subbasins'] = s
        elif any(k in base for k in ['stream', 'river', 'channel', 'network', 'drainage']):
            if 'order' in base:
                files['stream_order_shp'] = s
            else:
                files['streams'] = s
        elif any(k in base for k in ['pour', 'outlet', 'point']):
            files['pour_points'] = s
        elif any(k in base for k in ['order']):
            files['stream_order_shp'] = s

    # ── FALLBACK: if stream_order_shp not found, use streams ─────────────────
    if 'stream_order_shp' not in files and 'streams' in files:
        files['stream_order_shp'] = files['streams']

    print("\n🗺️  Auto-detected layers:")
    for key, val in files.items():
        print(f"   {key:25s} → {val}")

    missing = []
    required = ['dem', 'subbasins', 'streams', 'flow_dir', 'flow_acc']
    for req in required:
        if req not in files:
            missing.append(req)

    if missing:
        print(f"\n⚠️  Could not auto-detect: {missing}")
        print("   Please set paths manually in SECTION 2 — DATA PATHS.")
    else:
        print("\n✅ All required layers detected.")

    return files


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_zip(ZIP_PATH, EXTRACT_DIR)
    DETECTED_FILES = discover_files(EXTRACT_DIR)

    # Print for copy-paste into Section 2
    print("\n" + "="*60)
    print("📋 Copy these paths into SECTION 2 — DATA PATHS:")
    print("="*60)
    for k, v in DETECTED_FILES.items():
        print(f'  "{k}": r"{v}",')
