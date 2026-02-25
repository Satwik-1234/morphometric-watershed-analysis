"""
scripts/ci_runner.py
====================
CI-aware runner for the full morphometric pipeline.
Called by GitHub Actions auto_visualize.yml.

Reads environment variables set by the workflow:
  DATA_DIR, OUT_DIR, MAPS_DIR, PLOTS_DIR, TABLES_DIR,
  SHAPES_DIR, REPORT_DIR, HTML_DIR

Auto-detects all input file paths inside DATA_DIR.
Runs Sections 0–13 with non-interactive matplotlib backend.
"""

import os, sys, glob, zipfile, traceback
import matplotlib
matplotlib.use('Agg')   # must be before any pyplot import

# ─── resolve paths ──────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get('DATA_DIR',   '/github/workspace/data')
OUT_DIR    = os.environ.get('OUT_DIR',    '/github/workspace/outputs')
MAPS_DIR   = os.environ.get('MAPS_DIR',   os.path.join(OUT_DIR, 'previews'))
PLOTS_DIR  = os.environ.get('PLOTS_DIR',  os.path.join(OUT_DIR, 'previews'))
TABLES_DIR = os.environ.get('TABLES_DIR', os.path.join(OUT_DIR, 'tables'))
SHAPES_DIR = os.environ.get('SHAPES_DIR', os.path.join(OUT_DIR, 'shapefiles'))
REPORT_DIR = os.environ.get('REPORT_DIR', os.path.join(OUT_DIR, 'report'))
HTML_DIR   = os.environ.get('HTML_DIR',   os.path.join(OUT_DIR, 'html'))

for d in [OUT_DIR, MAPS_DIR, PLOTS_DIR, TABLES_DIR, SHAPES_DIR, REPORT_DIR, HTML_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── extract zip if present ──────────────────────────────────────────────────
def extract_zip(data_dir):
    zips = glob.glob(os.path.join(data_dir, '**/*.zip'), recursive=True)
    for z in zips:
        extract_to = os.path.join(data_dir, 'extracted')
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(extract_to)
        print(f"✅ Extracted {z}")

extract_zip(DATA_DIR)

# ─── auto-detect files ───────────────────────────────────────────────────────
def detect(data_dir):
    files = {}
    all_f = []
    for root, _, fnames in os.walk(data_dir):
        for f in fnames:
            all_f.append(os.path.join(root, f))

    for f in all_f:
        base = os.path.basename(f).lower()
        if f.endswith('.tif') or f.endswith('.tiff'):
            if any(k in base for k in ['dem','srtm','elevation','filled','fill']):
                files.setdefault('dem', f)
            elif any(k in base for k in ['flowdir','flow_dir','fdir','direction']):
                files.setdefault('flow_dir', f)
            elif any(k in base for k in ['flowacc','flow_acc','facc','accumulation']):
                files.setdefault('flow_acc', f)
        elif f.endswith('.shp'):
            if any(k in base for k in ['subbasin','sub_basin','watershed','basin','catchment']):
                files.setdefault('subbasins', f)
            elif any(k in base for k in ['stream','river','channel','network','drainage']):
                if 'order' in base:
                    files.setdefault('stream_order_shp', f)
                else:
                    files.setdefault('streams', f)
            elif any(k in base for k in ['pour','outlet','point']):
                files.setdefault('pour_points', f)

    if 'stream_order_shp' not in files and 'streams' in files:
        files['stream_order_shp'] = files['streams']

    print("\n🗺️ Detected layers:")
    for k, v in files.items():
        print(f"  {k:<25s} → {v}")
    return files


DETECTED = detect(DATA_DIR)

# ─── write config for sections ───────────────────────────────────────────────
import json
cfg = {
    'DATA_PATHS'  : DETECTED,
    'OUT_DIR'     : OUT_DIR,
    'MAPS_DIR'    : MAPS_DIR,
    'PLOTS_DIR'   : PLOTS_DIR,
    'TABLES_DIR'  : TABLES_DIR,
    'SHAPES_DIR'  : SHAPES_DIR,
    'REPORT_DIR'  : REPORT_DIR,
    'HTML_DIR'    : HTML_DIR,
    'N_SUBBASINS' : 5,
}
cfg_path = os.path.join(OUT_DIR, '_ci_config.json')
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f"\n✅ Config written to {cfg_path}")

# ─── run sections ────────────────────────────────────────────────────────────
SECTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sections')
SECTION_ORDER = [
    ('S1_environment',   'S1_environment.py'),
    ('S2_preprocessing', 'S2_data_loading.py'),
    ('S3_morphometry',   'S3_morphometric_params.py'),
    ('S4_maps',          'S4_maps.py'),
    ('S5_statistics',    'S5_statistics.py'),
    ('S6_prioritization','S6_prioritization.py'),
    ('S7_plotly',        'S7_plotly_visualizations.py'),
    ('S8_S9_export',     'S8_S9_export_report.py'),
    ('S10_tectonic',     'S10_tectonic_indices.py'),
    ('S11_steepness',    'S11_steepness_concavity.py'),
    ('S12_anomaly',      'S12_geomorphic_anomaly.py'),
    ('S13_flood',        'S13_flood_hazard.py'),
]

# Inject config into builtins so all sections can read it
import builtins
builtins._CI_CFG = cfg

success, failed = [], []

for folder, script in SECTION_ORDER:
    script_path = os.path.join(SECTIONS_DIR, folder, script)
    if not os.path.exists(script_path):
        print(f"\n⚠️  {script} not found — skipping")
        continue
    print(f"\n{'='*55}")
    print(f"  ▶  Running {folder}/{script}")
    print('='*55)
    try:
        with open(script_path, 'r', encoding='utf-8') as fh:
            source = fh.read()
        exec(compile(source, script_path, 'exec'), {'__file__': script_path})
        success.append(folder)
        print(f"  ✅ {folder} complete")
    except Exception as e:
        failed.append(folder)
        print(f"  ❌ {folder} FAILED: {e}")
        traceback.print_exc()
        # Continue to next section — don't abort the whole pipeline

print(f"\n{'='*55}")
print(f"  PIPELINE COMPLETE")
print(f"  ✅ Passed : {len(success)} sections")
print(f"  ❌ Failed : {len(failed)} sections")
if failed:
    print(f"  Failed: {', '.join(failed)}")
print('='*55)

sys.exit(0 if not failed else 1)
