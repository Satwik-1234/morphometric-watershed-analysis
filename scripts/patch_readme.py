"""
scripts/patch_readme.py
=======================
Reads morphometric_master_table.csv and flood_hazard_indices.csv
then patches the README.md between the marker comments:
  <!-- MORPHOMETRIC_TABLE_START -->
  <!-- MORPHOMETRIC_TABLE_END -->
Also patches image paths and run timestamp.
"""

import os, re, glob
from datetime import datetime, timezone

TABLES_DIR  = os.environ.get('TABLES_DIR', 'outputs/tables')
MAPS_DIR    = os.environ.get('MAPS_DIR',   'outputs/previews')
README_PATH = 'README.md'

# ── Load CSVs ─────────────────────────────────────────────────────────────────
try:
    import pandas as pd

    master_csv = os.path.join(TABLES_DIR, 'morphometric_master_table.csv')
    flood_csv  = os.path.join(TABLES_DIR, 'flood_hazard_indices.csv')
    iat_csv    = os.path.join(TABLES_DIR, 'tectonic_IAT.csv')

    if not os.path.exists(master_csv):
        print(f"⚠️  {master_csv} not found — skipping README patch")
        exit(0)

    df = pd.read_csv(master_csv, index_col=0)

    # Select key columns
    show_cols = {
        'Area_km2'             : 'Area (km²)',
        'Drainage_Density_Dd'  : 'Dd (km/km²)',
        'Elongation_Ratio_Re'  : 'Re',
        'Circularity_Ratio_Rc' : 'Rc',
        'Hypsometric_HI'       : 'HI',
        'Shape_Class'          : 'Shape Class',
        'Hyps_Class'           : 'Hypsometric Stage',
    }
    df_show = df[[c for c in show_cols if c in df.columns]].copy()
    df_show.columns = [show_cols[c] for c in df_show.columns]

    # Add IAT class if available
    if os.path.exists(iat_csv):
        df_iat = pd.read_csv(iat_csv, index_col=0)
        if 'IAT_class' in df_iat.columns:
            df_show = df_show.join(df_iat[['IAT_class']].rename(columns={'IAT_class': 'IAT Class'}))

    # Add flood priority if available
    if os.path.exists(flood_csv):
        df_fh = pd.read_csv(flood_csv, index_col=0)
        if 'FFPI_class' in df_fh.columns:
            df_show = df_show.join(df_fh[['FFPI_class']].rename(columns={'FFPI_class': 'Flood Priority'}))

    # Format floats
    for col in df_show.columns:
        try:
            df_show[col] = df_show[col].round(3)
        except Exception:
            pass

    # Build markdown table
    md_lines = []
    headers = ['Basin'] + df_show.columns.tolist()
    md_lines.append('| ' + ' | '.join(headers) + ' |')
    md_lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for idx, row in df_show.iterrows():
        vals = [str(idx)] + [str(v) if str(v) != 'nan' else '—' for v in row]
        md_lines.append('| ' + ' | '.join(vals) + ' |')

    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    table_block = (
        f"\n*Auto-generated: {now_utc}*\n\n"
        + "\n".join(md_lines)
        + "\n"
    )

    # ── Patch README ────────────────────────────────────────────────────
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme = f.read()

    pattern = r'(<!-- MORPHOMETRIC_TABLE_START -->)(.*?)(<!-- MORPHOMETRIC_TABLE_END -->)'
    replacement = r'\1' + table_block + r'\3'
    readme_new = re.sub(pattern, replacement, readme, flags=re.DOTALL)

    if readme_new == readme:
        print("⚠️  README markers not found — table not patched")
    else:
        with open(README_PATH, 'w', encoding='utf-8') as f:
            f.write(readme_new)
        print(f"✅ README patched with morphometric results table ({now_utc})")

    print(df_show.to_string())

except ImportError:
    print("⚠️  pandas not available — skipping README patch")
except Exception as e:
    print(f"⚠️  patch_readme.py failed: {e}")
