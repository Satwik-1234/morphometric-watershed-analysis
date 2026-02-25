"""
MASTER RUNNER — morphometric-watershed-analysis
================================================
Run all sections in sequence. Works locally and in Google Colab.
Usage:
    python MASTER_RUNNER.py                    # local
    exec(open('MASTER_RUNNER.py').read())      # Colab cell
"""
import os, sys
SECTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sections')
ORDER = [
    ('S0_extract',      'S0_extract_and_discover.py'),
    ('S1_environment',  'S1_environment.py'),
    ('S2_preprocessing','S2_data_loading.py'),
    ('S3_morphometry',  'S3_morphometric_params.py'),
    ('S4_maps',         'S4_maps.py'),
    ('S5_statistics',   'S5_statistics.py'),
    ('S6_prioritization','S6_prioritization.py'),
    ('S7_plotly',       'S7_plotly_visualizations.py'),
    ('S8_S9_export',    'S8_S9_export_report.py'),
    ('S10_tectonic',    'S10_tectonic_indices.py'),
    ('S11_steepness',   'S11_steepness_concavity.py'),
    ('S12_anomaly',     'S12_geomorphic_anomaly.py'),
    ('S13_flood',       'S13_flood_hazard.py'),
]
for folder, script in ORDER:
    path = os.path.join(SECTIONS_DIR, folder, script)
    print(f"\n{'='*55}\n  ▶  {folder}\n{'='*55}")
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'))
