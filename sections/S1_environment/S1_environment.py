"""
=============================================================================
SECTION 1 — ENVIRONMENT SETUP & LIBRARY IMPORTS
=============================================================================
Run in Google Colab. Installs missing packages and imports all libraries.
=============================================================================
"""

import subprocess, sys

def pip_install(*pkgs):
    """Silent pip install with error catching."""
    for pkg in pkgs:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"  ✅ {pkg}")
        except Exception as e:
            print(f"  ⚠️  {pkg} — install failed ({e}), will try to continue")

print("📦 Installing packages...")
pip_install(
    "geopandas",
    "rasterio",
    "rasterstats",
    "shapely",
    "fiona",
    "pyproj",
    "richdem",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "seaborn",
    "plotly",
    "matplotlib",
    "mapclassify",
    "contextily",
    "joypy",
    "xarray",
    "rioxarray",
    "earthpy",
    "tqdm",
    "openpyxl",
)

print("\n📚 Importing libraries...")

# ── STANDARD ──────────────────────────────────────────────────────────────────
import os
import warnings
import traceback
import zipfile
import json
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── GEOSPATIAL ────────────────────────────────────────────────────────────────
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
import rasterio.plot
import fiona
from shapely.geometry import (Point, LineString, MultiLineString,
                               Polygon, MultiPolygon, box)
from shapely.ops import unary_union, linemerge
from pyproj import CRS, Transformer
from rasterstats import zonal_stats

# ── RICHDEM (optional, graceful fallback) ─────────────────────────────────────
try:
    import richdem as rd
    RICHDEM_OK = True
    print("  ✅ richdem available")
except ImportError:
    RICHDEM_OK = False
    print("  ⚠️  richdem not available — slope/aspect computed via numpy")

# ── NUMERICAL ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ── SKLEARN ───────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── STATSMODELS ───────────────────────────────────────────────────────────────
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── VISUALIZATION — MATPLOTLIB ────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# ── VISUALIZATION — PLOTLY ────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── OPTIONAL ──────────────────────────────────────────────────────────────────
try:
    import joypy
    JOYPY_OK = True
except ImportError:
    JOYPY_OK = False
    print("  ⚠️  joypy not available — ridge plots skipped")

try:
    import earthpy.spatial as es
    EARTHPY_OK = True
except ImportError:
    EARTHPY_OK = False

try:
    import xarray as xr
    import rioxarray
    RIOXARRAY_OK = True
except ImportError:
    RIOXARRAY_OK = False

# ── GLOBAL SETTINGS ───────────────────────────────────────────────────────────
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
})

# ── OUTPUT DIRECTORIES ────────────────────────────────────────────────────────
OUT_DIR      = "/content/morphometric_outputs/"
MAPS_DIR     = os.path.join(OUT_DIR, "maps/")
PLOTS_DIR    = os.path.join(OUT_DIR, "plots/")
TABLES_DIR   = os.path.join(OUT_DIR, "tables/")
SHAPES_DIR   = os.path.join(OUT_DIR, "shapefiles/")
REPORT_DIR   = os.path.join(OUT_DIR, "report/")

for d in [OUT_DIR, MAPS_DIR, PLOTS_DIR, TABLES_DIR, SHAPES_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

print("\n✅ All libraries imported successfully.")
print(f"📁 Output directory: {OUT_DIR}")

# ── VERSION REPORT ────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  geopandas  : {gpd.__version__}")
print(f"  rasterio   : {rasterio.__version__}")
print(f"  numpy      : {np.__version__}")
print(f"  pandas     : {pd.__version__}")
print(f"  plotly     : {__import__('plotly').__version__}")
print(f"{'='*50}")
