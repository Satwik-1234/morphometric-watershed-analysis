"""
=============================================================================
SECTION 6 — WATERSHED PRIORITIZATION FRAMEWORK
=============================================================================
Method 1: Compound Parameter Ranking
Method 2: Entropy Weight Method
Method 3: PCA-Based Priority
Kendall's tau comparison + bar charts.
=============================================================================
"""

print("=" * 60)
print("SECTION 6 — WATERSHED PRIORITIZATION")
print("=" * 60)

# ── Erosion-sensitive parameters ──────────────────────────────────────────────
# Direct relation with erosion (higher = more erosion prone → higher rank = worse)
DIRECT_PARAMS = {
    'Drainage_Density_Dd' : 'Dd',
    'Stream_Frequency_Fs' : 'Fs',
    'Rbm'                 : 'Rb',
    'Ruggedness_Rn'       : 'Rn',
    'Relief_Ratio_Rh'     : 'Rh',
    'Hypsometric_HI'      : 'HI',
    'Melton_MRN'          : 'MRN',
}
# Inverse relation (higher = less erosion prone → lower rank = worse)
INVERSE_PARAMS = {
    'Elongation_Ratio_Re' : 'Re',
    'Circularity_Ratio_Rc': 'Rc',
    'Form_Factor_Ff'      : 'Ff',
}

# Keep only params actually in df_master
DIRECT_AVAIL  = {k: v for k, v in DIRECT_PARAMS.items()  if k in df_master.columns}
INVERSE_AVAIL = {k: v for k, v in INVERSE_PARAMS.items() if k in df_master.columns}
ALL_PRIORITY_COLS = list(DIRECT_AVAIL.keys()) + list(INVERSE_AVAIL.keys())

df_pri = df_master[ALL_PRIORITY_COLS].copy().astype(float).fillna(df_master[ALL_PRIORITY_COLS].median())

# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 1 — COMPOUND PARAMETER RANKING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Method 1] Compound Parameter Ranking...")

df_rank = pd.DataFrame(index=df_pri.index)

for col in DIRECT_AVAIL:
    # rank: highest value → rank 1 (most erosion prone)
    df_rank[col] = df_pri[col].rank(ascending=False, method='min')

for col in INVERSE_AVAIL:
    # rank: lowest value → rank 1 (most erosion prone)
    df_rank[col] = df_pri[col].rank(ascending=True, method='min')

df_rank['CF_M1'] = df_rank.mean(axis=1)
df_rank['Rank_M1'] = df_rank['CF_M1'].rank(ascending=True, method='min').astype(int)

# Priority classes
n = len(df_rank)
thresholds = np.percentile(df_rank['CF_M1'], [33, 66])
df_rank['Priority_M1'] = df_rank['CF_M1'].apply(
    lambda x: 'High' if x <= thresholds[0] else ('Moderate' if x <= thresholds[1] else 'Low')
)

print(df_rank[['CF_M1', 'Rank_M1', 'Priority_M1']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 2 — ENTROPY WEIGHT METHOD
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Method 2] Entropy Weight Method...")

def entropy_weight_score(df, direct_cols, inverse_cols):
    """
    1. Normalise each parameter (0–1)
    2. Compute Shannon entropy for each parameter
    3. Derive weights from entropy divergence
    4. Compute weighted score per subbasin
    """
    df_norm = pd.DataFrame(index=df.index)
    for col in direct_cols:
        mn, mx = df[col].min(), df[col].max()
        df_norm[col] = (df[col] - mn) / (mx - mn + 1e-12)  # 0=best 1=worst
    for col in inverse_cols:
        mn, mx = df[col].min(), df[col].max()
        # Invert: low value = high risk → normalise inverted
        df_norm[col] = 1 - (df[col] - mn) / (mx - mn + 1e-12)

    # Entropy for each criterion
    n, m   = df_norm.shape
    weights = []
    for col in df_norm.columns:
        p = df_norm[col] / (df_norm[col].sum() + 1e-12)
        p = p.clip(lower=1e-12)  # avoid log(0)
        e = -np.sum(p * np.log(p)) / np.log(n + 1e-12)
        d = 1 - e
        weights.append(d)

    weights = np.array(weights)
    weights /= (weights.sum() + 1e-12)   # normalise to sum=1

    # Weighted score
    score = (df_norm.values * weights).sum(axis=1)
    return score, dict(zip(df_norm.columns, weights))


score_m2, ew_weights = entropy_weight_score(
    df_pri, list(DIRECT_AVAIL.keys()), list(INVERSE_AVAIL.keys())
)
df_rank['Score_M2'] = score_m2
df_rank['Rank_M2']  = pd.Series(score_m2, index=df_pri.index).rank(
    ascending=False, method='min'
).astype(int)

thresh_m2 = np.percentile(score_m2, [66, 33])
df_rank['Priority_M2'] = df_rank['Score_M2'].apply(
    lambda x: 'High' if x >= thresh_m2[0] else ('Moderate' if x >= thresh_m2[1] else 'Low')
)

print("  Entropy weights:")
for k, w in sorted(ew_weights.items(), key=lambda x: -x[1]):
    print(f"    {k}: {w:.4f}")
print(df_rank[['Score_M2', 'Rank_M2', 'Priority_M2']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 3 — PCA-BASED PRIORITY
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Method 3] PCA-Based Priority...")

# Re-run PCA on priority parameters only
scaler_p   = StandardScaler()
X_p        = scaler_p.fit_transform(df_pri.fillna(df_pri.median()))
pca_p      = PCA()
scores_p   = pca_p.fit_transform(X_p)
exp_var_p  = pca_p.explained_variance_ratio_

# Composite score: weighted sum of PC scores by explained variance
n_retain = min(3, len(exp_var_p))
weights_p = exp_var_p[:n_retain] / exp_var_p[:n_retain].sum()

# Sign convention: check if PC1 aligns with erosion risk
# (higher PC1 loading on Dd/Rn = higher risk = positive score)
pc1_loadings = pd.Series(pca_p.components_[0], index=ALL_PRIORITY_COLS)
direct_sign  = np.sign(pc1_loadings[list(DIRECT_AVAIL.keys())].mean())
if direct_sign < 0:
    scores_p = -scores_p   # flip sign

pca_composite = (scores_p[:, :n_retain] * weights_p).sum(axis=1)
df_rank['Score_M3'] = pca_composite
df_rank['Rank_M3']  = pd.Series(pca_composite, index=df_pri.index).rank(
    ascending=False, method='min'
).astype(int)

thresh_m3 = np.percentile(pca_composite, [66, 33])
df_rank['Priority_M3'] = df_rank['Score_M3'].apply(
    lambda x: 'High' if x >= thresh_m3[0] else ('Moderate' if x >= thresh_m3[1] else 'Low')
)
print(df_rank[['Score_M3', 'Rank_M3', 'Priority_M3']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON — KENDALL's TAU
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Comparison] Kendall's tau agreement analysis...")

r12, p12 = stats.kendalltau(df_rank['Rank_M1'], df_rank['Rank_M2'])
r13, p13 = stats.kendalltau(df_rank['Rank_M1'], df_rank['Rank_M3'])
r23, p23 = stats.kendalltau(df_rank['Rank_M2'], df_rank['Rank_M3'])

df_kendall = pd.DataFrame({
    'Comparison': ['M1 vs M2', 'M1 vs M3', 'M2 vs M3'],
    'Kendall_tau': [r12, r13, r23],
    'p_value'    : [p12, p13, p23],
    'Agreement'  : ['Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'
                    for r in [r12, r13, r23]],
})
print(df_kendall.to_string(index=False))

# ── Ranking comparison bar chart ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

basins  = df_rank.index.tolist()
x       = np.arange(len(basins))
width   = 0.28

ax1.bar(x - width, df_rank['Rank_M1'], width, label='Method 1 (Compound)',  color='steelblue')
ax1.bar(x,         df_rank['Rank_M2'], width, label='Method 2 (Entropy)',   color='darkorange')
ax1.bar(x + width, df_rank['Rank_M3'], width, label='Method 3 (PCA-based)', color='green')
ax1.set_xticks(x)
ax1.set_xticklabels(basins)
ax1.set_ylabel("Rank (1 = Highest Priority)")
ax1.set_title("Prioritization Rank Comparison Across Methods")
ax1.legend()
ax1.invert_yaxis()   # rank 1 at top

# Priority colour map
priority_map = {'High': '#d73027', 'Moderate': '#fee090', 'Low': '#4575b4'}
for i, bid in enumerate(basins):
    for j, (col, method) in enumerate([
        ('Priority_M1', 'M1'), ('Priority_M2', 'M2'), ('Priority_M3', 'M3')
    ]):
        ax2.bar(i * 4 + j, 1,
                color=priority_map.get(df_rank.loc[bid, col], 'grey'),
                edgecolor='black', linewidth=0.7)
        ax2.text(i * 4 + j, 0.5, df_rank.loc[bid, col][:1],
                 ha='center', va='center', fontsize=9, fontweight='bold')

ax2.set_xticks([i * 4 + 1 for i in range(len(basins))])
ax2.set_xticklabels(basins)
ax2.set_title("Priority Class by Method")
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in priority_map.items()]
ax2.legend(handles=legend_patches, loc='upper right')
ax2.set_yticks([])

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "prioritization_comparison.png"), dpi=180, bbox_inches='tight')
plt.close(fig)

# ── Save outputs ─────────────────────────────────────────────────────────────
ranking_table = df_rank[['CF_M1','Rank_M1','Priority_M1',
                          'Score_M2','Rank_M2','Priority_M2',
                          'Score_M3','Rank_M3','Priority_M3']].copy()
ranking_table.to_csv(os.path.join(TABLES_DIR, "prioritization_ranking.csv"))
df_kendall.to_csv(os.path.join(TABLES_DIR, "kendall_tau.csv"), index=False)

# Save priority shapefile
gdf_priority = gdf_sub.merge(
    ranking_table.reset_index(), on='basin_id', how='left'
)
gdf_priority.to_file(os.path.join(SHAPES_DIR, "subbasins_priority.shp"))

print(f"\n  ✅ Priority shapefile saved: {SHAPES_DIR}subbasins_priority.shp")
print("\n✅ SECTION 6 complete.")
print("\n  FINAL RANKING TABLE:")
print(ranking_table.to_string())
