"""
=============================================================================
SECTION 5 — STATISTICAL ANALYSIS
=============================================================================
Descriptive stats, correlation matrix, VIF, PCA, clustering.
=============================================================================
"""

print("=" * 60)
print("SECTION 5 — STATISTICAL ANALYSIS")
print("=" * 60)

# ── Select numeric morphometric columns for analysis ─────────────────────────
STAT_COLS = [
    'Area_km2', 'Perimeter_km', 'Basin_Length_km',
    'Drainage_Density_Dd', 'Stream_Frequency_Fs', 'Texture_Ratio_T',
    'Form_Factor_Ff', 'Elongation_Ratio_Re', 'Circularity_Ratio_Rc',
    'Compactness_Cc', 'LengthOverlandFlow_Lg', 'ChannelOverlandFlow_C',
    'Basin_Relief_H_m', 'Relief_Ratio_Rh', 'Relative_Relief',
    'Ruggedness_Rn', 'Melton_MRN', 'Hypsometric_HI',
    'Slope_Mean_deg', 'TRI_Mean', 'Rbm',
]
# Keep only columns that actually exist in df_master
STAT_COLS = [c for c in STAT_COLS if c in df_master.columns]
df_stat   = df_master[STAT_COLS].copy().astype(float)
df_stat.dropna(axis=1, how='all', inplace=True)
STAT_COLS = df_stat.columns.tolist()

print(f"  Parameters for analysis: {len(STAT_COLS)}")
print(f"  Subbasins: {len(df_stat)}")

# ─────────────────────────────────────────────────────────────────────────────
#  A. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[A] Descriptive Statistics...")

desc_extra = df_stat.agg([
    'mean', 'median', 'std',
    lambda x: (x.std()/x.mean()*100) if x.mean() != 0 else np.nan,  # CV%
    lambda x: float(stats.skew(x.dropna())),
    lambda x: float(stats.kurtosis(x.dropna())),
])
desc_extra.index = ['Mean', 'Median', 'Std', 'CV%', 'Skewness', 'Kurtosis']
desc_full = pd.concat([df_stat.describe(), desc_extra])

csv_path = os.path.join(TABLES_DIR, "descriptive_statistics.csv")
desc_full.to_csv(csv_path)
print(f"  ✅ Saved: {csv_path}")
print(desc_full.to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  B. CORRELATION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[B] Correlation Matrices (Pearson + Spearman)...")

# Pearson
corr_pearson  = df_stat.corr(method='pearson')
corr_spearman = df_stat.corr(method='spearman')

# Heatmap — Pearson
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax_corr, corr_mat, title in [
    (axes[0], corr_pearson,  "Pearson Correlation"),
    (axes[1], corr_spearman, "Spearman Correlation"),
]:
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat, mask=mask, ax=ax_corr,
        cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'size': 7},
        linewidths=0.5, square=True, cbar_kws={'shrink': 0.7},
    )
    ax_corr.set_title(title, fontsize=13, fontweight='bold')
    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45,
                              ha='right', fontsize=7.5)
    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✅ Correlation heatmap saved")

corr_pearson.to_csv(os.path.join(TABLES_DIR, "correlation_pearson.csv"))
corr_spearman.to_csv(os.path.join(TABLES_DIR, "correlation_spearman.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  C. VARIANCE INFLATION FACTOR
# ─────────────────────────────────────────────────────────────────────────────

print("\n[C] VIF Analysis...")
# Require at least 2 samples per predictor — only feasible if n > n_params
if len(df_stat) > len(STAT_COLS):
    df_vif    = df_stat.dropna()
    X_vif     = sm.add_constant(df_vif)
    vif_data  = pd.DataFrame({
        'Feature': df_vif.columns,
        'VIF'    : [variance_inflation_factor(X_vif.values, i + 1)
                    for i in range(len(df_vif.columns))]
    }).sort_values('VIF', ascending=False)
    print(vif_data.to_string(index=False))
    vif_data.to_csv(os.path.join(TABLES_DIR, "vif.csv"), index=False)
else:
    print(f"  ⚠️  VIF skipped: n_basins ({len(df_stat)}) ≤ n_params ({len(STAT_COLS)})")
    vif_data = pd.DataFrame(columns=['Feature', 'VIF'])

# ─────────────────────────────────────────────────────────────────────────────
#  D. PCA
# ─────────────────────────────────────────────────────────────────────────────

print("\n[D] Principal Component Analysis...")

# Standardize
scaler    = StandardScaler()
df_scaled = df_stat.fillna(df_stat.median())
X_scaled  = scaler.fit_transform(df_scaled)

pca      = PCA()
scores   = pca.fit_transform(X_scaled)
n_comp   = len(pca.explained_variance_ratio_)

# Scree data
exp_var      = pca.explained_variance_ratio_ * 100
cum_var      = np.cumsum(exp_var)
n_comp_95    = np.searchsorted(cum_var, 95) + 1

print(f"  Total components: {n_comp}")
print(f"  Components to explain 95% variance: {n_comp_95}")
for i in range(min(n_comp, 5)):
    print(f"  PC{i+1}: {exp_var[i]:.2f}%  (cumulative: {cum_var[i]:.2f}%)")

# ── Scree plot ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, n_comp + 1), exp_var, color='steelblue', alpha=0.8, label='Individual')
ax1.plot(range(1, n_comp + 1), cum_var, 'ro-', ms=5, label='Cumulative')
ax1.axhline(95, color='green', linestyle='--', lw=1.2, label='95% threshold')
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)")
ax1.set_title("Scree Plot — PCA")
ax1.legend()
ax1.set_xlim(0.5, n_comp + 0.5)

# ── Biplot (PC1 vs PC2) ───────────────────────────────────────────────────────
pc1_scores = scores[:, 0]
pc2_scores = scores[:, 1] if n_comp > 1 else np.zeros(len(scores))

ax2.scatter(pc1_scores, pc2_scores, c='darkorange', s=120, zorder=5, edgecolors='black')
for i, bid in enumerate(df_stat.index):
    ax2.annotate(bid, (pc1_scores[i], pc2_scores[i]),
                 textcoords='offset points', xytext=(6, 3), fontsize=9)

# Loading vectors
loadings = pca.components_.T
scale    = max(abs(pc1_scores).max(), abs(pc2_scores).max())
for j, feat in enumerate(STAT_COLS):
    ax2.annotate(
        '', xy=(loadings[j, 0] * scale * 0.5, loadings[j, 1] * scale * 0.5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.2)
    )
    ax2.text(loadings[j, 0] * scale * 0.55, loadings[j, 1] * scale * 0.55,
             feat, fontsize=6.5, color='royalblue', ha='center')

ax2.set_xlabel(f"PC1 ({exp_var[0]:.1f}%)")
ax2.set_ylabel(f"PC2 ({exp_var[1]:.1f}%)" if n_comp > 1 else "PC2")
ax2.set_title("PCA Biplot (PC1 vs PC2)")
ax2.axhline(0, color='grey', lw=0.5, linestyle='--')
ax2.axvline(0, color='grey', lw=0.5, linestyle='--')

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "pca_scree_biplot.png"), dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✅ PCA scree + biplot saved")

# Save loadings
df_loadings = pd.DataFrame(
    pca.components_[:min(n_comp, 5)].T,
    index=STAT_COLS,
    columns=[f"PC{i+1}" for i in range(min(n_comp, 5))],
)
df_loadings.to_csv(os.path.join(TABLES_DIR, "pca_loadings.csv"))

df_scores_df = pd.DataFrame(
    scores[:, :min(n_comp, 5)],
    index=df_stat.index,
    columns=[f"PC{i+1}" for i in range(min(n_comp, 5))],
)
df_scores_df.to_csv(os.path.join(TABLES_DIR, "pca_scores.csv"))

# ─────────────────────────────────────────────────────────────────────────────
#  E. CLUSTER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[E] Cluster Analysis...")

if len(df_scaled) >= 3:
    # ── Hierarchical ─────────────────────────────────────────────────────────
    Z = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=df_stat.index.tolist(), ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)")
    ax.set_xlabel("Subbasin")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hierarchical_dendrogram.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # ── K-means ──────────────────────────────────────────────────────────────
    k_range = range(2, min(len(df_scaled), 4))
    sil_scores = []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbs = km.fit_predict(X_scaled)
        if len(set(lbs)) > 1:
            sil_scores.append(silhouette_score(X_scaled, lbs))
        else:
            sil_scores.append(-1)

    best_k = k_range.start + int(np.argmax(sil_scores))
    print(f"  Best k (silhouette): {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    CLUSTER_LABELS = km_final.fit_predict(X_scaled)
    df_master['Cluster'] = CLUSTER_LABELS

    # Visualise clusters in PC space
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pc1_scores, pc2_scores,
        c=CLUSTER_LABELS, cmap='Set1', s=180, edgecolors='black', zorder=5
    )
    for i, bid in enumerate(df_stat.index):
        ax.annotate(bid, (pc1_scores[i], pc2_scores[i]),
                    textcoords='offset points', xytext=(6, 3), fontsize=9)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel(f"PC1 ({exp_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({exp_var[1]:.1f}%)" if n_comp > 1 else "PC2")
    ax.set_title(f"K-means Clustering (k={best_k}) in PCA Space")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "kmeans_clusters.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Cluster analysis complete (k={best_k})")
else:
    print(f"  ⚠️  Clustering skipped: only {len(df_scaled)} basins (need ≥ 3)")
    CLUSTER_LABELS = np.zeros(len(df_stat), dtype=int)
    df_master['Cluster'] = CLUSTER_LABELS

print("\n✅ SECTION 5 complete.")
