"""Clustering jerárquico aglomerativo Ward y comparación con K-Means."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

K, LINKAGE = 2, "ward"

def assign_group(price: float) -> str:
    if price <= 100000: return "grupo1"
    if price <= 500000: return "grupo2"
    return "grupo3"

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    variables = pd.read_csv(root / "outputs" / "tables" / "paso19_variables_clustering.csv")["variable"].tolist()
    if len(variables) != 36 or {"Id", "SalePrice", "PriceGroup"} & set(variables): raise RuntimeError("Variables inválidas")
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    prepared = scaler.fit_transform(imputer.fit_transform(data[variables]))
    if prepared.shape != (1460, 36) or not np.isfinite(prepared).all(): raise RuntimeError("Matriz preparada inválida")
    model = AgglomerativeClustering(n_clusters=K, linkage=LINKAGE)
    labels = model.fit_predict(prepared); sizes = np.bincount(labels, minlength=K); silhouette = float(silhouette_score(prepared, labels))
    if len(np.unique(labels)) != 2 or not np.isfinite(silhouette): raise RuntimeError("Clusters jerárquicos inválidos")
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    for directory in (tables_dir, metrics_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"indice_original": np.arange(len(data)), "cluster_hierarchical": labels, "SalePrice": data.SalePrice}).to_csv(tables_dir / "paso21_asignaciones_jerarquico.csv", index=False)
    def profile(aggregation: str) -> pd.DataFrame:
        out = data[variables].groupby(labels).agg(aggregation).T.reset_index().rename(columns={"index":"variable", 0:"cluster_0", 1:"cluster_1"})
        out["diferencia_absoluta"] = (out.cluster_0 - out.cluster_1).abs(); out["diferencia_relativa"] = out.diferencia_absoluta / out.cluster_0.abs().replace(0, np.nan)
        return out.fillna(0)
    profile("mean").to_csv(tables_dir / "paso21_perfil_media.csv", index=False); profile("median").to_csv(tables_dir / "paso21_perfil_mediana.csv", index=False)
    std_means = pd.DataFrame(prepared, columns=variables).groupby(labels).mean().T.reset_index().rename(columns={"index":"variable", 0:"cluster_0_mean_std", 1:"cluster_1_mean_std"})
    std_means["diferencia_std_abs"] = (std_means.cluster_0_mean_std - std_means.cluster_1_mean_std).abs(); std_means.sort_values("diferencia_std_abs", ascending=False).to_csv(tables_dir / "paso21_variables_discriminantes.csv", index=False)
    pd.DataFrame({"cluster_hierarchical": labels, "SalePrice": data.SalePrice}).groupby("cluster_hierarchical").SalePrice.agg(count="count", mean="mean", median="median", min="min", max="max", Q1=lambda x:x.quantile(.25), Q3=lambda x:x.quantile(.75)).reset_index().to_csv(tables_dir / "paso21_saleprice_por_cluster.csv", index=False)
    data["PriceGroup"] = data.SalePrice.map(assign_group)
    frequencies = pd.crosstab(pd.Series(labels, name="cluster_hierarchical"), data.PriceGroup).reindex(columns=["grupo1","grupo2","grupo3"], fill_value=0)
    pct = frequencies.div(frequencies.sum(axis=1), axis=0) * 100
    cross = frequencies.reset_index().melt(id_vars="cluster_hierarchical", var_name="PriceGroup", value_name="frecuencia"); cross["porcentaje_cluster"] = cross.apply(lambda r:pct.loc[r.cluster_hierarchical, r.PriceGroup], axis=1); cross.to_csv(tables_dir / "paso21_cluster_vs_pricegroup.csv", index=False)
    kmeans = pd.read_csv(tables_dir / "paso20_asignaciones_kmeans.csv")
    if len(kmeans) != len(labels) or not np.array_equal(kmeans.indice_original.to_numpy(), np.arange(len(data))): raise RuntimeError("Asignaciones K-Means incompatibles")
    contingency = pd.crosstab(kmeans.cluster_kmeans, labels).rename_axis("cluster_kmeans"); contingency.columns = [f"hierarchical_{c}" for c in contingency.columns]; contingency.to_csv(tables_dir / "paso21_comparacion_kmeans_jerarquico.csv")
    ari = float(adjusted_rand_score(kmeans.cluster_kmeans, labels)); pd.DataFrame([{ "adjusted_rand_index": ari }]).to_csv(metrics_dir / "paso21_agreement.csv", index=False)
    size_rows = [{"metodo":"KMeans", "cluster":int(c), "n":int(n), "pct":float(n/len(labels))} for c,n in enumerate(np.bincount(kmeans.cluster_kmeans, minlength=2))] + [{"metodo":"Hierarchical", "cluster":int(c), "n":int(n), "pct":float(n/len(labels))} for c,n in enumerate(sizes)]
    pd.DataFrame(size_rows).to_csv(tables_dir / "paso21_tamano_clusters.csv", index=False)
    km_sil = float(pd.read_csv(root / "outputs" / "metrics" / "paso19_mejor_k.csv").iloc[0].best_silhouette)
    pd.DataFrame([{ "metodo":"KMeans", "n_clusters":2, "silhouette":km_sil }, {"metodo":"Hierarchical", "n_clusters":2, "silhouette":silhouette}]).to_csv(tables_dir / "paso21_comparacion_silhouette.csv", index=False)
    tree = linkage(prepared, method="ward"); plt.figure(figsize=(14, 7)); dendrogram(tree, truncate_mode="lastp", p=30, show_leaf_counts=True, no_labels=True); plt.title("Dendrograma jerárquico Ward (truncado)"); plt.xlabel("Clusters/hojas agrupadas"); plt.ylabel("Distancia Ward"); plt.tight_layout(); plt.savefig(figures_dir / "paso21_dendrograma.png", dpi=150); plt.close()
    pca = PCA(n_components=2); points = pca.fit_transform(prepared); plt.figure(figsize=(8,6)); plt.scatter(points[:,0], points[:,1], c=labels); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Clustering jerárquico proyectado en PCA"); plt.tight_layout(); plt.savefig(figures_dir / "paso21_jerarquico_pca.png", dpi=150); plt.close()
    plt.figure(figsize=(8,5)); x=np.arange(2); width=.35; km_sizes=np.bincount(kmeans.cluster_kmeans, minlength=2); plt.bar(x-width/2, km_sizes, width, label="KMeans"); plt.bar(x+width/2, sizes, width, label="Hierarchical"); plt.xticks(x,["cluster 0","cluster 1"]); plt.ylabel("Número de viviendas"); plt.title("Tamaños: K-Means vs jerárquico"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso21_comparacion_tamanos.png", dpi=150); plt.close()
    pd.DataFrame([{ "n_clusters":K, "silhouette":silhouette, "cluster_0_size":sizes[0], "cluster_1_size":sizes[1], "linkage":LINKAGE }]).to_csv(metrics_dir / "paso21_resumen_jerarquico.csv", index=False)
    print(f"Hierarchical Ward; silhouette={silhouette:.6f}; tamaños={sizes.tolist()}; ARI={ari:.6f}"); print(std_means.sort_values("diferencia_std_abs", ascending=False).head(10).to_string(index=False)); print(cross.to_string(index=False))

if __name__ == "__main__": main()
