"""K-Means definitivo K=2 y caracterización exploratoria de viviendas."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE, N_INIT, K = 42, 20, 2

def assign_group(price: float) -> str:
    if price <= 100000: return "grupo1"
    if price <= 500000: return "grupo2"
    return "grupo3"

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    variable_table = pd.read_csv(root / "outputs" / "tables" / "paso19_variables_clustering.csv")
    variables = variable_table["variable"].tolist()
    numeric = data.select_dtypes(include="number").columns.tolist()
    if len(variables) != 36 or variables != [c for c in numeric if c not in {"Id", "SalePrice"}]: raise RuntimeError("No se conservaron exactamente las variables del Paso 19")
    if {"Id", "SalePrice"} & set(variables): raise RuntimeError("Id/SalePrice no pueden entrar al clustering")
    imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
    prepared = scaler.fit_transform(imputer.fit_transform(data[variables]))
    if prepared.shape != (1460, 36) or not np.isfinite(prepared).all(): raise RuntimeError("Matriz preparada inválida")
    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(prepared); sizes = np.bincount(labels, minlength=K)
    silhouette = float(silhouette_score(prepared, labels))
    if len(labels) != 1460 or len(np.unique(labels)) != 2 or not np.isfinite(silhouette).all(): raise RuntimeError("Clusters inválidos")
    if sizes.tolist() != [769, 691]: raise RuntimeError(f"Tamaños inesperados: {sizes.tolist()}")
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    for directory in (tables_dir, metrics_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"indice_original": np.arange(len(data)), "cluster_kmeans": labels, "SalePrice": data["SalePrice"]}).to_csv(tables_dir / "paso20_asignaciones_kmeans.csv", index=False)
    def profile(aggregation: str) -> pd.DataFrame:
        values = data[variables].groupby(labels).agg(aggregation).T.reset_index().rename(columns={"index":"variable", 0:"cluster_0", 1:"cluster_1"})
        values["diferencia_absoluta"] = (values.cluster_0 - values.cluster_1).abs()
        values["diferencia_relativa"] = values.diferencia_absoluta / values.cluster_0.abs().replace(0, np.nan)
        return values.fillna(0)
    profile("mean").to_csv(tables_dir / "paso20_perfil_clusters_media.csv", index=False); profile("median").to_csv(tables_dir / "paso20_perfil_clusters_mediana.csv", index=False)
    discriminants = pd.DataFrame({"variable": variables, "centroid_cluster_0_std": model.cluster_centers_[0], "centroid_cluster_1_std": model.cluster_centers_[1]})
    discriminants["diferencia_std_abs"] = (discriminants.centroid_cluster_0_std - discriminants.centroid_cluster_1_std).abs(); discriminants.sort_values("diferencia_std_abs", ascending=False).head(15).to_csv(tables_dir / "paso20_variables_discriminantes.csv", index=False)
    original_centers = scaler.inverse_transform(model.cluster_centers_)
    pd.DataFrame(original_centers, columns=variables).assign(cluster=np.arange(K)).loc[:, ["cluster", *variables]].to_csv(tables_dir / "paso20_centroides_originales.csv", index=False)
    sale = data.assign(cluster_kmeans=labels).groupby("cluster_kmeans")["SalePrice"].agg(count="count", mean="mean", median="median", min="min", max="max", Q1=lambda x: x.quantile(.25), Q3=lambda x: x.quantile(.75)).reset_index()
    sale.to_csv(tables_dir / "paso20_saleprice_por_cluster.csv", index=False)
    data["PriceGroup"] = data["SalePrice"].map(assign_group)
    cross = pd.crosstab(pd.Series(labels, name="cluster_kmeans"), data["PriceGroup"], normalize="index").reindex(columns=["grupo1", "grupo2", "grupo3"], fill_value=0)
    frequencies = pd.crosstab(pd.Series(labels, name="cluster_kmeans"), data["PriceGroup"]).reindex(columns=["grupo1", "grupo2", "grupo3"], fill_value=0)
    long = frequencies.reset_index().melt(id_vars="cluster_kmeans", var_name="PriceGroup", value_name="frecuencia"); long["porcentaje_cluster"] = long.apply(lambda row: cross.loc[row.cluster_kmeans, row.PriceGroup] * 100, axis=1)
    long.to_csv(tables_dir / "paso20_cluster_vs_pricegroup.csv", index=False)
    plt.figure(figsize=(7, 5)); plt.bar(["cluster 0", "cluster 1"], sizes); plt.ylabel("Número de viviendas"); plt.title("Tamaño de clusters K-Means"); plt.tight_layout(); plt.savefig(figures_dir / "paso20_tamano_clusters.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5)); plt.boxplot([data.loc[labels == cluster, "SalePrice"] for cluster in range(K)], tick_labels=["cluster 0", "cluster 1"]); plt.ylabel("SalePrice"); plt.title("SalePrice por cluster"); plt.tight_layout(); plt.savefig(figures_dir / "paso20_saleprice_por_cluster.png", dpi=150); plt.close()
    top = discriminants.sort_values("diferencia_std_abs", ascending=False).head(10).sort_values("diferencia_std_abs"); plt.figure(figsize=(9, 6)); plt.barh(top.variable, top.diferencia_std_abs); plt.xlabel("Diferencia absoluta estandarizada"); plt.ylabel("Variable"); plt.title("Variables discriminantes entre clusters"); plt.tight_layout(); plt.savefig(figures_dir / "paso20_variables_discriminantes.png", dpi=150); plt.close()
    pca = PCA(n_components=2, random_state=RANDOM_STATE); points = pca.fit_transform(prepared); plt.figure(figsize=(8, 6)); plt.scatter(points[:, 0], points[:, 1], c=labels); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-Means proyectado en PCA"); plt.tight_layout(); plt.savefig(figures_dir / "paso20_kmeans_pca.png", dpi=150); plt.close()
    explained = pca.explained_variance_ratio_; pd.DataFrame([{ "explained_variance_pc1": explained[0], "explained_variance_pc2": explained[1], "explained_variance_total_2pc": explained.sum() }]).to_csv(metrics_dir / "paso20_pca_resumen.csv", index=False)
    pd.DataFrame([{ "n_clusters":K, "silhouette":silhouette, "cluster_0_size":sizes[0], "cluster_1_size":sizes[1], "cluster_0_pct":sizes[0]/len(labels), "cluster_1_pct":sizes[1]/len(labels), "saleprice_mean_cluster_0":sale.loc[sale.cluster_kmeans == 0, "mean"].iloc[0], "saleprice_mean_cluster_1":sale.loc[sale.cluster_kmeans == 1, "mean"].iloc[0], "saleprice_median_cluster_0":sale.loc[sale.cluster_kmeans == 0, "median"].iloc[0], "saleprice_median_cluster_1":sale.loc[sale.cluster_kmeans == 1, "median"].iloc[0], "random_state":RANDOM_STATE, "n_init":N_INIT }]).to_csv(metrics_dir / "paso20_resumen_kmeans.csv", index=False)
    print(f"K={K}; silhouette={silhouette:.6f}; tamaños={sizes.tolist()}; PCA 2PC={explained.sum():.6f}"); print(sale.to_string(index=False)); print(discriminants.sort_values("diferencia_std_abs", ascending=False).head(10).to_string(index=False)); print(long.to_string(index=False))

if __name__ == "__main__": main()
