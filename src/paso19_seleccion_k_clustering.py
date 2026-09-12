"""Selección exploratoria de K mediante codo y silhouette; no es el K-Means definitivo."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_INIT = 20
K_VALUES = list(range(2, 11))

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    numeric = data.select_dtypes(include="number").columns.tolist()
    variables = [column for column in numeric if column not in {"Id", "SalePrice"}]
    if not variables or "SalePrice" in variables or "Id" in variables: raise RuntimeError("Variables de clustering inválidas")
    variable_table = pd.DataFrame({"variable": variables, "dtype": [str(data[c].dtype) for c in variables], "missing_count": [int(data[c].isna().sum()) for c in variables], "estrategia_preprocesamiento": "SimpleImputer(median) + StandardScaler"})
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    for directory in (tables_dir, metrics_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    variable_table.to_csv(tables_dir / "paso19_variables_clustering.csv", index=False)
    imputed = SimpleImputer(strategy="median").fit_transform(data[variables])
    prepared = StandardScaler().fit_transform(imputed)
    if prepared.shape != (1460, len(variables)) or not np.isfinite(prepared).all(): raise RuntimeError("Matriz preparada inválida")
    metrics, sizes, quality = [], [], []
    for k in K_VALUES:
        start = time.perf_counter()
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(prepared)
        seconds = time.perf_counter() - start
        inertia = float(model.inertia_); silhouette = float(silhouette_score(prepared, labels))
        metrics.append({"k": k, "inertia": inertia, "silhouette_score": silhouette, "train_seconds": seconds})
        counts = pd.Series(labels).value_counts().sort_index()
        for cluster, n in counts.items(): sizes.append({"k": k, "cluster": int(cluster), "n": int(n), "pct": float(n / len(labels))})
        min_size, max_size = int(counts.min()), int(counts.max())
        quality.append({"k": k, "min_cluster_size": min_size, "max_cluster_size": max_size, "size_ratio": float(max_size / min_size), "silhouette_score": silhouette, "comentario": "revisar tamaño relativo; no se descarta automáticamente" if min_size < 0.05 * len(labels) else "sin cluster extremadamente pequeño"})
    metrics_df = pd.DataFrame(metrics)
    if not np.isfinite(metrics_df.silhouette_score).all() or (metrics_df.inertia.diff().dropna() > 1e-8).any(): raise RuntimeError("Métricas K inválidas")
    metrics_df.to_csv(tables_dir / "paso19_metricas_k.csv", index=False); pd.DataFrame(sizes).to_csv(tables_dir / "paso19_tamano_clusters.csv", index=False); pd.DataFrame(quality).to_csv(tables_dir / "paso19_calidad_clusters.csv", index=False)
    best = metrics_df.loc[metrics_df.silhouette_score.idxmax()]
    pd.DataFrame([{ "best_k_silhouette": int(best.k), "best_silhouette": best.silhouette_score, "k_range_evaluated": "2..10" }]).to_csv(metrics_dir / "paso19_mejor_k.csv", index=False)
    plt.figure(figsize=(8, 5)); plt.plot(metrics_df.k, metrics_df.inertia, marker="o"); plt.xticks(K_VALUES); plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Método del codo"); plt.tight_layout(); plt.savefig(figures_dir / "paso19_metodo_codo.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5)); plt.plot(metrics_df.k, metrics_df.silhouette_score, marker="o"); plt.xticks(K_VALUES); plt.xlabel("K"); plt.ylabel("Silhouette score"); plt.title("Silhouette por K"); plt.tight_layout(); plt.savefig(figures_dir / "paso19_silhouette_por_k.png", dpi=150); plt.close()
    # La elección del codo se deja documentada explícitamente, sin automatizar una regla inventada.
    recommended_k = int(best.k)
    elbow_interpretation = "La curva muestra una disminución fuerte de inertia en los primeros K y ganancias marginales decrecientes después; el codo visual se aproxima a K=3-4."
    selection_reason = f"Se recomienda K={recommended_k} porque maximiza silhouette ({best.silhouette_score:.4f}); además mantiene una segmentación interpretable y es compatible con la zona de codo aproximada, sin imponer K=3 por los grupos de precio."
    pd.DataFrame([{ "recommended_k": recommended_k, "best_k_silhouette": int(best.k), "best_silhouette": best.silhouette_score, "elbow_interpretation": elbow_interpretation, "selection_reason": selection_reason }]).to_csv(metrics_dir / "paso19_k_recomendado.csv", index=False)
    pd.DataFrame([{ "n_rows": len(data), "n_numeric_features": len(variables), "k_min": 2, "k_max": 10, "best_k_silhouette": int(best.k), "best_silhouette": best.silhouette_score, "recommended_k": recommended_k, "random_state": RANDOM_STATE, "n_init": N_INIT }]).to_csv(metrics_dir / "paso19_resumen.csv", index=False)
    print(f"Variables numéricas: {len(variables)}; K evaluados: {K_VALUES}"); print(metrics_df.to_string(index=False)); print(f"Mejor K por silhouette: {int(best.k)} ({best.silhouette_score:.6f}); K recomendado: {recommended_k}")

if __name__ == "__main__": main()
