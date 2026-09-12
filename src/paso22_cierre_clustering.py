"""Cierre comparativo de clustering usando únicamente artefactos existentes."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["KMeans", "Hierarchical_Ward"]
GROUPS = ["grupo1", "grupo2", "grupo3"]

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    km = pd.read_csv(tables_dir / "paso20_asignaciones_kmeans.csv")
    hier = pd.read_csv(tables_dir / "paso21_asignaciones_jerarquico.csv")
    if len(km) != 1460 or len(hier) != 1460: raise RuntimeError("Asignaciones incompletas")
    source = pd.read_csv(root / "data" / "housing_train.csv"); source["PriceGroup"] = source.SalePrice.map(lambda p: "grupo1" if p <= 100000 else ("grupo2" if p <= 500000 else "grupo3"))
    if source.PriceGroup.value_counts().reindex(GROUPS, fill_value=0).tolist() != [123, 1328, 9]: raise RuntimeError("Distribución PriceGroup inválida")
    assignments = {"KMeans": km.cluster_kmeans.to_numpy(), "Hierarchical_Ward": hier.cluster_hierarchical.to_numpy()}
    semantic = {"KMeans": {0: "perfil_bajo", 1: "perfil_alto"}, "Hierarchical_Ward": {1: "perfil_bajo", 0: "perfil_alto"}}
    mapping_rows = [{"metodo": method, "cluster_original": cluster, "perfil_semantico": semantic[method][cluster]} for method in METHODS for cluster in (0, 1)]
    pd.DataFrame(mapping_rows).to_csv(tables_dir / "paso22_mapeo_semantico_clusters.csv", index=False)
    profile_rows, price_rows, group2_rows, concentration_rows, method_rows = [], [], [], [], []
    for method in METHODS:
        labels = assignments[method]; semantic_labels = pd.Series(labels).map(semantic[method])
        price = source.SalePrice
        for profile_name in ("perfil_bajo", "perfil_alto"):
            mask = semantic_labels == profile_name; values = price[mask]
            stats = {"metodo": method, "perfil_semantico": profile_name, "n": int(mask.sum()), "saleprice_mean": float(values.mean()), "saleprice_median": float(values.median()), "saleprice_q1": float(values.quantile(.25)), "saleprice_q3": float(values.quantile(.75))}
            price_rows.append(stats)
            for group in GROUPS:
                n = int((mask & (source.PriceGroup == group)).sum()); group_rows = {"metodo": method, "perfil_semantico": profile_name, "price_group": group, "n": n, "pct_dentro_perfil": float(n / mask.sum() * 100)}; profile_rows.append(group_rows)
            if profile_name == "perfil_bajo": low_group = mask
            else: high_group = mask
        for group, expected_profile in (("grupo1", "perfil_bajo"), ("grupo3", "perfil_alto")):
            total = int((source.PriceGroup == group).sum()); n_expected = int(((source.PriceGroup == group) & (semantic_labels == expected_profile)).sum()); concentration_rows.append({"metodo": method, "price_group": group, "total_group": total, "n_en_perfil_esperado": n_expected, "pct_en_perfil_esperado": n_expected / total * 100})
        total_g2 = int((source.PriceGroup == "grupo2").sum())
        for profile_name in ("perfil_bajo", "perfil_alto"):
            n = int(((source.PriceGroup == "grupo2") & (semantic_labels == profile_name)).sum()); group2_rows.append({"metodo": method, "perfil_semantico": profile_name, "n": n, "pct_grupo2": n / total_g2 * 100})
        method_price = {r["perfil_semantico"]: r for r in price_rows if r["metodo"] == method}
        silhouette = float(pd.read_csv(tables_dir / "paso21_comparacion_silhouette.csv").query("metodo == 'Hierarchical'").silhouette.iloc[0]) if method == "Hierarchical_Ward" else float(pd.read_csv(tables_dir / "paso21_comparacion_silhouette.csv").query("metodo == 'KMeans'").silhouette.iloc[0])
        method_rows.append({"metodo": method, "n_clusters": 2, "silhouette": silhouette, "cluster_small_size": min(method_price["perfil_bajo"]["n"], method_price["perfil_alto"]["n"]), "cluster_large_size": max(method_price["perfil_bajo"]["n"], method_price["perfil_alto"]["n"]), "saleprice_low_profile_mean": method_price["perfil_bajo"]["saleprice_mean"], "saleprice_high_profile_mean": method_price["perfil_alto"]["saleprice_mean"], "observacion": "perfiles semánticos alineados; labels originales arbitrarios"})
    pd.DataFrame(method_rows).to_csv(tables_dir / "paso22_comparacion_metodos_clustering.csv", index=False); pd.DataFrame(profile_rows).to_csv(tables_dir / "paso22_pricegroup_vs_perfiles.csv", index=False); pd.DataFrame(concentration_rows).to_csv(tables_dir / "paso22_concentracion_extremos.csv", index=False); pd.DataFrame(group2_rows).to_csv(tables_dir / "paso22_distribucion_grupo2.csv", index=False); pd.DataFrame(price_rows).to_csv(tables_dir / "paso22_saleprice_perfiles.csv", index=False)
    price_frame = pd.DataFrame(price_rows)
    difference_rows = []
    for method in METHODS:
        subset = price_frame[price_frame.metodo == method].set_index("perfil_semantico")
        difference_rows.append({"metodo": method, "delta_mean_saleprice": subset.loc["perfil_alto", "saleprice_mean"] - subset.loc["perfil_bajo", "saleprice_mean"], "delta_median_saleprice": subset.loc["perfil_alto", "saleprice_median"] - subset.loc["perfil_bajo", "saleprice_median"]})
    pd.DataFrame(difference_rows).to_csv(metrics_dir / "paso22_diferencias_precio.csv", index=False)
    km_disc = pd.read_csv(tables_dir / "paso20_variables_discriminantes.csv"); hi_disc = pd.read_csv(tables_dir / "paso21_variables_discriminantes.csv")
    km_rank = {v: i + 1 for i, v in enumerate(km_disc.variable.head(10))}; hi_rank = {v: i + 1 for i, v in enumerate(hi_disc.variable.head(10))}; common = sorted(set(km_rank) & set(hi_rank), key=lambda v: (km_rank[v] + hi_rank[v], km_rank[v]))
    pd.DataFrame([{ "variable": v, "ranking_kmeans": km_rank[v], "ranking_jerarquico": hi_rank[v] } for v in common]).to_csv(tables_dir / "paso22_variables_comunes_clusters.csv", index=False)
    km_sil, hi_sil = method_rows[0]["silhouette"], method_rows[1]["silhouette"]; ari = float(pd.read_csv(metrics_dir / "paso21_agreement.csv").iloc[0].adjusted_rand_index); km_price = {r["perfil_semantico"]: r for r in price_rows if r["metodo"] == "KMeans"}; hi_price = {r["perfil_semantico"]: r for r in price_rows if r["metodo"] == "Hierarchical_Ward"}
    pd.DataFrame([{ "metodo": "KMeans", "silhouette": km_sil, "ARI": ari, "tamano_perfil_bajo": km_price["perfil_bajo"]["n"], "tamano_perfil_alto": km_price["perfil_alto"]["n"], "saleprice_medio_bajo": km_price["perfil_bajo"]["saleprice_mean"], "saleprice_medio_alto": km_price["perfil_alto"]["saleprice_mean"], "variables_discriminantes_comunes": ", ".join(common), "interpretacion_general": "mayor silhouette y perfiles más equilibrados"}, {"metodo": "Hierarchical_Ward", "silhouette": hi_sil, "ARI": ari, "tamano_perfil_bajo": hi_price["perfil_bajo"]["n"], "tamano_perfil_alto": hi_price["perfil_alto"]["n"], "saleprice_medio_bajo": hi_price["perfil_bajo"]["saleprice_mean"], "saleprice_medio_alto": hi_price["perfil_alto"]["saleprice_mean"], "variables_discriminantes_comunes": ", ".join(common), "interpretacion_general": "dendrograma aporta estructura jerárquica; perfiles más desiguales"}]).to_csv(tables_dir / "paso22_resumen_kmeans_vs_jerarquico.csv", index=False)
    for method in METHODS:
        subset = pd.DataFrame(profile_rows).query("metodo == @method"); pivot = subset.pivot(index="perfil_semantico", columns="price_group", values="pct_dentro_perfil").reindex(index=["perfil_bajo", "perfil_alto"], columns=GROUPS)
        x = np.arange(2); width = .25; plt.figure(figsize=(8,5))
        for i, group in enumerate(GROUPS): plt.bar(x + (i - 1) * width, pivot[group], width, label=group)
        plt.xticks(x, ["perfil bajo", "perfil alto"]); plt.ylabel("Porcentaje dentro del perfil"); plt.ylim(0, 100); plt.title(f"PriceGroup por perfil: {method}"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / f"paso22_pricegroup_por_perfil_{method}.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 5)); x = np.arange(4); width = .25
    combined = pd.DataFrame(profile_rows); combined["etiqueta"] = combined.metodo.str.replace("_Ward", "", regex=False) + "\n" + combined.perfil_semantico.str.replace("perfil_", "", regex=False)
    for i, group in enumerate(GROUPS):
        values = [combined.loc[(combined.metodo == method) & (combined.perfil_semantico == profile) & (combined.price_group == group), "pct_dentro_perfil"].iloc[0] for method, profile in [("KMeans", "perfil_bajo"), ("KMeans", "perfil_alto"), ("Hierarchical_Ward", "perfil_bajo"), ("Hierarchical_Ward", "perfil_alto")]]
        plt.bar(x + (i - 1) * width, values, width, label=group)
    plt.xticks(x, ["KMeans\nbajo", "KMeans\nalto", "Jerárquico\nbajo", "Jerárquico\nalto"]); plt.ylabel("Porcentaje dentro del perfil"); plt.ylim(0, 100); plt.title("PriceGroup por perfil semántico"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso22_pricegroup_por_perfil.png", dpi=150); plt.close()
    plt.figure(figsize=(8,5)); x=np.arange(2); width=.35; plt.bar(x-width/2, [km_price["perfil_bajo"]["saleprice_mean"], km_price["perfil_alto"]["saleprice_mean"]], width, label="KMeans"); plt.bar(x+width/2, [hi_price["perfil_bajo"]["saleprice_mean"], hi_price["perfil_alto"]["saleprice_mean"]], width, label="Hierarchical"); plt.xticks(x,["perfil bajo","perfil alto"]); plt.ylabel("SalePrice medio"); plt.title("SalePrice medio por perfil"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso22_saleprice_perfiles.png", dpi=150); plt.close()
    plt.figure(figsize=(7,5)); plt.bar(["KMeans","Hierarchical"],[km_sil,hi_sil]); plt.ylim(0,1); plt.ylabel("Silhouette"); plt.title("Silhouette por método"); plt.tight_layout(); plt.savefig(figures_dir / "paso22_silhouette_metodos.png", dpi=150); plt.close()
    common_df = pd.DataFrame([{ "variable": v, "ranking_kmeans": km_rank[v], "ranking_jerarquico": hi_rank[v] } for v in common]).sort_values("ranking_kmeans", ascending=False); plt.figure(figsize=(9,6)); plt.barh(common_df.variable, 1 / common_df.ranking_kmeans); plt.xlabel("Importancia relativa del ranking K-Means"); plt.title("Variables estructurales comunes"); plt.tight_layout(); plt.savefig(figures_dir / "paso22_variables_comunes.png", dpi=150); plt.close()
    pd.DataFrame([{ "kmeans_silhouette": km_sil, "hierarchical_silhouette": hi_sil, "adjusted_rand_index": ari, "preferred_method": "KMeans", "kmeans_low_profile_size": km_price["perfil_bajo"]["n"], "kmeans_high_profile_size": km_price["perfil_alto"]["n"], "hierarchical_low_profile_size": hi_price["perfil_bajo"]["n"], "hierarchical_high_profile_size": hi_price["perfil_alto"]["n"], "grupo1_pct_expected_profile_kmeans": concentration_rows[0]["pct_en_perfil_esperado"], "grupo1_pct_expected_profile_hierarchical": concentration_rows[2]["pct_en_perfil_esperado"], "grupo3_pct_expected_profile_kmeans": concentration_rows[1]["pct_en_perfil_esperado"], "grupo3_pct_expected_profile_hierarchical": concentration_rows[3]["pct_en_perfil_esperado"] }]).to_csv(metrics_dir / "paso22_resumen_clustering.csv", index=False)
    print(f"Cierre sin entrenamiento; silhouette KMeans={km_sil:.4f}; jerárquico={hi_sil:.4f}; ARI={ari:.4f}"); print(pd.DataFrame(concentration_rows).to_string(index=False)); print(pd.DataFrame(group2_rows).to_string(index=False))

if __name__ == "__main__": main()
