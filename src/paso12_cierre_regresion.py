"""Cierra el bloque de regresión usando únicamente métricas ya validadas."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_dir = root / "outputs" / "metrics"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    for directory in (metrics_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Estos valores se leen de la tabla maestra generada en el Paso 11.
    models = pd.read_csv(tables_dir / "paso11_comparacion_regresion_completa.csv")
    models = models[["modelo", "mae_test", "rmse_test", "r2_test", "rmse_train"]].copy()
    models["gap_rmse_train_test"] = models["rmse_test"] - models["rmse_train"]
    models = models.sort_values("rmse_test").reset_index(drop=True)
    models["ranking_rmse"] = models.index + 1
    models[["modelo", "mae_test", "rmse_test", "r2_test", "rmse_train", "gap_rmse_train_test", "ranking_rmse"]].to_csv(
        tables_dir / "paso12_comparacion_final_regresion.csv", index=False
    )

    baseline_rmse = float(models.loc[models["modelo"] == "arbol_sin_poda", "rmse_test"].iloc[0])
    improvement_rows = []
    for model in ["arbol_podado", "random_forest", "xgboost", "mlp"]:
        rmse = float(models.loc[models["modelo"] == model, "rmse_test"].iloc[0])
        improvement_rows.append({
            "comparacion": f"{model}_vs_arbol_sin_poda",
            "delta_rmse_abs": rmse - baseline_rmse,
            "delta_rmse_pct": (rmse - baseline_rmse) / baseline_rmse * 100,
        })
    for model_1, model_2 in [("xgboost", "random_forest"), ("xgboost", "mlp"), ("random_forest", "mlp")]:
        rmse_1 = float(models.loc[models["modelo"] == model_1, "rmse_test"].iloc[0])
        rmse_2 = float(models.loc[models["modelo"] == model_2, "rmse_test"].iloc[0])
        improvement_rows.append({
            "comparacion": f"{model_1}_vs_{model_2}",
            "delta_rmse_abs": rmse_1 - rmse_2,
            "delta_rmse_pct": (rmse_1 - rmse_2) / rmse_2 * 100,
        })
    pd.DataFrame(improvement_rows).to_csv(tables_dir / "paso12_mejoras_relativas.csv", index=False)

    professor_rows = []
    references = {"arbol_sin_poda": 41002.2, "arbol_podado": 42927.25, "random_forest": 27588.9}
    for model, reference in references.items():
        rmse = float(models.loc[models["modelo"] == model, "rmse_test"].iloc[0])
        professor_rows.append({
            "modelo": model, "rmse_dataset": rmse, "rmse_profesor": reference,
            "diferencia_absoluta": abs(rmse - reference), "diferencia_pct": abs(rmse - reference) / reference * 100,
            "comportamiento_dataset": "poda mejora" if model == "arbol_podado" else "resultado observado",
            "comportamiento_profesor": "poda empeora" if model == "arbol_podado" else "resultado de referencia",
            "comentario": "diferencia documentada por split, poda, preprocesamiento, implementación, codificación o hiperparámetros",
        })
    pd.DataFrame(professor_rows).to_csv(tables_dir / "paso12_comparacion_profesor_regresion.csv", index=False)

    interpretations = {
        "arbol_sin_poda": "sobreajuste muy fuerte",
        "arbol_podado": "menor sobreajuste y mejor generalización que baseline",
        "random_forest": "gap existente pero buen desempeño en test",
        "xgboost": "mejor RMSE test; controlar sobreajuste, pero buena generalización relativa",
        "mlp": "buen desempeño, aunque inferior a RF/XGB en este dataset",
    }
    generalization = models[["modelo", "rmse_train", "rmse_test", "gap_rmse_train_test"]].copy()
    generalization["ratio_test_train"] = generalization.apply(
        lambda row: None if row["rmse_train"] == 0 else row["rmse_test"] / row["rmse_train"], axis=1
    )
    generalization["interpretacion"] = generalization["modelo"].map(interpretations)
    generalization.rename(columns={"gap_rmse_train_test": "gap_abs"}).to_csv(tables_dir / "paso12_train_test_generalizacion.csv", index=False)

    pros_cons = [
        {"modelo": "arbol_sin_poda", "ventajas": "interpretable; sencillo; poca preparación conceptual", "desventajas": "alta varianza; puede sobreajustar", "cuando_usarlo": "baseline interpretable y rápido", "resultado_en_este_dataset": "menor desempeño y sobreajuste muy fuerte"},
        {"modelo": "arbol_podado", "ventajas": "menor complejidad; mayor interpretabilidad", "desventajas": "poda excesiva puede generar underfitting", "cuando_usarlo": "cuando se prioriza interpretación con complejidad controlada", "resultado_en_este_dataset": "mejoró frente al árbol sin poda"},
        {"modelo": "random_forest", "ventajas": "robusto; reduce varianza; buen desempeño tabular", "desventajas": "menor interpretabilidad; más costo que un árbol", "cuando_usarlo": "baseline robusto para datos tabulares", "resultado_en_este_dataset": "segundo mejor RMSE"},
        {"modelo": "xgboost", "ventajas": "alto desempeño; relaciones complejas; corrige errores secuencialmente", "desventajas": "más hiperparámetros; puede sobreajustar; menos interpretable", "cuando_usarlo": "cuando se busca desempeño y se puede validar cuidadosamente", "resultado_en_este_dataset": "mejor modelo"},
        {"modelo": "mlp", "ventajas": "capacidad no lineal; arquitectura flexible", "desventajas": "sensible a escalado; entrenamiento más complejo; menos interpretable; no siempre mejor en tabulares pequeños", "cuando_usarlo": "cuando hay relaciones no lineales y datos/preprocesamiento adecuados", "resultado_en_este_dataset": "tercer lugar, cercano a RF"},
    ]
    pd.DataFrame(pros_cons).to_csv(tables_dir / "paso12_ventajas_desventajas.csv", index=False)

    saleprice_range = 755000 - 34900
    relative = models[["modelo", "rmse_test"]].rename(columns={"rmse_test": "rmse"})
    relative["saleprice_range"] = saleprice_range
    relative["rmse_pct_range"] = relative["rmse"] / saleprice_range * 100
    relative.to_csv(tables_dir / "paso12_rmse_relativo.csv", index=False)

    best = models.iloc[0]
    second = models.iloc[1]
    xgb_rmse = float(models.loc[models["modelo"] == "xgboost", "rmse_test"].iloc[0])
    rf_rmse = float(models.loc[models["modelo"] == "random_forest", "rmse_test"].iloc[0])
    mlp_rmse = float(models.loc[models["modelo"] == "mlp", "rmse_test"].iloc[0])
    pd.DataFrame([{
        "best_model": best["modelo"], "best_rmse": best["rmse_test"], "best_mae": best["mae_test"], "best_r2": best["r2_test"],
        "second_best_model": second["modelo"], "second_best_rmse": second["rmse_test"],
        "xgboost_vs_rf_rmse_abs": xgb_rmse - rf_rmse, "xgboost_vs_rf_rmse_pct": (xgb_rmse - rf_rmse) / rf_rmse * 100,
        "xgboost_vs_mlp_rmse_abs": xgb_rmse - mlp_rmse, "xgboost_vs_mlp_rmse_pct": (xgb_rmse - mlp_rmse) / mlp_rmse * 100,
        "tree_pruning_effect": "mejoró en nuestro experimento",
        "professor_pruning_effect": "empeoró: 41002.2 a 42927.25 (+1925.05)",
    }]).to_csv(metrics_dir / "paso12_resumen_regresion.csv", index=False)

    plt.figure(figsize=(9, 5))
    ordered = models.sort_values("rmse_test", ascending=False)
    plt.barh(ordered["modelo"], ordered["rmse_test"])
    plt.xlabel("RMSE test")
    plt.ylabel("Modelo")
    plt.title("Ranking de RMSE de regresión")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso12_ranking_rmse_regresion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(models["modelo"], models["r2_test"])
    plt.ylabel("R² test")
    plt.title("Comparación de R² de regresión")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "paso12_comparacion_r2_regresion.png", dpi=150)
    plt.close()

    long = models.melt(id_vars="modelo", value_vars=["rmse_train", "rmse_test"], var_name="conjunto", value_name="rmse")
    plt.figure(figsize=(10, 5))
    for label, group in long.groupby("conjunto"):
        plt.plot(group["modelo"], group["rmse"], marker="o", label=label)
    plt.ylabel("RMSE")
    plt.title("RMSE train vs. test")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "paso12_train_vs_test_rmse.png", dpi=150)
    plt.close()

    print(f"Mejor modelo: {best['modelo']} con RMSE {best['rmse_test']}")
    print(models[["ranking_rmse", "modelo", "rmse_test", "mae_test", "r2_test"]].to_string(index=False))
    print(f"XGBoost vs RF delta RMSE: {xgb_rmse - rf_rmse} ({(xgb_rmse - rf_rmse) / rf_rmse * 100}%)")
    print(f"XGBoost vs MLP delta RMSE: {xgb_rmse - mlp_rmse} ({(xgb_rmse - mlp_rmse) / mlp_rmse * 100}%)")


if __name__ == "__main__":
    main()
