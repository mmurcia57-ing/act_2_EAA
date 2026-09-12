"""Matriz de correlaciones Pearson y revisión preliminar de redundancia."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def correlation_interpretation(value: float) -> str:
    absolute = abs(value)
    if absolute >= 0.70:
        return "fuerte"
    if absolute >= 0.50:
        return "moderada-alta"
    if absolute >= 0.30:
        return "moderada"
    return "baja"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    numeric = data.select_dtypes(include="number").columns.tolist()
    predictor_numeric = [column for column in numeric if column != "Id"]
    correlation_data = data[predictor_numeric]
    corr = correlation_data.corr(method="pearson")

    tables = root / "outputs" / "tables"
    metrics = root / "outputs" / "metrics"
    figures = root / "outputs" / "figures"
    for directory in (tables, metrics, figures):
        directory.mkdir(parents=True, exist_ok=True)

    corr.to_csv(tables / "paso05_matriz_correlacion.csv")

    sale_corr = corr["SalePrice"].drop("SalePrice").dropna().to_frame("correlacion_saleprice")
    sale_corr["abs_correlacion"] = sale_corr["correlacion_saleprice"].abs()
    sale_corr["interpretacion"] = sale_corr["correlacion_saleprice"].map(correlation_interpretation)
    sale_corr = sale_corr.sort_values("abs_correlacion", ascending=False).rename_axis("variable").reset_index()
    sale_corr.to_csv(tables / "paso05_correlacion_saleprice.csv", index=False)
    sale_corr.head(10).to_csv(tables / "paso05_top_saleprice.csv", index=False)

    predictor_columns = [column for column in predictor_numeric if column != "SalePrice"]
    pair_rows = []
    for variable_1, variable_2 in combinations(predictor_columns, 2):
        value = corr.loc[variable_1, variable_2]
        if pd.notna(value) and abs(value) >= 0.70:
            pair_rows.append({
                "variable_1": variable_1,
                "variable_2": variable_2,
                "correlacion": value,
                "abs_correlacion": abs(value),
                "interpretacion": "fuerte",
                "decision_preliminar": "revisar_redundancia",
            })
    pairs = pd.DataFrame(pair_rows).sort_values("abs_correlacion", ascending=False)
    pairs.to_csv(tables / "paso05_pares_alta_correlacion.csv", index=False)

    reference_pairs = [
        ("GarageCars", "GarageArea"),
        ("TotalBsmtSF", "1stFlrSF"),
        ("YearBuilt", "GarageYrBlt"),
        ("GrLivArea", "TotRmsAbvGrd"),
    ]
    reference_rows = []
    for variable_1, variable_2 in reference_pairs:
        if variable_1 in corr.columns and variable_2 in corr.columns:
            value = corr.loc[variable_1, variable_2]
            reference_rows.append({
                "variable_1": variable_1,
                "variable_2": variable_2,
                "correlacion_dataset": value,
                "observacion": "par con relación alta o relevante para revisar",
                "coincide_con_referencia_conceptual": bool(pd.notna(value) and abs(value) >= 0.50),
            })
    pd.DataFrame(reference_rows).to_csv(tables / "paso05_pares_referencia_profesor.csv", index=False)

    redundancy = pairs[["variable_1", "variable_2", "correlacion"]].copy()
    redundancy["candidata_a_revisar"] = "sí"
    redundancy["justificacion"] = "posible información redundante; revisar según modelo y contexto"
    redundancy.to_csv(tables / "paso05_redundancia_preliminar.csv", index=False)

    plt.figure(figsize=(14, 12))
    image = plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, label="Correlación de Pearson")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Matriz de correlación de variables numéricas")
    plt.tight_layout()
    plt.savefig(figures / "paso05_heatmap_correlacion.png", dpi=150)
    plt.close()

    top_plot = sale_corr.head(12).sort_values("correlacion_saleprice")
    plt.figure(figsize=(9, 7))
    plt.barh(top_plot["variable"], top_plot["correlacion_saleprice"])
    plt.xlabel("Correlación con SalePrice")
    plt.ylabel("Variable")
    plt.title("Top 12 correlaciones absolutas con SalePrice")
    plt.tight_layout()
    plt.savefig(figures / "paso05_top_correlacion_saleprice.png", dpi=150)
    plt.close()

    strongest_pair = pairs.iloc[0] if len(pairs) else None
    summary = pd.DataFrame([{
        "n_variables_numericas": len(predictor_numeric),
        "n_pares_abs_r_ge_070": len(pairs),
        "max_correlacion_saleprice_variable": sale_corr.iloc[0]["variable"],
        "max_correlacion_saleprice_valor": sale_corr.iloc[0]["correlacion_saleprice"],
        "max_correlacion_predictores_variable_1": strongest_pair["variable_1"] if strongest_pair is not None else "",
        "max_correlacion_predictores_variable_2": strongest_pair["variable_2"] if strongest_pair is not None else "",
        "max_correlacion_predictores_valor": strongest_pair["correlacion"] if strongest_pair is not None else "",
    }])
    summary.to_csv(metrics / "paso05_correlaciones_resumen.csv", index=False)

    comparison_rows = [
        {
            "hallazgo": "OverallQual entre las más correlacionadas con SalePrice",
            "valor_dataset": sale_corr.loc[sale_corr["variable"] == "OverallQual", "correlacion_saleprice"].iloc[0],
            "referencia_profesor": "debe aparecer entre las variables más relacionadas",
            "coincide": "OverallQual" in sale_corr.head(10)["variable"].tolist(),
            "comentario": "comparación conceptual; no se inventaron valores de referencia",
        },
        {
            "hallazgo": "GrLivArea entre las más correlacionadas con SalePrice",
            "valor_dataset": sale_corr.loc[sale_corr["variable"] == "GrLivArea", "correlacion_saleprice"].iloc[0],
            "referencia_profesor": "debe aparecer entre las variables más relacionadas",
            "coincide": "GrLivArea" in sale_corr.head(10)["variable"].tolist(),
            "comentario": "comparación conceptual; no se inventaron valores de referencia",
        },
    ]
    for variable_1, variable_2 in reference_pairs:
        value = corr.loc[variable_1, variable_2]
        comparison_rows.append({
            "hallazgo": f"{variable_1}-{variable_2} como par relevante",
            "valor_dataset": value,
            "referencia_profesor": "par con correlación alta o relevante",
            "coincide": bool(pd.notna(value) and abs(value) >= 0.50),
            "comentario": "comparación conceptual; no se inventaron valores de referencia",
        })
    pd.DataFrame(comparison_rows).to_csv(tables / "paso05_comparacion_referencia_profesor.csv", index=False)

    print(f"Variables numéricas incluidas: {len(predictor_numeric)}")
    print("Top 10 correlaciones con SalePrice:")
    print(sale_corr.head(10)[["variable", "correlacion_saleprice"]].to_string(index=False))
    print(f"Pares con |r| >= 0.70: {len(pairs)}")
    print("Top pares:")
    print(pairs.head(10)[["variable_1", "variable_2", "correlacion"]].to_string(index=False))
    print("Pares de referencia:")
    print(pd.DataFrame(reference_rows).to_string(index=False))


if __name__ == "__main__":
    main()
