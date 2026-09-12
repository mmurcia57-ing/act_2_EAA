"""Estadística descriptiva de las variables numéricas del dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def interpret_skewness(value: float) -> str:
    if value > 1:
        return "sesgo positivo fuerte"
    if value > 0.5:
        return "sesgo positivo moderado"
    if value >= -0.5:
        return "aproximadamente simétrica"
    if value >= -1:
        return "sesgo negativo moderado"
    return "sesgo negativo fuerte"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    numeric = data.select_dtypes(include="number").columns.tolist()
    numeric_data = data[numeric]

    tables = root / "outputs" / "tables"
    metrics = root / "outputs" / "metrics"
    figures = root / "outputs" / "figures"
    for directory in (tables, metrics, figures):
        directory.mkdir(parents=True, exist_ok=True)

    descriptive = pd.DataFrame({
        "variable": numeric,
        "count": [data[c].count() for c in numeric],
        "missing_count": [data[c].isna().sum() for c in numeric],
        "missing_pct": [data[c].isna().mean() * 100 for c in numeric],
        "min": [data[c].min() for c in numeric],
        "q1": [data[c].quantile(0.25) for c in numeric],
        "median": [data[c].median() for c in numeric],
        "mean": [data[c].mean() for c in numeric],
        "q3": [data[c].quantile(0.75) for c in numeric],
        "max": [data[c].max() for c in numeric],
        "std": [data[c].std() for c in numeric],
        "iqr": [data[c].quantile(0.75) - data[c].quantile(0.25) for c in numeric],
        "skewness": [data[c].skew() for c in numeric],
        "zero_count": [(data[c] == 0).sum() for c in numeric],
        "zero_pct": [(data[c] == 0).mean() * 100 for c in numeric],
    })
    descriptive.to_csv(tables / "paso03_estadistica_numerica.csv", index=False)

    sale = descriptive[descriptive["variable"] == "SalePrice"].iloc[0]
    sale_metrics = sale[["count", "min", "q1", "median", "mean", "q3", "max", "std", "skewness"]]
    sale_metrics.to_frame().T.to_csv(metrics / "paso03_saleprice_metricas.csv", index=False)

    median_zero = descriptive[descriptive["median"] == 0].copy()
    median_zero["interpretacion_preliminar"] = median_zero.apply(
        lambda row: (
            "alta concentración en cero; posible característica ausente en muchas viviendas; "
            "candidato a transformación binaria, no eliminar automáticamente"
        ), axis=1
    )
    median_zero[["variable", "median", "mean", "max", "zero_count", "zero_pct", "interpretacion_preliminar"]].to_csv(
        tables / "paso03_variables_mediana_cero.csv", index=False
    )

    low_variability = []
    for column in numeric:
        counts = data[column].value_counts(dropna=False)
        dominant_value = counts.index[0]
        dominant_frequency = int(counts.iloc[0])
        dominant_pct = dominant_frequency / len(data) * 100
        zero_pct = float(descriptive.loc[descriptive["variable"] == column, "zero_pct"].iloc[0])
        if zero_pct >= 95 or dominant_pct > 95:
            low_variability.append({
                "variable": column,
                "valor_dominante": dominant_value,
                "frecuencia_dominante": dominant_frequency,
                "porcentaje_dominante": dominant_pct,
                "decision_preliminar": "revisar",
                "justificacion": "más del 95% de observaciones en un mismo valor; evaluar transformación o exclusión",
            })
    pd.DataFrame(low_variability).sort_values("porcentaje_dominante", ascending=False).to_csv(
        tables / "paso03_baja_variabilidad_preliminar.csv", index=False
    )

    skewed = descriptive.copy()
    skewed["interpretacion"] = skewed["skewness"].map(interpret_skewness)
    skewed.reindex(skewed["skewness"].abs().sort_values(ascending=False).index).head(15)[
        ["variable", "skewness", "mean", "median", "min", "max", "interpretacion"]
    ].to_csv(tables / "paso03_variables_sesgadas.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(data["SalePrice"], bins=30, edgecolor="black", alpha=0.75)
    plt.axvline(data["SalePrice"].mean(), linestyle="--", label="Media")
    plt.axvline(data["SalePrice"].median(), linestyle="--", label="Mediana")
    plt.title("Distribución de SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "paso03_saleprice_distribucion.png", dpi=150)
    plt.close()

    top_zero = descriptive.sort_values("zero_pct", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top_zero["variable"], top_zero["zero_pct"])
    plt.title("Top 10 variables numéricas por porcentaje de ceros")
    plt.xlabel("Variable")
    plt.ylabel("Porcentaje de ceros")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "paso03_variables_ceros_top.png", dpi=150)
    plt.close()

    references = {
        "SalePrice min": (sale["min"], 34900),
        "SalePrice median": (sale["median"], 163000),
        "SalePrice mean": (sale["mean"], 180921),
        "SalePrice max": (sale["max"], 755000),
        "LotFrontage missing": (int(data["LotFrontage"].isna().sum()), 259),
        "MasVnrArea missing": (int(data["MasVnrArea"].isna().sum()), 8),
        "GarageYrBlt missing": (int(data["GarageYrBlt"].isna().sum()), 81),
    }
    comparison = []
    for metric, (value, reference) in references.items():
        difference = value - reference
        tolerance = 1 if metric == "SalePrice mean" else 0.01
        comparison.append({
            "metrica": metric,
            "valor_dataset": value,
            "valor_referencia_profesor": reference,
            "diferencia": difference,
            "coincide_aproximadamente": abs(difference) <= tolerance,
            "comentario": "coincide con la referencia" if abs(difference) <= tolerance else "discrepancia documentada; no se modificaron los datos",
        })
    pd.DataFrame(comparison).to_csv(tables / "paso03_comparacion_referencia_profesor.csv", index=False)

    print(f"Filas: {len(data)}")
    print(f"Variables numericas analizadas: {len(numeric)}")
    print("SalePrice:")
    print(sale_metrics.to_string())
    print(f"Variables con mediana 0: {len(median_zero)}")
    print("Top 10 por porcentaje de ceros:")
    print(descriptive.sort_values("zero_pct", ascending=False)[["variable", "zero_pct"]].head(10).to_string(index=False))
    print("Top 10 por valor absoluto de skewness:")
    print(descriptive.reindex(descriptive["skewness"].abs().sort_values(ascending=False).index)[["variable", "skewness"]].head(10).to_string(index=False))
    print(f"Candidatas de baja variabilidad: {[row['variable'] for row in low_variability]}")


if __name__ == "__main__":
    main()
