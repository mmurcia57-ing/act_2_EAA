"""Frecuencias y cardinalidad de variables categóricas, sin imputar ni codificar."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    categorical = data.select_dtypes(include=["str", "category"]).columns.tolist()
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    for directory in (tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    summaries = []
    long_frequencies = []
    for column in categorical:
        series = data[column]
        counts = series.value_counts(dropna=True)
        counts_all = series.value_counts(dropna=False)
        top = counts.iloc[0] if len(counts) else 0
        bottom = counts.iloc[-1] if len(counts) else 0
        top_category = counts.index[0] if len(counts) else "<NA>"
        bottom_category = counts.index[-1] if len(counts) else "<NA>"
        summaries.append({
            "variable": column,
            "n_categorias_sin_missing": int(series.nunique(dropna=True)),
            "n_categorias_con_missing": int(series.nunique(dropna=False)),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "categoria_mas_frecuente": top_category,
            "frecuencia_mas_frecuente": int(top),
            "porcentaje_mas_frecuente": float(top / len(data) * 100),
            "categoria_menos_frecuente": bottom_category,
            "frecuencia_menos_frecuente": int(bottom),
            "porcentaje_menos_frecuente": float(bottom / len(data) * 100),
        })
        for category, frequency in counts_all.items():
            is_missing = pd.isna(category)
            long_frequencies.append({
                "variable": column,
                "categoria": "<NA>" if is_missing else category,
                "frecuencia": int(frequency),
                "porcentaje": float(frequency / len(data) * 100),
                "es_missing": bool(is_missing),
            })

    summary = pd.DataFrame(summaries).sort_values("n_categorias_sin_missing", ascending=False)
    summary.to_csv(tables / "paso04_resumen_categoricas.csv", index=False)
    frequencies = pd.DataFrame(long_frequencies).sort_values(
        ["variable", "frecuencia"], ascending=[True, False]
    )
    frequencies.to_csv(tables / "paso04_frecuencias_categoricas.csv", index=False)

    high = summary[summary["n_categorias_sin_missing"] >= 10].copy()
    high["n_categorias"] = high["n_categorias_sin_missing"]
    high["comentario"] = "alta cardinalidad preliminar; OneHotEncoder puede aumentar la dimensionalidad"
    high[["variable", "n_categorias", "missing_count", "missing_pct", "comentario"]].to_csv(
        tables / "paso04_alta_cardinalidad.csv", index=False
    )

    dominant = summary[summary["porcentaje_mas_frecuente"] >= 90].copy()
    dominant[["variable", "categoria_mas_frecuente", "frecuencia_mas_frecuente", "porcentaje_mas_frecuente"]].rename(
        columns={
            "categoria_mas_frecuente": "categoria_dominante",
            "frecuencia_mas_frecuente": "frecuencia",
            "porcentaje_mas_frecuente": "porcentaje",
        }
    ).assign(
        decision_preliminar="revisar",
        justificacion="poca variabilidad aparente; no eliminar automáticamente",
    ).to_csv(tables / "paso04_categorias_dominantes.csv", index=False)

    rare = frequencies[(~frequencies["es_missing"]) & ((frequencies["frecuencia"] < 10) | (frequencies["porcentaje"] < 1))].copy()
    rare["comentario"] = "categoría poco frecuente; identificar sin agrupar todavía"
    rare[["variable", "categoria", "frecuencia", "porcentaje", "comentario"]].to_csv(
        tables / "paso04_categorias_raras.csv", index=False
    )

    missing = summary[summary["missing_count"] > 0][["variable", "missing_count", "missing_pct"]].copy()
    missing["interpretacion_preliminar"] = "missing identificado; revisar semántica antes de decidir tratamiento"
    missing["tipo_missing_preliminar"] = "por_revisar"
    missing.to_csv(tables / "paso04_missing_categoricas_preliminar.csv", index=False)

    key_names = ["MSZoning", "Street", "Neighborhood", "HouseStyle", "Exterior1st", "Exterior2nd", "SaleType"]
    key = summary[summary["variable"].isin(key_names)].copy()
    key[["variable", "n_categorias_sin_missing", "categoria_mas_frecuente", "frecuencia_mas_frecuente", "porcentaje_mas_frecuente", "missing_count"]].rename(
        columns={
            "n_categorias_sin_missing": "n_categorias",
            "categoria_mas_frecuente": "top_categoria",
            "frecuencia_mas_frecuente": "top_frecuencia",
            "porcentaje_mas_frecuente": "top_porcentaje",
        }
    ).to_csv(tables / "paso04_categoricas_clave.csv", index=False)

    neighborhood = data["Neighborhood"].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(neighborhood.index, neighborhood.values)
    plt.title("Frecuencias de Neighborhood")
    plt.xlabel("Categoría")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "paso04_neighborhood_frecuencias.png", dpi=150)
    plt.close()

    top_cardinality = summary.head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top_cardinality["variable"], top_cardinality["n_categorias_sin_missing"])
    plt.title("Top 10 variables categóricas por cardinalidad")
    plt.xlabel("Variable")
    plt.ylabel("Categorías sin missing")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "paso04_top_cardinalidad_categoricas.png", dpi=150)
    plt.close()

    comparison = pd.DataFrame([
        {
            "metrica": "Neighborhood categorías",
            "valor_dataset": int(data["Neighborhood"].nunique()),
            "valor_referencia_profesor": 25,
            "coincide": data["Neighborhood"].nunique() == 25,
            "comentario": "coincide con la referencia" if data["Neighborhood"].nunique() == 25 else "discrepancia documentada; no se modificaron los datos",
        },
        {
            "metrica": "Street categorías",
            "valor_dataset": int(data["Street"].nunique()),
            "valor_referencia_profesor": 2,
            "coincide": data["Street"].nunique() == 2,
            "comentario": "coincide con la referencia" if data["Street"].nunique() == 2 else "discrepancia documentada; no se modificaron los datos",
        },
    ])
    comparison.to_csv(tables / "paso04_comparacion_referencia_profesor.csv", index=False)

    print(f"Variables categóricas: {len(categorical)}")
    print("Top 10 por cardinalidad:")
    print(summary[["variable", "n_categorias_sin_missing"]].head(10).to_string(index=False))
    print(f"Categorías dominantes: {len(dominant)} variables")
    print(f"Variables con categorías raras: {rare['variable'].nunique()}")
    print(f"Variables categóricas con missing: {len(missing)}")
    print("Neighborhood top 5:")
    print(neighborhood.head(5).to_string())
    print("Street:")
    print(data["Street"].value_counts().to_string())


if __name__ == "__main__":
    main()
