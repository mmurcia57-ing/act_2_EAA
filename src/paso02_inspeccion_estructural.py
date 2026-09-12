"""Inspección estructural reproducible del dataset de vivienda.

Este paso solo describe los datos: no imputa, elimina columnas, codifica ni
entrena modelos.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "housing_train.csv"
    tables = root / "outputs" / "tables"
    metrics = root / "outputs" / "metrics"
    figures = root / "outputs" / "figures"
    for directory in (tables, metrics, figures):
        directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    numeric = data.select_dtypes(include="number").columns.tolist()
    categorical = data.select_dtypes(include=["str", "category"]).columns.tolist()

    print(f"Shape: {data.shape}")
    print(f"Columnas: {data.columns.tolist()}")
    print("Dtypes:")
    print(data.dtypes.to_string())
    print("Primeras 5 filas:")
    print(data.head(5).to_string())
    print("Ultimas 5 filas:")
    print(data.tail(5).to_string())
    print(f"Memoria aproximada (bytes): {data.memory_usage(deep=True).sum()}")
    print(f"Variables numericas ({len(numeric)}): {numeric}")
    print(f"Variables categoricas ({len(categorical)}): {categorical}")

    type_rows = []
    for column in data.columns:
        type_rows.append({
            "variable": column,
            "dtype": str(data[column].dtype),
            "tipo_academico": "numerica" if column in numeric else "categorica",
        })
    pd.DataFrame(type_rows).to_csv(tables / "paso02_tipos_variables.csv", index=False)

    pd.DataFrame([{
        "filas": len(data),
        "columnas": len(data.columns),
        "variables_numericas": len(numeric),
        "variables_categoricas": len(categorical),
        "target": "SalePrice",
    }]).to_csv(metrics / "paso02_resumen_estructura.csv", index=False)

    cardinality_rows = []
    for column in categorical:
        counts = data[column].value_counts(dropna=True)
        cardinality_rows.append({
            "variable": column,
            "categorias_distintas": int(data[column].nunique(dropna=True)),
            "categoria_mas_frecuente": counts.index[0] if len(counts) else "",
            "frecuencia_mas_frecuente": int(counts.iloc[0]) if len(counts) else 0,
            "missing_count": int(data[column].isna().sum()),
        })
    pd.DataFrame(cardinality_rows).sort_values(
        "categorias_distintas", ascending=False
    ).to_csv(tables / "paso02_cardinalidad_categoricas.csv", index=False)

    missing = pd.DataFrame({
        "variable": data.columns,
        "missing_count": data.isna().sum().values,
        "missing_pct": (data.isna().mean().values * 100),
        "dtype": [str(dtype) for dtype in data.dtypes],
    })
    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    missing.to_csv(tables / "paso02_missing_resumen.csv", index=False)

    saleprice = data["SalePrice"]
    sale_summary = pd.DataFrame([{
        "count": saleprice.count(),
        "min": saleprice.min(),
        "max": saleprice.max(),
        "mean": saleprice.mean(),
        "median": saleprice.median(),
        "std": saleprice.std(),
        "Q1": saleprice.quantile(0.25),
        "Q3": saleprice.quantile(0.75),
        "mean_minus_median": saleprice.mean() - saleprice.median(),
    }])
    sale_summary.to_csv(metrics / "paso02_saleprice_resumen.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(saleprice, bins=30, edgecolor="black", alpha=0.75)
    plt.axvline(saleprice.mean(), color="red", linestyle="--", label="Media")
    plt.axvline(saleprice.median(), color="green", linestyle="--", label="Mediana")
    plt.title("Distribución inicial de SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "paso02_saleprice_histograma.png", dpi=150)
    plt.close()

    decisions = pd.DataFrame([
        {
            "variable": "Id",
            "rol": "identificador",
            "decision_preliminar": "excluir de modelado",
            "justificacion": "identificador único, no magnitud del inmueble",
        },
        {
            "variable": "SalePrice",
            "rol": "target",
            "decision_preliminar": "no usar como predictor",
            "justificacion": "variable objetivo",
        },
    ])
    decisions.to_csv(tables / "paso02_decisiones_preliminares.csv", index=False)

    print(f"Columnas con missing: {len(missing)}")
    print("Top 5 categóricas por cardinalidad:")
    print(pd.DataFrame(cardinality_rows).sort_values("categorias_distintas", ascending=False).head(5).to_string(index=False))
    print("Resumen SalePrice:")
    print(sale_summary.to_string(index=False))


if __name__ == "__main__":
    main()
