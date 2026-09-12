"""Crea y valida grupos de precio para el bloque de clasificación."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


GROUPS = ["grupo1", "grupo2", "grupo3"]


def asignar_grupo(precio: float) -> str:
    if precio <= 100000:
        return "grupo1"
    if precio <= 500000:
        return "grupo2"
    return "grupo3"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    df = data.copy()
    df["PriceGroup"] = df["SalePrice"].map(asignar_grupo)
    if df["PriceGroup"].isna().any() or not set(df["PriceGroup"].unique()).issubset(GROUPS):
        raise RuntimeError("Hay registros sin grupo o fuera de las tres clases válidas")

    metrics_dir = root / "outputs" / "metrics"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    for directory in (metrics_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    counts = df["PriceGroup"].value_counts().reindex(GROUPS, fill_value=0)
    validation = pd.DataFrame([{
        "total_registros": len(df), "grupo1_count": counts["grupo1"], "grupo2_count": counts["grupo2"], "grupo3_count": counts["grupo3"],
        "grupo1_pct": counts["grupo1"] / len(df) * 100, "grupo2_pct": counts["grupo2"] / len(df) * 100, "grupo3_pct": counts["grupo3"] / len(df) * 100,
        "grupos_missing": int(df["PriceGroup"].isna().sum()), "grupos_invalidos": int((~df["PriceGroup"].isin(GROUPS)).sum()),
    }])
    validation.to_csv(metrics_dir / "paso13_validacion_grupos.csv", index=False)

    distribution_rows = []
    for group in GROUPS:
        prices = df.loc[df["PriceGroup"] == group, "SalePrice"]
        distribution_rows.append({
            "grupo": group, "frecuencia": len(prices), "porcentaje": len(prices) / len(df) * 100,
            "saleprice_min": prices.min(), "saleprice_median": prices.median(), "saleprice_mean": prices.mean(), "saleprice_max": prices.max(),
        })
    pd.DataFrame(distribution_rows).to_csv(tables_dir / "paso13_distribucion_grupos.csv", index=False)

    majority = counts.idxmax()
    minority = counts.idxmin()
    pd.DataFrame([{
        "clase_mayoritaria": majority, "count_mayoritaria": counts[majority], "clase_minoritaria": minority,
        "count_minoritaria": counts[minority], "ratio_mayoritaria_minoritaria": counts[majority] / counts[minority],
    }]).to_csv(metrics_dir / "paso13_desbalance_clases.csv", index=False)

    comparison = pd.DataFrame([
        {"hallazgo": "grupo2 es mayoritario", "resultado_dataset": f"{counts['grupo2']} ({counts['grupo2'] / len(df) * 100:.2f}%)", "referencia_profesor": "la mayoría de viviendas caerá en grupo2", "coincide_conceptualmente": counts["grupo2"] == counts.max(), "comentario": "coincide"},
        {"hallazgo": "grupo3 es muy minoritario", "resultado_dataset": f"{counts['grupo3']} ({counts['grupo3'] / len(df) * 100:.2f}%)", "referencia_profesor": "grupo3 será extremadamente pequeño", "coincide_conceptualmente": counts["grupo3"] == counts.min(), "comentario": "coincide conceptualmente"},
        {"hallazgo": "clasificación fuertemente desbalanceada", "resultado_dataset": f"ratio {counts[majority] / counts[minority]:.2f}:1", "referencia_profesor": "se espera desbalance", "coincide_conceptualmente": counts[majority] / counts[minority] >= 5, "comentario": "la clase mayoritaria domina ampliamente"},
    ])
    comparison.to_csv(tables_dir / "paso13_comparacion_referencia_profesor.csv", index=False)

    X = df.drop(columns=["SalePrice", "PriceGroup", "Id"])
    y = df["PriceGroup"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    split_indices = pd.concat([
        pd.DataFrame({"indice_original": X_train.index, "conjunto": "train", "PriceGroup": y_train.to_numpy()}),
        pd.DataFrame({"indice_original": X_test.index, "conjunto": "test", "PriceGroup": y_test.to_numpy()}),
    ]).sort_values("indice_original")
    split_indices.to_csv(tables_dir / "paso13_split_indices.csv", index=False)

    train_test_rows = []
    for subset, labels in [("train", y_train), ("test", y_test)]:
        subset_counts = labels.value_counts().reindex(GROUPS, fill_value=0)
        for group in GROUPS:
            train_test_rows.append({"conjunto": subset, "grupo": group, "frecuencia": subset_counts[group], "porcentaje": subset_counts[group] / len(labels) * 100})
    pd.DataFrame(train_test_rows).to_csv(tables_dir / "paso13_distribucion_train_test.csv", index=False)

    pct_total = counts / len(df) * 100
    pct_train = y_train.value_counts().reindex(GROUPS, fill_value=0) / len(y_train) * 100
    pct_test = y_test.value_counts().reindex(GROUPS, fill_value=0) / len(y_test) * 100
    pd.DataFrame({
        "grupo": GROUPS, "pct_total": pct_total.values, "pct_train": pct_train.values, "pct_test": pct_test.values,
        "diff_train_total": (pct_train - pct_total).values, "diff_test_total": (pct_test - pct_total).values,
    }).to_csv(tables_dir / "paso13_validacion_estratificacion.csv", index=False)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(GROUPS, counts.values)
    plt.title("Distribución de grupos de precio")
    plt.xlabel("Grupo")
    plt.ylabel("Frecuencia")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso13_distribucion_grupos.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(GROUPS, (counts / len(df) * 100).values)
    plt.title("Porcentaje de registros por grupo de precio")
    plt.xlabel("Grupo")
    plt.ylabel("Porcentaje")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso13_porcentaje_grupos.png", dpi=150)
    plt.close()

    pd.DataFrame([{
        "total_registros": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "grupo1_total": counts["grupo1"], "grupo2_total": counts["grupo2"], "grupo3_total": counts["grupo3"],
        "grupo1_test": (y_test == "grupo1").sum(), "grupo2_test": (y_test == "grupo2").sum(), "grupo3_test": (y_test == "grupo3").sum(),
        "ratio_mayoritaria_minoritaria": counts[majority] / counts[minority], "stratified_split": True, "random_state": 42,
    }]).to_csv(metrics_dir / "paso13_resumen_clasificacion.csv", index=False)

    print(f"Grupos: grupo1={counts['grupo1']} ({counts['grupo1'] / len(df) * 100:.2f}%), grupo2={counts['grupo2']} ({counts['grupo2'] / len(df) * 100:.2f}%), grupo3={counts['grupo3']} ({counts['grupo3'] / len(df) * 100:.2f}%)")
    print(f"Mayoritaria/minoritaria: {majority}/{minority}; ratio={counts[majority] / counts[minority]:.2f}")
    print("Train:")
    print(y_train.value_counts().reindex(GROUPS, fill_value=0).to_string())
    print("Test:")
    print(y_test.value_counts().reindex(GROUPS, fill_value=0).to_string())
    print(f"SalePrice en X: {'SalePrice' in X.columns}; PriceGroup en X: {'PriceGroup' in X.columns}; Id en X: {'Id' in X.columns}")


if __name__ == "__main__":
    main()
