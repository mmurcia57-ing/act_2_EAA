"""Baseline de árbol de decisión para clasificación de PriceGroup."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


LABELS = ["grupo1", "grupo2", "grupo3"]


def assign_group(price: float) -> str:
    if price <= 100000:
        return "grupo1"
    if price <= 500000:
        return "grupo2"
    return "grupo3"


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["str", "category"]).columns.tolist()
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical),
    ])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    data["PriceGroup"] = data["SalePrice"].map(assign_group)
    split_indices = pd.read_csv(root / "outputs" / "tables" / "paso13_split_indices.csv")
    train_indices = split_indices.loc[split_indices["conjunto"] == "train", "indice_original"].astype(int).tolist()
    test_indices = split_indices.loc[split_indices["conjunto"] == "test", "indice_original"].astype(int).tolist()
    y = data["PriceGroup"]
    if y.loc[test_indices].value_counts().reindex(LABELS, fill_value=0).tolist() != [25, 265, 2]:
        raise RuntimeError("El test reutilizado no contiene las clases esperadas 25/265/2")

    X = data.drop(columns=["SalePrice", "PriceGroup", "Id"])
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", DecisionTreeClassifier(random_state=42)),
    ])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    accuracy_train = accuracy_score(y_train, pred_train)
    balanced_train = balanced_accuracy_score(y_train, pred_train)
    macro_f1_train = f1_score(y_train, pred_train, labels=LABELS, average="macro", zero_division=0)
    accuracy_test = accuracy_score(y_test, pred_test)
    balanced_test = balanced_accuracy_score(y_test, pred_test)
    macro_precision = precision_score(y_test, pred_test, labels=LABELS, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, pred_test, labels=LABELS, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, pred_test, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, pred_test, average="weighted", zero_division=0)

    metrics_dir = root / "outputs" / "metrics"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    for directory in (metrics_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "modelo": "DecisionTreeClassifier", "n_train": len(X_train), "n_test": len(X_test), "accuracy": accuracy_test,
        "balanced_accuracy": balanced_test, "macro_precision": macro_precision, "macro_recall": macro_recall,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1, "train_seconds": train_seconds, "random_state": 42,
    }]).to_csv(metrics_dir / "paso14_arbol_clasificacion_metricas.csv", index=False)

    matrix = confusion_matrix(y_test, pred_test, labels=LABELS)
    pd.DataFrame(matrix, index=["grupo1", "grupo2", "grupo3"], columns=LABELS).rename_axis("real/predicho").to_csv(tables_dir / "paso14_matriz_confusion.csv")

    group_rows = []
    for i, group in enumerate(LABELS):
        support = int(matrix[i, :].sum())
        hits = int(matrix[i, i])
        predicted_total = int(matrix[:, i].sum())
        group_rows.append({
            "grupo": group, "soporte_real": support, "aciertos": hits, "errores": support - hits,
            "recall_grupo": hits / support if support else 0,
            "precision_grupo": hits / predicted_total if predicted_total else 0,
            "f1_grupo": (2 * (hits / support) * (hits / predicted_total) / ((hits / support) + (hits / predicted_total))) if support and predicted_total and (hits / support + hits / predicted_total) else 0,
        })
    pd.DataFrame(group_rows).to_csv(tables_dir / "paso14_metricas_por_grupo.csv", index=False)

    report = classification_report(y_test, pred_test, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.rename_axis("grupo").to_csv(tables_dir / "paso14_classification_report.csv")

    professor_matrix = {"grupo1": (25, 11, 0.44), "grupo2": (265, 255, 255 / 265), "grupo3": (2, 0, 0.0)}
    comparison_rows = []
    for row in group_rows:
        support, professor_hits, professor_recall = professor_matrix[row["grupo"]]
        comparison_rows.append({
            "grupo": row["grupo"], "soporte_test": row["soporte_real"], "aciertos_dataset": row["aciertos"], "recall_dataset": row["recall_grupo"],
            "aciertos_profesor": professor_hits, "recall_profesor": professor_recall, "diferencia_pp": (row["recall_grupo"] - professor_recall) * 100,
            "comentario": "grupo3 tiene solo 2 casos; no sobreinterpretar diferencias" if row["grupo"] == "grupo3" else "diferencia documentada frente a la matriz de referencia",
        })
    pd.DataFrame(comparison_rows).to_csv(tables_dir / "paso14_comparacion_profesor.csv", index=False)

    pd.DataFrame([
        {"conjunto": "train", "accuracy": accuracy_train, "balanced_accuracy": balanced_train, "macro_f1": macro_f1_train},
        {"conjunto": "test", "accuracy": accuracy_test, "balanced_accuracy": balanced_test, "macro_f1": macro_f1},
    ]).to_csv(tables_dir / "paso14_train_vs_test.csv", index=False)

    classifier = pipeline.named_steps["model"]
    pd.DataFrame([{"node_count": classifier.tree_.node_count, "max_depth": classifier.tree_.max_depth, "n_leaves": classifier.get_n_leaves()}]).to_csv(metrics_dir / "paso14_arbol_complejidad.csv", index=False)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(tables_dir / "paso14_feature_importance.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(len(LABELS)), LABELS)
    plt.yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - Árbol de clasificación")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso14_matriz_confusion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(LABELS, [row["recall_grupo"] for row in group_rows])
    plt.ylim(0, 1)
    plt.xlabel("Grupo")
    plt.ylabel("Recall / exactitud por grupo")
    plt.title("Recall por grupo")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso14_recall_por_grupo.png", dpi=150)
    plt.close()

    top = importance.head(15).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importancia")
    plt.ylabel("Feature transformada")
    plt.title("Top 15 importancias - Árbol de clasificación")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso14_top_feature_importance.png", dpi=150)
    plt.close()

    pd.DataFrame([{
        "accuracy": accuracy_test, "balanced_accuracy": balanced_test, "macro_f1": macro_f1,
        "grupo1_recall": group_rows[0]["recall_grupo"], "grupo2_recall": group_rows[1]["recall_grupo"], "grupo3_recall": group_rows[2]["recall_grupo"],
        "grupo1_support": group_rows[0]["soporte_real"], "grupo2_support": group_rows[1]["soporte_real"], "grupo3_support": group_rows[2]["soporte_real"],
        "node_count": classifier.tree_.node_count, "max_depth": classifier.tree_.max_depth, "n_leaves": classifier.get_n_leaves(),
        "train_accuracy": accuracy_train, "test_accuracy": accuracy_test, "gap_accuracy": accuracy_train - accuracy_test,
    }]).to_csv(metrics_dir / "paso14_resumen.csv", index=False)

    print(f"Accuracy test: {accuracy_test}; balanced accuracy: {balanced_test}; macro F1: {macro_f1}")
    print("Matriz de confusión:")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())
    print(pd.DataFrame(group_rows).to_string(index=False))
    print(f"Accuracy train: {accuracy_train}; complejidad: {classifier.tree_.node_count} nodos, profundidad {classifier.tree_.max_depth}, {classifier.get_n_leaves()} hojas")
    print("SalePrice en X:", "SalePrice" in X.columns, "PriceGroup en X:", "PriceGroup" in X.columns, "Id en X:", "Id" in X.columns)


if __name__ == "__main__":
    main()
