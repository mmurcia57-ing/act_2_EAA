"""Random Forest baseline para clasificación de PriceGroup."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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


def metrics(y_true: pd.Series, pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(accuracy_score(y_true, pred)),
        float(balanced_accuracy_score(y_true, pred)),
        float(f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    data["PriceGroup"] = data["SalePrice"].map(assign_group)
    split = pd.read_csv(root / "outputs" / "tables" / "paso13_split_indices.csv")
    train_idx = split.loc[split["conjunto"] == "train", "indice_original"].astype(int).tolist()
    test_idx = split.loc[split["conjunto"] == "test", "indice_original"].astype(int).tolist()
    y = data["PriceGroup"]
    test_counts = y.loc[test_idx].value_counts().reindex(LABELS, fill_value=0)
    if test_counts.tolist() != [25, 265, 2]:
        raise RuntimeError("El split reutilizado no contiene test 25/265/2")
    X = data.drop(columns=["SalePrice", "PriceGroup", "Id"])
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)
    accuracy_train, balanced_train, macro_f1_train = metrics(y_train, pred_train)
    accuracy_test, balanced_test, macro_f1_test = metrics(y_test, pred_test)
    macro_precision = float(precision_score(y_test, pred_test, labels=LABELS, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, pred_test, labels=LABELS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, pred_test, average="weighted", zero_division=0))

    metrics_dir = root / "outputs" / "metrics"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    for directory in (metrics_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "modelo": "RandomForestClassifier", "n_estimators": 300, "n_train": len(X_train), "n_test": len(X_test),
        "accuracy": accuracy_test, "balanced_accuracy": balanced_test, "macro_precision": macro_precision,
        "macro_recall": macro_recall, "macro_f1": macro_f1_test, "weighted_f1": weighted_f1,
        "train_seconds": train_seconds, "random_state": 42,
    }]).to_csv(metrics_dir / "paso15_random_forest_clasificacion_metricas.csv", index=False)

    matrix = confusion_matrix(y_test, pred_test, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("real/predicho").to_csv(tables_dir / "paso15_matriz_confusion.csv")
    group_rows = []
    for i, group in enumerate(LABELS):
        support = int(matrix[i, :].sum())
        hits = int(matrix[i, i])
        predicted_total = int(matrix[:, i].sum())
        precision = hits / predicted_total if predicted_total else 0
        recall = hits / support if support else 0
        group_rows.append({"grupo": group, "soporte_real": support, "aciertos": hits, "errores": support - hits, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0})
    pd.DataFrame(group_rows).to_csv(tables_dir / "paso15_metricas_por_grupo.csv", index=False)
    report = classification_report(y_test, pred_test, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.rename_axis("grupo").to_csv(tables_dir / "paso15_classification_report.csv")

    professor = {"grupo1": (24, 13, 13 / 24), "grupo2": (266, 265, 265 / 266), "grupo3": (2, 1, 0.5)}
    professor_rows = []
    for row in group_rows:
        support_ref, hits_ref, recall_ref = professor[row["grupo"]]
        """
        professor_rows.append({"grupo": row["grupo"], "soporte_dataset": row["soporte_real"], "soporte_profesor": support_ref, "aciertos_dataset": row["aciertos"], "recall_dataset": row["recall"], "aciertos_profesor": hits_ref, "recall_profesor": recall_ref, "comentario": "splits ligeramente distintos; grupo3 tiene solo 2 casos y no debe generalizarse" if row["grupo"] == "grupo3" else "comparación documentada sin cambiar el split")
        """
        professor_rows.append({
            "grupo": row["grupo"], "soporte_dataset": row["soporte_real"], "soporte_profesor": support_ref,
            "aciertos_dataset": row["aciertos"], "recall_dataset": row["recall"],
            "aciertos_profesor": hits_ref, "recall_profesor": recall_ref,
            "comentario": "splits distintos; grupo3 tiene solo 2 casos y no debe generalizarse" if row["grupo"] == "grupo3" else "comparacion documentada sin cambiar el split",
        })
    pd.DataFrame(professor_rows).to_csv(tables_dir / "paso15_comparacion_profesor.csv", index=False)

    tree_metrics = pd.read_csv(root / "outputs" / "metrics" / "paso14_arbol_clasificacion_metricas.csv").iloc[0]
    tree_group = pd.read_csv(root / "outputs" / "tables" / "paso14_metricas_por_grupo.csv").set_index("grupo")
    comparison = pd.DataFrame([
        {"modelo": "arbol_clasificacion", "accuracy": tree_metrics["accuracy"], "balanced_accuracy": tree_metrics["balanced_accuracy"], "macro_f1": tree_metrics["macro_f1"], "grupo1_recall": tree_group.loc["grupo1", "recall_grupo"], "grupo2_recall": tree_group.loc["grupo2", "recall_grupo"], "grupo3_recall": tree_group.loc["grupo3", "recall_grupo"]},
        {"modelo": "random_forest", "accuracy": accuracy_test, "balanced_accuracy": balanced_test, "macro_f1": macro_f1_test, "grupo1_recall": group_rows[0]["recall"], "grupo2_recall": group_rows[1]["recall"], "grupo3_recall": group_rows[2]["recall"]},
    ])
    comparison.to_csv(tables_dir / "paso15_comparacion_arbol_rf.csv", index=False)
    tree_row = comparison.iloc[0]
    rf_row = comparison.iloc[1]
    pd.DataFrame([{
        "delta_accuracy": rf_row["accuracy"] - tree_row["accuracy"], "delta_balanced_accuracy": rf_row["balanced_accuracy"] - tree_row["balanced_accuracy"], "delta_macro_f1": rf_row["macro_f1"] - tree_row["macro_f1"],
        "delta_grupo1_recall": rf_row["grupo1_recall"] - tree_row["grupo1_recall"], "delta_grupo2_recall": rf_row["grupo2_recall"] - tree_row["grupo2_recall"], "delta_grupo3_recall": rf_row["grupo3_recall"] - tree_row["grupo3_recall"],
    }]).to_csv(tables_dir / "paso15_mejora_vs_arbol.csv", index=False)

    pd.DataFrame([
        {"conjunto": "train", "accuracy": accuracy_train, "balanced_accuracy": balanced_train, "macro_f1": macro_f1_train},
        {"conjunto": "test", "accuracy": accuracy_test, "balanced_accuracy": balanced_test, "macro_f1": macro_f1_test},
    ]).to_csv(tables_dir / "paso15_train_vs_test.csv", index=False)

    model = pipeline.named_steps["model"]
    features = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(tables_dir / "paso15_feature_importance.csv", index=False)
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_cols = list(preprocessor.transformers_[0][2])
    categorical_cols = list(preprocessor.transformers_[1][2])
    originals = numeric_cols.copy()
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    for column, categories in zip(categorical_cols, encoder.categories_):
        originals.extend([column] * len(categories))
    aggregate = pd.DataFrame({"variable_original": originals, "importance": model.feature_importances_}).groupby("variable_original", as_index=False)["importance"].sum().rename(columns={"importance": "importance_total"}).sort_values("importance_total", ascending=False)
    aggregate.to_csv(tables_dir / "paso15_feature_importance_agregada.csv", index=False)

    for values, label_column, value_column, title, ylabel, filename in [
        (importance.head(15).sort_values("importance"), "feature", "importance", "Top 15 importancias - Random Forest", "Feature transformada", "paso15_top_feature_importance.png"),
        (aggregate.head(15).sort_values("importance_total"), "variable_original", "importance_total", "Top 15 importancias agregadas", "Variable original", "paso15_top_importance_agregada.png"),
    ]:
        plt.figure(figsize=(9, 7))
        plt.barh(values[label_column], values[value_column])
        plt.xlabel("Importancia")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(3), LABELS)
    plt.yticks(range(3), LABELS)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - Random Forest")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso15_matriz_confusion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(3)
    width = 0.35
    plt.bar(x - width / 2, comparison.iloc[0][["grupo1_recall", "grupo2_recall", "grupo3_recall"]].to_numpy(), width, label="Árbol")
    plt.bar(x + width / 2, comparison.iloc[1][["grupo1_recall", "grupo2_recall", "grupo3_recall"]].to_numpy(), width, label="Random Forest")
    plt.xticks(x, LABELS)
    plt.ylim(0, 1)
    plt.ylabel("Recall / exactitud por grupo")
    plt.title("Recall por grupo: árbol vs. Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "paso15_recall_por_grupo.png", dpi=150)
    plt.close()

    pd.DataFrame([{
        "accuracy": accuracy_test, "balanced_accuracy": balanced_test, "macro_f1": macro_f1_test,
        "grupo1_recall": group_rows[0]["recall"], "grupo2_recall": group_rows[1]["recall"], "grupo3_recall": group_rows[2]["recall"],
        "grupo1_support": group_rows[0]["soporte_real"], "grupo2_support": group_rows[1]["soporte_real"], "grupo3_support": group_rows[2]["soporte_real"],
        "train_accuracy": accuracy_train, "test_accuracy": accuracy_test, "gap_accuracy": accuracy_train - accuracy_test,
        "delta_accuracy_vs_tree": accuracy_test - tree_metrics["accuracy"], "delta_balanced_accuracy_vs_tree": balanced_test - tree_metrics["balanced_accuracy"], "delta_macro_f1_vs_tree": macro_f1_test - tree_metrics["macro_f1"],
        "n_estimators": 300, "train_seconds": train_seconds,
    }]).to_csv(metrics_dir / "paso15_resumen.csv", index=False)

    print(f"Accuracy: {accuracy_test}; balanced accuracy: {balanced_test}; macro F1: {macro_f1_test}")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())
    print(pd.DataFrame(group_rows).to_string(index=False))
    print(f"Train accuracy: {accuracy_train}; features: {len(features)}")
    print("Top 10 importancias:")
    print(importance.head(10).to_string(index=False))
    print("Top 10 agregadas:")
    print(aggregate.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
