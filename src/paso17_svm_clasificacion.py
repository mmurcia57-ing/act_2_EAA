"""Evaluación reproducible de SVM lineal y RBF para PriceGroup."""
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

LABELS = ["grupo1", "grupo2", "grupo3"]
MAPPING = {label: i for i, label in enumerate(LABELS)}
CONFIGS = [(kernel, C) for kernel in ("linear", "rbf") for C in (0.1, 1, 10)]

def assign_group(price: float) -> str:
    if price <= 100000: return "grupo1"
    if price <= 500000: return "grupo2"
    return "grupo3"

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["str", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

def calculate(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float, float]:
    return (float(accuracy_score(y_true, pred)),
            float(balanced_accuracy_score(y_true, pred)),
            float(precision_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            float(recall_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            float(f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)))

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    data["PriceGroup"] = data["SalePrice"].map(assign_group)
    split = pd.read_csv(root / "outputs" / "tables" / "paso13_split_indices.csv")
    train_idx = split.loc[split["conjunto"] == "train", "indice_original"].astype(int).tolist()
    test_idx = split.loc[split["conjunto"] == "test", "indice_original"].astype(int).tolist()
    y_labels = data["PriceGroup"]
    counts = y_labels.loc[test_idx].value_counts().reindex(LABELS, fill_value=0)
    if (len(train_idx), len(test_idx), counts.tolist()) != (1168, 292, [25, 265, 2]): raise RuntimeError("Split inválido")
    if split["indice_original"].duplicated().any() or set(train_idx) & set(test_idx): raise RuntimeError("Índices de split inválidos")
    X = data.drop(columns=["SalePrice", "PriceGroup", "Id"])
    if {"SalePrice", "PriceGroup", "Id"} & set(X.columns): raise RuntimeError("Variables excluidas presentes en X")
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train = y_labels.loc[train_idx].map(MAPPING).to_numpy(dtype=np.int32)
    y_test = y_labels.loc[test_idx].map(MAPPING).to_numpy(dtype=np.int32)
    if set(np.unique(np.concatenate([y_train, y_test]))) != {0, 1, 2}: raise RuntimeError("Mapping inválido")
    metrics_dir, tables_dir, figures_dir = [root / "outputs" / d for d in ("metrics", "tables", "figures")]
    for directory in (metrics_dir, tables_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    results, group_rows, matrices, fitted = [], [], {}, {}
    for kernel, C in CONFIGS:
        pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", SVC(kernel=kernel, C=C, class_weight=None))])
        start = time.perf_counter(); pipeline.fit(X_train, y_train); seconds = time.perf_counter() - start
        pred = pipeline.predict(X_test); values = calculate(y_test, pred)
        acc, balanced, macro_precision, macro_recall, macro_f1 = values
        weighted_f1 = float(f1_score(y_test, pred, average="weighted", zero_division=0))
        if not np.isfinite([*values, weighted_f1]).all(): raise RuntimeError("Métrica no finita")
        results.append({"kernel": kernel, "C": C, "accuracy": acc, "balanced_accuracy": balanced, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "train_seconds": seconds})
        matrix = confusion_matrix(y_test, pred, labels=[0, 1, 2]); matrices[(kernel, C)] = matrix; fitted[(kernel, C)] = pipeline
        for i, group in enumerate(LABELS):
            support, hits, predicted = int(matrix[i].sum()), int(matrix[i, i]), int(matrix[:, i].sum())
            precision = hits / predicted if predicted else 0.0; recall = hits / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            group_rows.append({"kernel": kernel, "C": C, "grupo": group, "support": support, "precision": precision, "recall": recall, "f1": f1})
        filename = f"paso17_cm_{kernel}_c{str(C).replace('.', '')}.csv"
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("real/predicho").to_csv(tables_dir / filename)
    result_df = pd.DataFrame(results); result_df.to_csv(tables_dir / "paso17_svm_resultados.csv", index=False)
    group_df = pd.DataFrame(group_rows); group_df.to_csv(tables_dir / "paso17_svm_metricas_por_grupo.csv", index=False)
    best = result_df.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False], kind="stable")
    top_balanced = best[best.balanced_accuracy == best.iloc[0].balanced_accuracy]
    top_f1 = top_balanced[top_balanced.macro_f1 == top_balanced.iloc[0].macro_f1]
    best_row = top_f1.sort_values("train_seconds", ascending=True).iloc[0]
    best_key = (best_row.kernel, float(best_row.C)); best_groups = group_df[(group_df.kernel == best_key[0]) & (group_df.C == best_key[1])].set_index("grupo")
    pd.DataFrame([{"kernel": best_row.kernel, "C": best_row.C, "accuracy": best_row.accuracy, "balanced_accuracy": best_row.balanced_accuracy, "macro_f1": best_row.macro_f1, "grupo1_recall": best_groups.loc["grupo1", "recall"], "grupo2_recall": best_groups.loc["grupo2", "recall"], "grupo3_recall": best_groups.loc["grupo3", "recall"], "train_seconds": best_row.train_seconds}]).to_csv(metrics_dir / "paso17_mejor_svm.csv", index=False)
    tree = pd.read_csv(metrics_dir / "paso14_arbol_clasificacion_metricas.csv").iloc[0]; rf = pd.read_csv(metrics_dir / "paso15_random_forest_clasificacion_metricas.csv").iloc[0]; xgb = pd.read_csv(metrics_dir / "paso16_xgboost_clasificacion_metricas.csv").iloc[0]
    tree_g = pd.read_csv(tables_dir / "paso14_metricas_por_grupo.csv").set_index("grupo"); rf_g = pd.read_csv(tables_dir / "paso15_metricas_por_grupo.csv").set_index("grupo"); xgb_g = pd.read_csv(tables_dir / "paso16_metricas_por_grupo.csv").set_index("grupo")
    def model_row(name, row, groups, recall_col):
        return {"modelo": name, "accuracy": row.accuracy, "balanced_accuracy": row.balanced_accuracy, "macro_f1": row.macro_f1, **{f"{g}_recall": groups.loc[g, recall_col] for g in LABELS}}
    comparison = pd.DataFrame([model_row("arbol", tree, tree_g, "recall_grupo"), model_row("random_forest", rf, rf_g, "recall"), model_row("xgboost", xgb, xgb_g, "recall"), {"modelo":"svm_mejor", "accuracy":best_row.accuracy, "balanced_accuracy":best_row.balanced_accuracy, "macro_f1":best_row.macro_f1, **{f"{g}_recall":best_groups.loc[g,"recall"] for g in LABELS}}])
    comparison.to_csv(tables_dir / "paso17_comparacion_clasificadores.csv", index=False)
    best_pipeline = fitted[best_key]; pred_train = best_pipeline.predict(X_train); train_values = calculate(y_train, pred_train); best_matrix = matrices[best_key]
    pd.DataFrame([{ "conjunto":"train", "accuracy":train_values[0], "balanced_accuracy":train_values[1], "macro_f1":train_values[4]}, {"conjunto":"test", "accuracy":best_row.accuracy, "balanced_accuracy":best_row.balanced_accuracy, "macro_f1":best_row.macro_f1}]).to_csv(tables_dir / "paso17_mejor_svm_train_vs_test.csv", index=False)
    pd.DataFrame([{ "best_kernel":best_row.kernel, "best_C":best_row.C, "best_accuracy":best_row.accuracy, "best_balanced_accuracy":best_row.balanced_accuracy, "best_macro_f1":best_row.macro_f1, "best_grupo1_recall":best_groups.loc["grupo1","recall"], "best_grupo2_recall":best_groups.loc["grupo2","recall"], "best_grupo3_recall":best_groups.loc["grupo3","recall"], "best_train_seconds":best_row.train_seconds, "best_train_accuracy":train_values[0], "best_test_accuracy":best_row.accuracy}]).to_csv(metrics_dir / "paso17_resumen.csv", index=False)
    labels = [f"{k}\nC={C}" for k, C in CONFIGS]
    plt.figure(figsize=(10, 5)); plt.plot(labels, result_df.accuracy, marker="o", label="Accuracy"); plt.plot(labels, result_df.balanced_accuracy, marker="o", label="Balanced accuracy"); plt.ylabel("Puntuación"); plt.title("SVM: accuracy y balanced accuracy"); plt.legend(); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(figures_dir / "paso17_svm_accuracy_comparacion.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 5)); plt.plot(labels, result_df.macro_f1, marker="o"); plt.ylabel("Macro F1"); plt.title("SVM: macro F1"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(figures_dir / "paso17_svm_macro_f1.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 5)); x = np.arange(6); width = .25
    for i, group in enumerate(LABELS): plt.bar(x + (i - 1) * width, group_df[group_df.grupo == group].recall, width, label=group)
    plt.xticks(x, labels, rotation=30); plt.ylim(0, 1); plt.ylabel("Recall"); plt.title("SVM: recall por grupo"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso17_svm_recall_por_grupo.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 6)); plt.imshow(best_matrix); plt.colorbar(); plt.xticks(range(3), LABELS); plt.yticks(range(3), LABELS)
    for i in range(3):
        for j in range(3): plt.text(j, i, str(best_matrix[i, j]), ha="center", va="center")
    plt.xlabel("Predicho"); plt.ylabel("Real"); plt.title(f"Matriz de confusión - SVM {best_row.kernel}, C={best_row.C}"); plt.tight_layout(); plt.savefig(figures_dir / "paso17_mejor_svm_matriz_confusion.png", dpi=150); plt.close()
    print(f"Mejor SVM: kernel={best_row.kernel}, C={best_row.C}; accuracy={best_row.accuracy}; balanced_accuracy={best_row.balanced_accuracy}; macro_f1={best_row.macro_f1}")
    print(result_df.to_string(index=False)); print(pd.DataFrame(best_matrix, index=LABELS, columns=LABELS).to_string()); print(f"Train accuracy mejor SVM: {train_values[0]}")

if __name__ == "__main__": main()
