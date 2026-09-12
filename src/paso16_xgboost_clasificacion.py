"""Baseline XGBoost multiclase para PriceGroup usando el split del Paso 13."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LABELS = ["grupo1", "grupo2", "grupo3"]
MAPPING = {label: i for i, label in enumerate(LABELS)}

def assign_group(price: float) -> str:
    if price <= 100000: return "grupo1"
    if price <= 500000: return "grupo2"
    return "grupo3"

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["str", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

def metrics(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    return (float(accuracy_score(y_true, pred)),
            float(balanced_accuracy_score(y_true, pred)),
            float(f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)))

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    data["PriceGroup"] = data["SalePrice"].map(assign_group)
    split = pd.read_csv(root / "outputs" / "tables" / "paso13_split_indices.csv")
    train_idx = split.loc[split["conjunto"] == "train", "indice_original"].astype(int).tolist()
    test_idx = split.loc[split["conjunto"] == "test", "indice_original"].astype(int).tolist()
    y_labels = data["PriceGroup"]
    test_counts = y_labels.loc[test_idx].value_counts().reindex(LABELS, fill_value=0)
    if (len(train_idx), len(test_idx), test_counts.tolist()) != (1168, 292, [25, 265, 2]):
        raise RuntimeError("El split reutilizado no coincide con el Paso 13")
    if split["indice_original"].duplicated().any() or set(train_idx) & set(test_idx):
        raise RuntimeError("Split inválido o con índices duplicados")
    X = data.drop(columns=["SalePrice", "PriceGroup", "Id"])
    if {"SalePrice", "PriceGroup", "Id"} & set(X.columns): raise RuntimeError("Variables excluidas presentes en X")
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train = y_labels.loc[train_idx].map(MAPPING).to_numpy(dtype=np.int32)
    y_test = y_labels.loc[test_idx].map(MAPPING).to_numpy(dtype=np.int32)
    if set(np.unique(np.concatenate([y_train, y_test]))) != {0, 1, 2}: raise RuntimeError("Mapping inválido")
    model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", model)])
    start = time.perf_counter(); pipeline.fit(X_train, y_train); train_seconds = time.perf_counter() - start
    pred_train = pipeline.predict(X_train).astype(int); pred_test = pipeline.predict(X_test).astype(int)
    acc_train, bal_train, f1_train = metrics(y_train, pred_train)
    acc, bal, macro_f1 = metrics(y_test, pred_test)
    macro_precision = float(precision_score(y_test, pred_test, labels=[0, 1, 2], average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, pred_test, labels=[0, 1, 2], average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, pred_test, average="weighted", zero_division=0))
    if not np.isfinite([acc, bal, macro_precision, macro_recall, macro_f1, weighted_f1]).all(): raise RuntimeError("Métrica no finita")
    metrics_dir, tables_dir, figures_dir = [root / "outputs" / d for d in ("metrics", "tables", "figures")]
    for directory in (metrics_dir, tables_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"modelo":"XGBClassifier", "version_xgboost":xgb.__version__, "n_estimators":300, "max_depth":3, "learning_rate":0.05, "subsample":0.8, "colsample_bytree":0.8, "n_train":len(X_train), "n_test":len(X_test), "accuracy":acc, "balanced_accuracy":bal, "macro_precision":macro_precision, "macro_recall":macro_recall, "macro_f1":macro_f1, "weighted_f1":weighted_f1, "train_seconds":train_seconds, "random_state":42}]).to_csv(metrics_dir / "paso16_xgboost_clasificacion_metricas.csv", index=False)
    matrix = confusion_matrix(y_test, pred_test, labels=[0, 1, 2])
    if matrix.shape != (3, 3) or matrix.sum() != 292: raise RuntimeError("Matriz inválida")
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("real/predicho").to_csv(tables_dir / "paso16_matriz_confusion.csv")
    group_rows = []
    for i, group in enumerate(LABELS):
        support, hits, predicted = int(matrix[i].sum()), int(matrix[i,i]), int(matrix[:,i].sum())
        precision = hits / predicted if predicted else 0.0; recall = hits / support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        group_rows.append({"grupo":group, "soporte":support, "aciertos":hits, "errores":support-hits, "precision":precision, "recall":recall, "f1":f1})
    pd.DataFrame(group_rows).to_csv(tables_dir / "paso16_metricas_por_grupo.csv", index=False)
    report = classification_report(y_test, pred_test, labels=[0,1,2], target_names=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.rename_axis("grupo").to_csv(tables_dir / "paso16_classification_report.csv")
    tree = pd.read_csv(metrics_dir / "paso14_arbol_clasificacion_metricas.csv").iloc[0]
    rf = pd.read_csv(metrics_dir / "paso15_random_forest_clasificacion_metricas.csv").iloc[0]
    tree_groups = pd.read_csv(tables_dir / "paso14_metricas_por_grupo.csv").set_index("grupo")
    rf_groups = pd.read_csv(tables_dir / "paso15_metricas_por_grupo.csv").set_index("grupo")
    comparison = pd.DataFrame([
        {"modelo":"arbol", "accuracy":tree.accuracy, "balanced_accuracy":tree.balanced_accuracy, "macro_f1":tree.macro_f1, **{f"{g}_recall":tree_groups.loc[g,"recall_grupo"] for g in LABELS}},
        {"modelo":"random_forest", "accuracy":rf.accuracy, "balanced_accuracy":rf.balanced_accuracy, "macro_f1":rf.macro_f1, **{f"{g}_recall":rf_groups.loc[g,"recall"] for g in LABELS}},
        {"modelo":"xgboost", "accuracy":acc, "balanced_accuracy":bal, "macro_f1":macro_f1, **{f"{g}_recall":group_rows[i]["recall"] for i,g in enumerate(LABELS)}},
    ])
    comparison.to_csv(tables_dir / "paso16_comparacion_clasificadores.csv", index=False)
    xrow = comparison.iloc[2]; delta_rows = []
    for name, ref in (("arbol", comparison.iloc[0]), ("random_forest", comparison.iloc[1])):
        row = {"comparacion":f"xgboost_vs_{name}", "delta_accuracy":xrow.accuracy-ref.accuracy, "delta_balanced_accuracy":xrow.balanced_accuracy-ref.balanced_accuracy, "delta_macro_f1":xrow.macro_f1-ref.macro_f1}
        row.update({f"delta_{g}_recall":xrow[f"{g}_recall"]-ref[f"{g}_recall"] for g in LABELS}); delta_rows.append(row)
    pd.DataFrame(delta_rows).to_csv(tables_dir / "paso16_mejora_xgboost.csv", index=False)
    pd.DataFrame([{"conjunto":"train", "accuracy":acc_train, "balanced_accuracy":bal_train, "macro_f1":f1_train}, {"conjunto":"test", "accuracy":acc, "balanced_accuracy":bal, "macro_f1":macro_f1}]).to_csv(tables_dir / "paso16_train_vs_test.csv", index=False)
    prep, fitted = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    feature_names = prep.get_feature_names_out(); importance = pd.DataFrame({"feature":feature_names, "importance":fitted.feature_importances_}).sort_values("importance", ascending=False); importance.to_csv(tables_dir / "paso16_feature_importance.csv", index=False)
    originals = list(prep.transformers_[0][2]); cats = list(prep.transformers_[1][2]); encoder = prep.named_transformers_["cat"].named_steps["encoder"]
    for col, categories in zip(cats, encoder.categories_): originals.extend([col]*len(categories))
    aggregate = pd.DataFrame({"variable_original":originals, "importance":fitted.feature_importances_}).groupby("variable_original", as_index=False).importance.sum().rename(columns={"importance":"importance_total"}).sort_values("importance_total", ascending=False); aggregate.to_csv(tables_dir / "paso16_feature_importance_agregada.csv", index=False)
    for values, col, value_col, title, ylabel, filename in [(importance.head(15).sort_values("importance"),"feature","importance","Top 15 importancias - XGBoost","Feature transformada","paso16_top_feature_importance.png"),(aggregate.head(15).sort_values("importance_total"),"variable_original","importance_total","Top 15 importancias agregadas - XGBoost","Variable original","paso16_top_importance_agregada.png")]:
        plt.figure(figsize=(9,7)); plt.barh(values[col], values[value_col]); plt.xlabel("Importancia"); plt.ylabel(ylabel); plt.title(title); plt.tight_layout(); plt.savefig(figures_dir / filename, dpi=150); plt.close()
    plt.figure(figsize=(7,6)); plt.imshow(matrix); plt.colorbar(); plt.xticks(range(3),LABELS); plt.yticks(range(3),LABELS)
    for i in range(3):
        for j in range(3): plt.text(j,i,str(matrix[i,j]),ha="center",va="center")
    plt.xlabel("Predicho"); plt.ylabel("Real"); plt.title("Matriz de confusión - XGBoost"); plt.tight_layout(); plt.savefig(figures_dir / "paso16_matriz_confusion.png", dpi=150); plt.close()
    plt.figure(figsize=(8,5)); x=np.arange(3); width=.25
    for i,(_, row) in enumerate(comparison.iterrows()): plt.bar(x+(i-1)*width,[row[f"{g}_recall"] for g in LABELS],width,label=row.modelo)
    plt.xticks(x,LABELS); plt.ylim(0,1); plt.ylabel("Recall / exactitud por grupo"); plt.title("Recall por grupo"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso16_recall_por_grupo.png", dpi=150); plt.close()
    pd.DataFrame([{ "accuracy":acc, "balanced_accuracy":bal, "macro_f1":macro_f1, **{f"{g}_recall":group_rows[i]["recall"] for i,g in enumerate(LABELS)}, "train_accuracy":acc_train, "test_accuracy":acc, "gap_accuracy":acc_train-acc, "delta_accuracy_vs_tree":xrow.accuracy-tree.accuracy, "delta_balanced_accuracy_vs_tree":xrow.balanced_accuracy-tree.balanced_accuracy, "delta_macro_f1_vs_tree":xrow.macro_f1-tree.macro_f1, "delta_accuracy_vs_rf":xrow.accuracy-rf.accuracy, "delta_balanced_accuracy_vs_rf":xrow.balanced_accuracy-rf.balanced_accuracy, "delta_macro_f1_vs_rf":xrow.macro_f1-rf.macro_f1, "n_estimators":300, "max_depth":3, "learning_rate":.05, "train_seconds":train_seconds }]).to_csv(metrics_dir / "paso16_resumen.csv", index=False)
    print(f"XGBoost {xgb.__version__}; Accuracy: {acc}; balanced accuracy: {bal}; macro F1: {macro_f1}"); print(pd.DataFrame(matrix,index=LABELS,columns=LABELS).to_string()); print(pd.DataFrame(group_rows).to_string(index=False)); print(f"Train accuracy: {acc_train}; Test accuracy: {acc}"); print("Top 10 importancias:"); print(importance.head(10).to_string(index=False)); print("Top 10 agregadas:"); print(aggregate.head(10).to_string(index=False))

if __name__ == "__main__": main()
