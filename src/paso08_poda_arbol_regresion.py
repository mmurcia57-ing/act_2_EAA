"""Compara un árbol de regresión sin poda y otro con cost-complexity pruning."""
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


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


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(r2_score(y_true, y_pred)),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Baseline reproducible del Paso 07.
    baseline_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", DecisionTreeRegressor(random_state=42)),
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_model = baseline_pipeline.named_steps["model"]
    baseline_train_pred = baseline_pipeline.predict(X_train)
    baseline_test_pred = baseline_pipeline.predict(X_test)
    baseline_mae_test, baseline_rmse_test, baseline_r2_test = metrics(y_test, baseline_test_pred)
    baseline_mae_train, baseline_rmse_train, baseline_r2_train = metrics(y_train, baseline_train_pred)

    # Selección interna: ningún dato del test externo participa en la elección.
    X_inner_train, X_validation, y_inner_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42
    )
    inner_preprocessor = make_preprocessor(X_inner_train)
    X_inner_transformed = inner_preprocessor.fit_transform(X_inner_train, y_inner_train)
    X_validation_transformed = inner_preprocessor.transform(X_validation)
    unpruned_inner = DecisionTreeRegressor(random_state=42)
    pruning_path = unpruned_inner.cost_complexity_pruning_path(X_inner_transformed, y_inner_train)
    all_alphas = np.unique(np.maximum(pruning_path.ccp_alphas, 0))
    if len(all_alphas) > 40:
        positions = np.unique(np.linspace(0, len(all_alphas) - 1, 40, dtype=int))
        alphas = all_alphas[positions]
    else:
        alphas = all_alphas

    search_rows = []
    for alpha in alphas:
        candidate = DecisionTreeRegressor(random_state=42, ccp_alpha=float(alpha))
        candidate.fit(X_inner_transformed, y_inner_train)
        pred_inner = candidate.predict(X_inner_transformed)
        pred_validation = candidate.predict(X_validation_transformed)
        search_rows.append({
            "ccp_alpha": float(alpha),
            "rmse_train_interno": float(np.sqrt(mean_squared_error(y_inner_train, pred_inner))),
            "rmse_validation": float(np.sqrt(mean_squared_error(y_validation, pred_validation))),
            "node_count": candidate.tree_.node_count,
            "max_depth": candidate.tree_.max_depth,
            "n_leaves": candidate.get_n_leaves(),
        })
    search = pd.DataFrame(search_rows).sort_values("ccp_alpha").reset_index(drop=True)
    selected_row = search.sort_values(["rmse_validation", "node_count", "ccp_alpha"]).iloc[0]
    selected_alpha = float(selected_row["ccp_alpha"])

    final_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", DecisionTreeRegressor(random_state=42, ccp_alpha=selected_alpha)),
    ])
    start = time.perf_counter()
    final_pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    pruned_model = final_pipeline.named_steps["model"]
    pruned_train_pred = final_pipeline.predict(X_train)
    pruned_test_pred = final_pipeline.predict(X_test)
    pruned_mae_test, pruned_rmse_test, pruned_r2_test = metrics(y_test, pruned_test_pred)
    pruned_mae_train, pruned_rmse_train, pruned_r2_train = metrics(y_train, pruned_train_pred)

    out_tables = root / "outputs" / "tables"
    out_metrics = root / "outputs" / "metrics"
    out_figures = root / "outputs" / "figures"
    for directory in (out_tables, out_metrics, out_figures):
        directory.mkdir(parents=True, exist_ok=True)
    search.to_csv(out_tables / "paso08_busqueda_ccp_alpha.csv", index=False)

    complexity = pd.DataFrame([
        {"modelo": "arbol_sin_poda", "ccp_alpha": 0.0, "node_count": baseline_model.tree_.node_count, "max_depth": baseline_model.tree_.max_depth, "n_leaves": baseline_model.get_n_leaves()},
        {"modelo": "arbol_podado", "ccp_alpha": selected_alpha, "node_count": pruned_model.tree_.node_count, "max_depth": pruned_model.tree_.max_depth, "n_leaves": pruned_model.get_n_leaves()},
    ])
    complexity.to_csv(out_tables / "paso08_complejidad_comparada.csv", index=False)

    delta_rmse = pruned_rmse_test - baseline_rmse_test
    model_comparison = pd.DataFrame([
        {"modelo": "arbol_sin_poda", "rmse_train": baseline_rmse_train, "rmse_test": baseline_rmse_test, "mae_test": baseline_mae_test, "r2_test": baseline_r2_test, "gap_rmse_train_test": baseline_rmse_test - baseline_rmse_train, "node_count": baseline_model.tree_.node_count, "max_depth": baseline_model.tree_.max_depth, "n_leaves": baseline_model.get_n_leaves(), "delta_rmse_test": 0.0, "delta_rmse_pct": 0.0},
        {"modelo": "arbol_podado", "rmse_train": pruned_rmse_train, "rmse_test": pruned_rmse_test, "mae_test": pruned_mae_test, "r2_test": pruned_r2_test, "gap_rmse_train_test": pruned_rmse_test - pruned_rmse_train, "node_count": pruned_model.tree_.node_count, "max_depth": pruned_model.tree_.max_depth, "n_leaves": pruned_model.get_n_leaves(), "delta_rmse_test": delta_rmse, "delta_rmse_pct": delta_rmse / baseline_rmse_test * 100},
    ])
    model_comparison.to_csv(out_tables / "paso08_comparacion_modelos.csv", index=False)

    professor = pd.DataFrame([
        {"modelo": "arbol_sin_poda", "rmse_dataset": baseline_rmse_test, "rmse_profesor": 41002.2, "diferencia_absoluta": abs(baseline_rmse_test - 41002.2), "diferencia_pct": abs(baseline_rmse_test - 41002.2) / 41002.2 * 100, "comentario": "diferencia documentada; no se forzaron parámetros"},
        {"modelo": "arbol_podado", "rmse_dataset": pruned_rmse_test, "rmse_profesor": 42927.25, "diferencia_absoluta": abs(pruned_rmse_test - 42927.25), "diferencia_pct": abs(pruned_rmse_test - 42927.25) / 42927.25 * 100, "comentario": "diferencia documentada; alpha elegido con validación interna"},
    ])
    professor.to_csv(out_tables / "paso08_comparacion_profesor.csv", index=False)

    pd.DataFrame([{
        "rmse_baseline": baseline_rmse_test, "rmse_podado": pruned_rmse_test,
        "mae_baseline": baseline_mae_test, "mae_podado": pruned_mae_test,
        "r2_baseline": baseline_r2_test, "r2_podado": pruned_r2_test,
        "rmse_train_baseline": baseline_rmse_train, "rmse_train_podado": pruned_rmse_train,
        "alpha_seleccionado": selected_alpha, "nodes_baseline": baseline_model.tree_.node_count, "nodes_podado": pruned_model.tree_.node_count,
        "depth_baseline": baseline_model.tree_.max_depth, "depth_podado": pruned_model.tree_.max_depth,
        "leaves_baseline": baseline_model.get_n_leaves(), "leaves_podado": pruned_model.get_n_leaves(),
        "delta_rmse": delta_rmse, "delta_rmse_pct": delta_rmse / baseline_rmse_test * 100,
    }]).to_csv(out_metrics / "paso08_poda_resumen.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(["Árbol sin poda", "Árbol podado", "Referencia sin poda", "Referencia podada"], [baseline_rmse_test, pruned_rmse_test, 41002.2, 42927.25])
    plt.ylabel("RMSE")
    plt.title("Comparación de RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_figures / "paso08_rmse_comparacion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(["Sin poda", "Podado"], [baseline_model.tree_.node_count, pruned_model.tree_.node_count])
    plt.ylabel("Número de nodos")
    plt.title("Complejidad del árbol")
    plt.tight_layout()
    plt.savefig(out_figures / "paso08_complejidad_arbol.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(search["ccp_alpha"], search["rmse_validation"], marker="o")
    plt.axvline(selected_alpha, linestyle="--", label=f"Alpha elegido: {selected_alpha:.6g}")
    plt.xlabel("ccp_alpha")
    plt.ylabel("RMSE validation")
    plt.title("Selección de poda con validación interna")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figures / "paso08_ccp_alpha_validacion.png", dpi=150)
    plt.close()

    print(f"RMSE baseline: {baseline_rmse_test}")
    print(f"Alpha seleccionado: {selected_alpha}")
    print(f"RMSE train podado: {pruned_rmse_train}")
    print(f"RMSE test podado: {pruned_rmse_test}")
    print(f"MAE test podado: {pruned_mae_test}")
    print(f"R2 test podado: {pruned_r2_test}")
    print(f"Delta RMSE: {delta_rmse} ({delta_rmse / baseline_rmse_test * 100}%)")
    print(f"Complejidad baseline -> podado: nodos {baseline_model.tree_.node_count}->{pruned_model.tree_.node_count}, profundidad {baseline_model.tree_.max_depth}->{pruned_model.tree_.max_depth}, hojas {baseline_model.get_n_leaves()}->{pruned_model.get_n_leaves()}")


if __name__ == "__main__":
    main()
