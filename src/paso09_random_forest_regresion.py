"""Baseline reproducible de Random Forest para regresión de SalePrice."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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


def calculate_metrics(y_true: pd.Series, prediction: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, prediction)),
        float(np.sqrt(mean_squared_error(y_true, prediction))),
        float(r2_score(y_true, prediction)),
    )


def feature_mapping(preprocessor: ColumnTransformer) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve nombres transformados y su variable original por posición."""
    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    categorical_feature_names = encoder.get_feature_names_out(categorical_columns)
    transformed_names = preprocessor.get_feature_names_out()
    original_names = numeric_columns.copy()
    for column, categories in zip(categorical_columns, encoder.categories_):
        original_names.extend([column] * len(categories))
    original_names_array = np.asarray(original_names, dtype=object)
    if len(transformed_names) != len(original_names_array):
        raise RuntimeError("No coincide el mapeo de features transformadas y variables originales")
    return transformed_names, original_names_array


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)
    mae_train, rmse_train, r2_train = calculate_metrics(y_train, pred_train)
    mae_test, rmse_test, r2_test = calculate_metrics(y_test, pred_test)

    metrics_dir = root / "outputs" / "metrics"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    for directory in (metrics_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "modelo": "RandomForestRegressor",
        "n_estimators": 300, "n_train": len(X_train), "n_test": len(X_test),
        "mae_train": mae_train, "rmse_train": rmse_train, "r2_train": r2_train,
        "mae_test": mae_test, "rmse_test": rmse_test, "r2_test": r2_test,
        "train_seconds": train_seconds, "random_state": 42,
    }]).to_csv(metrics_dir / "paso09_random_forest_regresion_metricas.csv", index=False)

    tree_unpruned = {"modelo": "arbol_sin_poda", "mae_test": 27494.58, "rmse_test": 42453.07, "r2_test": 0.7650, "rmse_train": 0.00}
    tree_pruned = {"modelo": "arbol_podado", "mae_test": 25776.51, "rmse_test": 40386.27, "r2_test": 0.7874, "rmse_train": 19979.70}
    forest = {"modelo": "random_forest", "mae_test": mae_test, "rmse_test": rmse_test, "r2_test": r2_test, "rmse_train": rmse_train}
    regression_comparison = pd.DataFrame([tree_unpruned, tree_pruned, forest])
    regression_comparison["gap_rmse_train_test"] = regression_comparison["rmse_test"] - regression_comparison["rmse_train"]
    regression_comparison.to_csv(tables_dir / "paso09_comparacion_regresion.csv", index=False)

    improvements = pd.DataFrame([
        {"comparacion": "random_forest_vs_arbol_sin_poda", "delta_rmse": mae_test * 0 + rmse_test - tree_unpruned["rmse_test"], "delta_rmse_pct": (rmse_test - tree_unpruned["rmse_test"]) / tree_unpruned["rmse_test"] * 100},
        {"comparacion": "random_forest_vs_arbol_podado", "delta_rmse": rmse_test - tree_pruned["rmse_test"], "delta_rmse_pct": (rmse_test - tree_pruned["rmse_test"]) / tree_pruned["rmse_test"] * 100},
    ])
    improvements.to_csv(tables_dir / "paso09_mejora_random_forest.csv", index=False)

    professor = pd.DataFrame([{
        "modelo": "random_forest", "rmse_dataset": rmse_test, "rmse_profesor": 27588.9,
        "diferencia_absoluta": abs(rmse_test - 27588.9), "diferencia_pct": abs(rmse_test - 27588.9) / 27588.9 * 100,
        "comentario": "diferencia documentada; pueden influir split, R/Python, encoding, missing, parámetros y versiones",
    }])
    professor.to_csv(tables_dir / "paso09_comparacion_profesor.csv", index=False)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names, original_names = feature_mapping(preprocessor)
    model = pipeline.named_steps["model"]
    feature_importance = pd.DataFrame({"feature": transformed_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    feature_importance.to_csv(tables_dir / "paso09_feature_importance.csv", index=False)
    aggregate = pd.DataFrame({"variable_original": original_names, "importance": model.feature_importances_}).groupby("variable_original", as_index=False)["importance"].sum().rename(columns={"importance": "importance_total"}).sort_values("importance_total", ascending=False)
    aggregate.to_csv(tables_dir / "paso09_feature_importance_agregada.csv", index=False)

    predictions = pd.DataFrame({"indice_original": X_test.index, "SalePrice_real": y_test.to_numpy(), "SalePrice_predicho": pred_test})
    predictions["error"] = predictions["SalePrice_real"] - predictions["SalePrice_predicho"]
    predictions["error_absoluto"] = predictions["error"].abs()
    predictions.sort_values("indice_original").to_csv(tables_dir / "paso09_predicciones_test.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(feature_importance.head(15).sort_values("importance")["feature"], feature_importance.head(15).sort_values("importance")["importance"])
    plt.xlabel("Importancia")
    plt.ylabel("Feature transformada")
    plt.title("Top 15 importancias de Random Forest")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso09_top_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.barh(aggregate.head(15).sort_values("importance_total")["variable_original"], aggregate.head(15).sort_values("importance_total")["importance_total"])
    plt.xlabel("Importancia agregada")
    plt.ylabel("Variable original")
    plt.title("Top 15 importancias agregadas por variable original")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso09_top_importance_agregada.png", dpi=150)
    plt.close()

    limits = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred_test, alpha=0.7)
    plt.plot(limits, limits, linestyle="--", label="Predicción ideal")
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("Random Forest - Real vs. Predicho")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "paso09_real_vs_predicho.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(["Árbol sin poda", "Árbol podado", "Random Forest"], [tree_unpruned["rmse_test"], tree_pruned["rmse_test"], rmse_test])
    plt.ylabel("RMSE test")
    plt.title("Comparación de RMSE de modelos")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso09_comparacion_rmse.png", dpi=150)
    plt.close()

    improvement_unpruned = tree_unpruned["rmse_test"] - rmse_test
    improvement_pruned = tree_pruned["rmse_test"] - rmse_test
    pd.DataFrame([{
        "rmse_tree_unpruned": tree_unpruned["rmse_test"], "rmse_tree_pruned": tree_pruned["rmse_test"], "rmse_random_forest": rmse_test,
        "mae_random_forest": mae_test, "r2_random_forest": r2_test, "rmse_train_random_forest": rmse_train,
        "gap_train_test": rmse_test - rmse_train, "improvement_vs_unpruned_abs": improvement_unpruned, "improvement_vs_unpruned_pct": improvement_unpruned / tree_unpruned["rmse_test"] * 100,
        "improvement_vs_pruned_abs": improvement_pruned, "improvement_vs_pruned_pct": improvement_pruned / tree_pruned["rmse_test"] * 100,
        "n_estimators": 300, "train_seconds": train_seconds,
    }]).to_csv(metrics_dir / "paso09_resumen.csv", index=False)

    print(f"Train/test: {len(X_train)}/{len(X_test)}")
    print("Predictores originales: 79")
    print(f"Features después de encoding: {len(transformed_names)}")
    print(f"MAE train: {mae_train}; RMSE train: {rmse_train}; R2 train: {r2_train}")
    print(f"MAE test: {mae_test}; RMSE test: {rmse_test}; R2 test: {r2_test}")
    print(f"Gap RMSE train-test: {rmse_test - rmse_train}")
    print("Top 10 feature importances:")
    print(feature_importance.head(10).to_string(index=False))
    print("Top 10 importancias agregadas:")
    print(aggregate.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
