"""Baseline reproducible de XGBoost para regresión de SalePrice."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


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
    """Mapea cada feature transformada a su variable original por posición."""
    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    transformed_names = preprocessor.get_feature_names_out()
    original_names = numeric_columns.copy()
    for column, categories in zip(categorical_columns, encoder.categories_):
        original_names.extend([column] * len(categories))
    original_names = np.asarray(original_names, dtype=object)
    if len(transformed_names) != len(original_names):
        raise RuntimeError("No coincide el mapeo de features y variables originales")
    return transformed_names, original_names


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = make_preprocessor(X_train)
    model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
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

    version = xgboost.__version__
    pd.DataFrame([{
        "modelo": "XGBRegressor", "n_estimators": 500, "max_depth": 3, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "n_train": len(X_train), "n_test": len(X_test),
        "mae_train": mae_train, "rmse_train": rmse_train, "r2_train": r2_train,
        "mae_test": mae_test, "rmse_test": rmse_test, "r2_test": r2_test,
        "train_seconds": train_seconds, "random_state": 42, "xgboost_version": version,
    }]).to_csv(metrics_dir / "paso10_xgboost_regresion_metricas.csv", index=False)

    previous = pd.DataFrame([
        {"modelo": "arbol_sin_poda", "mae_test": 27494.58, "rmse_test": 42453.07, "r2_test": 0.7650, "rmse_train": 0.00},
        {"modelo": "arbol_podado", "mae_test": 25776.51, "rmse_test": 40386.27, "r2_test": 0.7874, "rmse_train": 19979.70},
        {"modelo": "random_forest", "mae_test": 17410.11, "rmse_test": 28929.45, "r2_test": 0.8909, "rmse_train": 11063.98},
        {"modelo": "xgboost", "mae_test": mae_test, "rmse_test": rmse_test, "r2_test": r2_test, "rmse_train": rmse_train},
    ])
    previous["gap_rmse_train_test"] = previous["rmse_test"] - previous["rmse_train"]
    previous.to_csv(tables_dir / "paso10_comparacion_regresion.csv", index=False)

    improvement_rows = []
    for reference, reference_rmse in [("arbol_sin_poda", 42453.07), ("arbol_podado", 40386.27), ("random_forest", 28929.45)]:
        improvement_rows.append({
            "comparacion": f"xgboost_vs_{reference}",
            "delta_rmse": rmse_test - reference_rmse,
            "delta_rmse_pct": (rmse_test - reference_rmse) / reference_rmse * 100,
        })
    pd.DataFrame(improvement_rows).to_csv(tables_dir / "paso10_mejora_xgboost.csv", index=False)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names, original_names = feature_mapping(preprocessor)
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance = pd.DataFrame({"feature": transformed_names, "importance": importances}).sort_values("importance", ascending=False)
    feature_importance.to_csv(tables_dir / "paso10_feature_importance.csv", index=False)
    aggregate = pd.DataFrame({"variable_original": original_names, "importance": importances}).groupby("variable_original", as_index=False)["importance"].sum().rename(columns={"importance": "importance_total"}).sort_values("importance_total", ascending=False)
    aggregate.to_csv(tables_dir / "paso10_feature_importance_agregada.csv", index=False)

    predictions = pd.DataFrame({"indice_original": X_test.index, "SalePrice_real": y_test.to_numpy(), "SalePrice_predicho": pred_test})
    predictions["error"] = predictions["SalePrice_real"] - predictions["SalePrice_predicho"]
    predictions["error_absoluto"] = predictions["error"].abs()
    predictions.sort_values("indice_original").to_csv(tables_dir / "paso10_predicciones_test.csv", index=False)

    professor_rmse = 27588.9
    pd.DataFrame([{
        "modelo": "xgboost", "rmse_dataset": rmse_test, "rmse_profesor": professor_rmse,
        "diferencia_absoluta": abs(rmse_test - professor_rmse), "diferencia_pct": abs(rmse_test - professor_rmse) / professor_rmse * 100,
        "comentario": "diferencia documentada; pueden influir split, R/Python, encoding, missing, parámetros y versiones",
    }]).to_csv(tables_dir / "paso10_comparacion_profesor.csv", index=False)

    improvement_unpruned = 42453.07 - rmse_test
    improvement_pruned = 40386.27 - rmse_test
    improvement_rf = 28929.45 - rmse_test
    pd.DataFrame([{
        "rmse_tree_unpruned": 42453.07, "rmse_tree_pruned": 40386.27, "rmse_random_forest": 28929.45, "rmse_xgboost": rmse_test,
        "mae_xgboost": mae_test, "r2_xgboost": r2_test, "rmse_train_xgboost": rmse_train, "gap_train_test": rmse_test - rmse_train,
        "improvement_vs_unpruned_abs": improvement_unpruned, "improvement_vs_unpruned_pct": improvement_unpruned / 42453.07 * 100,
        "improvement_vs_pruned_abs": improvement_pruned, "improvement_vs_pruned_pct": improvement_pruned / 40386.27 * 100,
        "improvement_vs_rf_abs": improvement_rf, "improvement_vs_rf_pct": improvement_rf / 28929.45 * 100,
        "n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "train_seconds": train_seconds,
    }]).to_csv(metrics_dir / "paso10_resumen.csv", index=False)

    for values, title, ylabel, filename in [
        (feature_importance.head(15).sort_values("importance"), "Top 15 importancias de XGBoost", "Feature transformada", "paso10_top_feature_importance.png"),
        (aggregate.head(15).sort_values("importance_total"), "Top 15 importancias agregadas", "Variable original", "paso10_top_importance_agregada.png"),
    ]:
        value_column = "importance" if "importance" in values.columns else "importance_total"
        label_column = "feature" if "feature" in values.columns else "variable_original"
        plt.figure(figsize=(9, 7))
        plt.barh(values[label_column], values[value_column])
        plt.xlabel("Importancia")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150)
        plt.close()

    limits = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred_test, alpha=0.7)
    plt.plot(limits, limits, linestyle="--", label="Predicción ideal")
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("XGBoost - Real vs. Predicho")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "paso10_real_vs_predicho.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(["Árbol sin poda", "Árbol podado", "Random Forest", "XGBoost"], [42453.07, 40386.27, 28929.45, rmse_test])
    plt.ylabel("RMSE test")
    plt.title("Comparación de RMSE de modelos")
    plt.tight_layout()
    plt.savefig(figures_dir / "paso10_comparacion_rmse.png", dpi=150)
    plt.close()

    print(f"XGBoost version: {version}")
    print(f"Train/test: {len(X_train)}/{len(X_test)}")
    print(f"MAE train: {mae_train}; RMSE train: {rmse_train}; R2 train: {r2_train}")
    print(f"MAE test: {mae_test}; RMSE test: {rmse_test}; R2 test: {r2_test}")
    print(f"Gap RMSE train-test: {rmse_test - rmse_train}")
    print("Top 10 feature importances:")
    print(feature_importance.head(10).to_string(index=False))
    print("Top 10 importancias agregadas:")
    print(aggregate.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
