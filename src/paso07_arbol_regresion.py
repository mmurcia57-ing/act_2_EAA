"""Baseline de árbol de decisión para regresión de SalePrice sin leakage."""
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


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(r2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(include=["str", "category"]).columns.tolist()
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42)),
    ])

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    mae_test, rmse_test, r2_test = regression_metrics(y_test, y_pred_test)
    mae_train, rmse_train, r2_train = regression_metrics(y_train, y_pred_train)

    out_metrics = root / "outputs" / "metrics"
    out_tables = root / "outputs" / "tables"
    out_figures = root / "outputs" / "figures"
    for directory in (out_metrics, out_tables, out_figures):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "modelo": "DecisionTreeRegressor_sin_poda",
        "n_train": len(X_train), "n_test": len(X_test),
        "mae": mae_test, "rmse": rmse_test, "r2": r2_test,
        "train_seconds": train_seconds, "random_state": 42,
    }]).to_csv(out_metrics / "paso07_arbol_regresion_metricas.csv", index=False)

    model = pipeline.named_steps["model"]
    pd.DataFrame([{
        "node_count": model.tree_.node_count,
        "max_depth": model.tree_.max_depth,
        "n_leaves": model.get_n_leaves(),
    }]).to_csv(out_metrics / "paso07_arbol_complejidad.csv", index=False)

    comparison = pd.DataFrame([{
        "metrica": "RMSE",
        "valor_dataset": rmse_test,
        "valor_profesor": 41002.2,
        "diferencia_absoluta": abs(rmse_test - 41002.2),
        "diferencia_pct": abs(rmse_test - 41002.2) / 41002.2 * 100,
        "comentario": "diferencia documentada; no se modificaron semilla, split, preprocesamiento ni hiperparámetros",
    }])
    comparison.to_csv(out_tables / "paso07_comparacion_profesor.csv", index=False)

    pd.DataFrame([
        {"conjunto": "train", "n": len(y_train), "rmse": rmse_train, "mae": mae_train, "r2": r2_train},
        {"conjunto": "test", "n": len(y_test), "rmse": rmse_test, "mae": mae_test, "r2": r2_test},
    ]).to_csv(out_tables / "paso07_train_vs_test.csv", index=False)

    predictions = pd.DataFrame({
        "indice_original": X_test.index,
        "SalePrice_real": y_test.to_numpy(),
        "SalePrice_predicho": y_pred_test,
    })
    predictions["error"] = predictions["SalePrice_real"] - predictions["SalePrice_predicho"]
    predictions["error_absoluto"] = predictions["error"].abs()
    predictions.sort_values("indice_original").to_csv(out_tables / "paso07_predicciones_test.csv", index=False)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(out_tables / "paso07_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    limits = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    plt.plot(limits, limits, linestyle="--", label="Predicción ideal")
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("Árbol de decisión - Real vs. Predicho")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figures / "paso07_real_vs_predicho.png", dpi=150)
    plt.close()

    errors = y_test.to_numpy() - y_pred_test
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, edgecolor="black", alpha=0.75)
    plt.axvline(0, linestyle="--", label="Error cero")
    plt.xlabel("SalePrice real - SalePrice predicho")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de errores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figures / "paso07_distribucion_errores.png", dpi=150)
    plt.close()

    top_importance = importance.head(15).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top_importance["feature"], top_importance["importance"])
    plt.xlabel("Importancia")
    plt.ylabel("Feature transformada")
    plt.title("Top 15 importancias de variables")
    plt.tight_layout()
    plt.savefig(out_figures / "paso07_top_feature_importance.png", dpi=150)
    plt.close()

    print(f"Train/test: {len(X_train)}/{len(X_test)}")
    print(f"Predictores originales: {X.shape[1]}")
    print(f"Features después de encoding: {len(feature_names)}")
    print(f"MAE test: {mae_test}")
    print(f"RMSE test: {rmse_test}")
    print(f"R2 test: {r2_test}")
    print(f"RMSE train: {rmse_train}")
    print(f"Complejidad: nodes={model.tree_.node_count}, depth={model.tree_.max_depth}, leaves={model.get_n_leaves()}")
    print("Top 10 importancias:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
