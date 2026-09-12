"""MLP de regresión con Keras, preprocesamiento y target sin leakage."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["str", "category"]).columns.tolist()
    numeric_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    categorical_pipeline = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    from sklearn.pipeline import Pipeline

    return ColumnTransformer([
        ("num", Pipeline(numeric_pipeline), numeric),
        ("cat", Pipeline(categorical_pipeline), categorical),
    ])


def calculate_metrics(y_true: pd.Series, prediction: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, prediction)),
        float(np.sqrt(mean_squared_error(y_true, prediction))),
        float(r2_score(y_true, prediction)),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # El preprocesador y el escalador del target se ajustan únicamente con train.
    preprocessor = make_preprocessor(X_train)
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_ready.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    start = time.perf_counter()
    history = model.fit(
        X_train_ready,
        y_train_scaled,
        epochs=500,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=0,
    )
    train_seconds = time.perf_counter() - start

    pred_train_scaled = model.predict(X_train_ready, verbose=0).ravel()
    pred_test_scaled = model.predict(X_test_ready, verbose=0).ravel()
    pred_train = y_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).ravel()
    pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
    mae_train, rmse_train, r2_train = calculate_metrics(y_train, pred_train)
    mae_test, rmse_test, r2_test = calculate_metrics(y_test, pred_test)

    metrics_dir = root / "outputs" / "metrics"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    for directory in (metrics_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    effective_epochs = len(history.history["loss"])
    architecture = "Dense(128)-Dense(64)-Dense(32)-Dense(1)"
    pd.DataFrame([{
        "modelo": "MLPRegressor_Keras", "implementacion": "TensorFlow/Keras", "arquitectura": architecture,
        "activacion": "ReLU en capas ocultas; linear en salida", "solver_optimizador": "Adam",
        "max_iter_epochs": 500, "iteraciones_epochs_reales": effective_epochs, "batch_size": 32,
        "early_stopping": "val_loss, patience=30, restore_best_weights=True", "n_train": len(X_train), "n_test": len(X_test),
        "mae_train": mae_train, "rmse_train": rmse_train, "r2_train": r2_train,
        "mae_test": mae_test, "rmse_test": rmse_test, "r2_test": r2_test,
        "train_seconds": train_seconds, "random_state": 42,
    }]).to_csv(metrics_dir / "paso11_mlp_regresion_metricas.csv", index=False)

    history_table = pd.DataFrame({"iteracion": np.arange(1, effective_epochs + 1), "loss": history.history["loss"]})
    if "val_loss" in history.history:
        history_table["validation_score"] = history.history["val_loss"]
    history_table.to_csv(tables_dir / "paso11_historial_entrenamiento.csv", index=False)

    predictions = pd.DataFrame({"indice_original": X_test.index, "SalePrice_real": y_test.to_numpy(), "SalePrice_predicho": pred_test})
    predictions["error"] = predictions["SalePrice_real"] - predictions["SalePrice_predicho"]
    predictions["error_absoluto"] = predictions["error"].abs()
    predictions.sort_values("indice_original").to_csv(tables_dir / "paso11_predicciones_test.csv", index=False)

    previous = pd.DataFrame([
        {"modelo": "arbol_sin_poda", "mae_test": 27494.58, "rmse_test": 42453.07, "r2_test": 0.7650, "rmse_train": 0.00},
        {"modelo": "arbol_podado", "mae_test": 25776.51, "rmse_test": 40386.27, "r2_test": 0.7874, "rmse_train": 19979.70},
        {"modelo": "random_forest", "mae_test": 17410.11, "rmse_test": 28929.45, "r2_test": 0.8909, "rmse_train": 11063.98},
        {"modelo": "xgboost", "mae_test": 15852.43, "rmse_test": 25564.80, "r2_test": 0.9148, "rmse_train": 11960.51},
        {"modelo": "mlp", "mae_test": mae_test, "rmse_test": rmse_test, "r2_test": r2_test, "rmse_train": rmse_train},
    ])
    previous["gap_rmse_train_test"] = previous["rmse_test"] - previous["rmse_train"]
    previous.to_csv(tables_dir / "paso11_comparacion_regresion_completa.csv", index=False)

    ranking = previous.sort_values("rmse_test").reset_index(drop=True)
    ranking.insert(0, "ranking", np.arange(1, len(ranking) + 1))
    ranking[["ranking", "modelo", "rmse_test", "mae_test", "r2_test"]].to_csv(tables_dir / "paso11_ranking_regresion.csv", index=False)

    comparison_mlp = pd.DataFrame([
        {"comparacion": "mlp_vs_xgboost", "delta_rmse": rmse_test - 25564.80, "delta_rmse_pct": (rmse_test - 25564.80) / 25564.80 * 100},
        {"comparacion": "mlp_vs_random_forest", "delta_rmse": rmse_test - 28929.45, "delta_rmse_pct": (rmse_test - 28929.45) / 28929.45 * 100},
    ])
    comparison_mlp.to_csv(tables_dir / "paso11_comparacion_mlp.csv", index=False)

    pd.DataFrame([{
        "implementacion_mlp": "TensorFlow/Keras", "arquitectura": architecture,
        "rmse_tree_unpruned": 42453.07, "rmse_tree_pruned": 40386.27, "rmse_random_forest": 28929.45, "rmse_xgboost": 25564.80,
        "rmse_mlp": rmse_test, "mae_mlp": mae_test, "r2_mlp": r2_test, "rmse_train_mlp": rmse_train,
        "gap_train_test_mlp": rmse_test - rmse_train, "delta_mlp_vs_xgb_abs": rmse_test - 25564.80, "delta_mlp_vs_xgb_pct": (rmse_test - 25564.80) / 25564.80 * 100,
        "delta_mlp_vs_rf_abs": rmse_test - 28929.45, "delta_mlp_vs_rf_pct": (rmse_test - 28929.45) / 28929.45 * 100,
        "train_seconds": train_seconds, "iterations_or_epochs": effective_epochs,
    }]).to_csv(metrics_dir / "paso11_resumen.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(history_table["iteracion"], history_table["loss"], label="loss train")
    if "validation_score" in history_table:
        plt.plot(history_table["iteracion"], history_table["validation_score"], label="loss validation")
    plt.xlabel("Época")
    plt.ylabel("Pérdida MSE en target escalado")
    plt.title("Curva de entrenamiento de la MLP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "paso11_curva_entrenamiento.png", dpi=150)
    plt.close()

    limits = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred_test, alpha=0.7)
    plt.plot(limits, limits, linestyle="--", label="Predicción ideal")
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("MLP - Real vs. Predicho")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "paso11_real_vs_predicho.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(previous["modelo"], previous["rmse_test"])
    plt.ylabel("RMSE test")
    plt.title("Comparación de RMSE de modelos de regresión")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "paso11_comparacion_rmse_modelos.png", dpi=150)
    plt.close()

    print(f"Implementación: TensorFlow/Keras {tf.__version__}")
    print(f"Arquitectura: {architecture}")
    print(f"Epochs reales: {effective_epochs}")
    print(f"Train/test: {len(X_train)}/{len(X_test)}")
    print(f"MAE train: {mae_train}; RMSE train: {rmse_train}; R2 train: {r2_train}")
    print(f"MAE test: {mae_test}; RMSE test: {rmse_test}; R2 test: {r2_test}")
    print(f"Gap RMSE train-test: {rmse_test - rmse_train}")
    print(ranking[["ranking", "modelo", "rmse_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
