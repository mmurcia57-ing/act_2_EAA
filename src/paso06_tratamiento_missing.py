"""Tratamiento reproducible de valores faltantes sin modificar el CSV original."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df_original = pd.read_csv(root / "data" / "housing_train.csv")
    df = df_original.copy()
    tables = root / "outputs" / "tables"
    metrics = root / "outputs" / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    missing_before = pd.DataFrame({
        "variable": df_original.columns,
        "dtype": [str(dtype) for dtype in df_original.dtypes],
        "missing_count": df_original.isna().sum().values,
        "missing_pct": (df_original.isna().mean().values * 100),
    })
    missing_before = missing_before[missing_before["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    missing_before.to_csv(tables / "paso06_missing_antes.csv", index=False)

    masvnr_missing = df_original["MasVnrType"].isna()
    masvnr_area_zero = int((masvnr_missing & (df_original["MasVnrArea"] == 0)).sum())
    masvnr_area_gt_zero = int((masvnr_missing & (df_original["MasVnrArea"] > 0)).sum())
    masvnr_known_area_cases = masvnr_area_zero + masvnr_area_gt_zero
    masvnr_none_ratio = masvnr_area_zero / masvnr_known_area_cases if masvnr_known_area_cases else 0
    masvnr_strategy = "None" if masvnr_none_ratio >= 0.90 else "Unknown"
    pd.DataFrame([
        {"metrica": "missing_masvnrtype", "valor": int(masvnr_missing.sum())},
        {"metrica": "missing_pct_masvnrtype", "valor": float(masvnr_missing.mean() * 100)},
        {"metrica": "missing_masvnrarea", "valor": int(df_original["MasVnrArea"].isna().sum())},
        {"metrica": "missing_type_area_zero", "valor": masvnr_area_zero},
        {"metrica": "missing_type_area_gt_zero", "valor": masvnr_area_gt_zero},
    ]).to_csv(tables / "paso06_masvnrtype_diagnostico.csv", index=False)

    structural_categorical = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "PoolQC", "Fence", "MiscFeature",
    ]
    strategies: dict[str, dict[str, object]] = {}
    for column in missing_before["variable"]:
        missing_count = int(df_original[column].isna().sum())
        missing_pct = float(df_original[column].isna().mean() * 100)
        if column == "LotFrontage":
            value = float(df_original[column].median())
            strategies[column] = {"tipo_missing": "dato_faltante_real", "estrategia": "mediana", "valor": value, "justificacion": "variable numérica real; la mediana es robusta frente a outliers"}
        elif column == "MasVnrArea":
            value = 0
            strategies[column] = {"tipo_missing": "estructural_posible", "estrategia": "cero", "valor": value, "justificacion": "la mediana es 0 y los ceros representan ausencia de revestimiento"}
        elif column == "GarageYrBlt":
            value = 0
            strategies[column] = {"tipo_missing": "estructural_posible", "estrategia": "cero", "valor": value, "justificacion": "0 representa sin garaje/no aplica; no es un año real"}
        elif column == "Electrical":
            value = df_original[column].mode(dropna=True).iloc[0]
            strategies[column] = {"tipo_missing": "dato_faltante_real", "estrategia": "moda", "valor": value, "justificacion": "muy pocos casos y no representa ausencia estructural"}
        elif column == "MasVnrType":
            strategies[column] = {"tipo_missing": "estructural_posible", "estrategia": masvnr_strategy.lower(), "valor": masvnr_strategy, "justificacion": f"{masvnr_area_zero} de {masvnr_known_area_cases} casos comparables tienen MasVnrArea igual a 0"}
        elif column in structural_categorical:
            strategies[column] = {"tipo_missing": "estructural_posible", "estrategia": "none", "valor": "None", "justificacion": "NA puede representar que la característica no existe o no aplica"}
        else:
            strategies[column] = {"tipo_missing": "por_revisar", "estrategia": "revisar", "valor": "", "justificacion": "requiere revisión semántica antes de imputar"}

    strategy_rows = []
    for column in missing_before["variable"]:
        strategy = strategies[column]
        strategy_rows.append({
            "variable": column,
            "tipo_variable": "numerica" if pd.api.types.is_numeric_dtype(df_original[column]) else "categorica",
            "missing_count": int(df_original[column].isna().sum()),
            "missing_pct": float(df_original[column].isna().mean() * 100),
            "tipo_missing": strategy["tipo_missing"],
            "estrategia": strategy["estrategia"],
            "valor_imputacion": strategy["valor"],
            "justificacion": strategy["justificacion"],
        })
    strategy_table = pd.DataFrame(strategy_rows)
    strategy_table.to_csv(tables / "paso06_estrategia_missing.csv", index=False)

    for column, strategy in strategies.items():
        if strategy["estrategia"] == "mediana":
            df[column] = df[column].fillna(strategy["valor"])
        elif strategy["estrategia"] == "moda":
            df[column] = df[column].fillna(strategy["valor"])
        elif strategy["estrategia"] in ("cero", "none", "unknown"):
            df[column] = df[column].fillna(strategy["valor"])

    consistency_rows = []
    garage_none = df["GarageType"] == "None"
    garage_consistent = garage_none & (df["GarageYrBlt"] == 0)
    consistency_rows.append({"regla": "GarageType == None implica GarageYrBlt == 0", "casos_evaluados": int(garage_none.sum()), "casos_consistentes": int(garage_consistent.sum()), "casos_inconsistentes": int((garage_none & ~garage_consistent).sum()), "comentario": "solo se reportan inconsistencias; no se corrigen automáticamente"})
    basement_none = df["BsmtQual"] == "None"
    basement_numeric = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
    basement_consistent = basement_none & (df[basement_numeric].fillna(0).sum(axis=1) == 0)
    consistency_rows.append({"regla": "BsmtQual == None compatible con variables numéricas de sótano en 0", "casos_evaluados": int(basement_none.sum()), "casos_consistentes": int(basement_consistent.sum()), "casos_inconsistentes": int((basement_none & ~basement_consistent).sum()), "comentario": "se revisan variables numéricas asociadas; no se corrigen automáticamente"})
    pool_none = df["PoolQC"] == "None"
    pool_consistent = pool_none & (df["PoolArea"] == 0)
    consistency_rows.append({"regla": "PoolQC == None implica PoolArea == 0", "casos_evaluados": int(pool_none.sum()), "casos_consistentes": int(pool_consistent.sum()), "casos_inconsistentes": int((pool_none & ~pool_consistent).sum()), "comentario": "solo se reportan inconsistencias; no se corrigen automáticamente"})
    pd.DataFrame(consistency_rows).to_csv(tables / "paso06_consistencia_estructural.csv", index=False)

    total_missing_after = int(df.isna().sum().sum())
    if total_missing_after != 0:
        remaining = df.columns[df.isna().any()].tolist()
        raise RuntimeError(f"Quedan missing después de imputar: {remaining}")
    df.to_csv(tables / "housing_train_imputado.csv", index=False)
    # La categoría explícita "None" es un token NA por defecto en pandas.
    # Se valida el CSV derivado conservando esos tokens como texto.
    roundtrip = pd.read_csv(tables / "housing_train_imputado.csv", keep_default_na=False)
    if int(roundtrip.isna().sum().sum()) != 0 or roundtrip.shape != df.shape:
        raise RuntimeError("El CSV derivado no conserva shape o ausencia total de missing")

    comparison = strategy_table[["variable", "missing_count", "estrategia", "valor_imputacion"]].rename(columns={"missing_count": "missing_antes"})
    comparison["missing_despues"] = [int(df[column].isna().sum()) for column in comparison["variable"]]
    comparison = comparison[["variable", "missing_antes", "missing_despues", "estrategia", "valor_imputacion"]]
    comparison.to_csv(tables / "paso06_missing_antes_despues.csv", index=False)

    median_cells = sum(int(df_original[column].isna().sum()) for column, value in strategies.items() if value["estrategia"] == "mediana")
    mode_cells = sum(int(df_original[column].isna().sum()) for column, value in strategies.items() if value["estrategia"] == "moda")
    zero_cells = sum(int(df_original[column].isna().sum()) for column, value in strategies.items() if value["estrategia"] == "cero")
    none_cells = sum(int(df_original[column].isna().sum()) for column, value in strategies.items() if value["estrategia"] == "none")
    unknown_cells = sum(int(df_original[column].isna().sum()) for column, value in strategies.items() if value["estrategia"] == "unknown")
    pd.DataFrame([{
        "filas": len(df), "columnas": len(df.columns),
        "columnas_con_missing_antes": len(missing_before), "total_missing_antes": int(df_original.isna().sum().sum()),
        "columnas_con_missing_despues": int(df.isna().any().sum()), "total_missing_despues": total_missing_after,
        "n_imputaciones_mediana": median_cells, "n_imputaciones_moda": mode_cells, "n_imputaciones_cero": zero_cells,
        "n_imputaciones_none": none_cells, "n_imputaciones_unknown": unknown_cells,
    }]).to_csv(metrics / "paso06_resumen.csv", index=False)

    comparison_reference = pd.DataFrame([
        {"variable": "LotFrontage", "missing_dataset": int(df_original["LotFrontage"].isna().sum()), "estrategia_dataset": "mediana", "referencia_profesor": "259 missing; mediana", "coincide": True, "comentario": "coincide"},
        {"variable": "MasVnrArea", "missing_dataset": int(df_original["MasVnrArea"].isna().sum()), "estrategia_dataset": "0", "referencia_profesor": "8 missing; tratamiento cercano a 0", "coincide": int(df_original["MasVnrArea"].isna().sum()) == 8, "comentario": "el tratamiento coincide, pero el conteo real difiere de la referencia"},
        {"variable": "GarageYrBlt", "missing_dataset": int(df_original["GarageYrBlt"].isna().sum()), "estrategia_dataset": "0 estructural", "referencia_profesor": "81 missing; estructural", "coincide": True, "comentario": "coincide"},
        {"variable": "BsmtQual", "missing_dataset": int(df_original["BsmtQual"].isna().sum()), "estrategia_dataset": "None", "referencia_profesor": "missing estructural", "coincide": True, "comentario": "coincide"},
        {"variable": "GarageType", "missing_dataset": int(df_original["GarageType"].isna().sum()), "estrategia_dataset": "None", "referencia_profesor": "missing estructural", "coincide": True, "comentario": "coincide"},
        {"variable": "Electrical", "missing_dataset": int(df_original["Electrical"].isna().sum()), "estrategia_dataset": "moda", "referencia_profesor": "muy pocos casos; moda", "coincide": True, "comentario": "coincide"},
        {"variable": "MasVnrType", "missing_dataset": int(df_original["MasVnrType"].isna().sum()), "estrategia_dataset": masvnr_strategy, "referencia_profesor": "la guía complementaria reporta pocos missing", "coincide": False, "comentario": "se conserva el comportamiento observado en el dataset descargado y no se alteran los datos para reproducir artificialmente la referencia"},
    ])
    comparison_reference.to_csv(tables / "paso06_comparacion_referencia_profesor.csv", index=False)

    print(f"Columnas con missing antes: {len(missing_before)}")
    print(f"Total missing antes: {int(df_original.isna().sum().sum())}")
    print(f"Columnas con missing después: {int(df.isna().any().sum())}")
    print(f"Total missing después: {total_missing_after}")
    print(f"MasVnrType: {int(masvnr_missing.sum())} ({masvnr_missing.mean() * 100:.2f}%), area=0: {masvnr_area_zero}, area>0: {masvnr_area_gt_zero}, estrategia: {masvnr_strategy}")
    print(pd.DataFrame(consistency_rows).to_string(index=False))


if __name__ == "__main__":
    main()
