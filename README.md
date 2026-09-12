# Actividad 2 - Aprendizaje Automático

Preparación reproducible y validación del dataset USA Housing para regresión y posterior clasificación por grupos de precio.

Dataset: `housing_train.csv`, referencia [USA Housing Dataset en Kaggle](https://www.kaggle.com/gpandi007/usa-housing-dataset). La copia local debe estar en `data/housing_train.csv` y no se versiona.

## Estructura

`data/` dataset; `docs/` documentación; `notebooks/` notebooks; `src/` scripts; `outputs/` resultados, modelos y logs.

## Paso 01 en PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
& .\.venv\Scripts\python.exe src\paso01_validar_dataset.py
```
