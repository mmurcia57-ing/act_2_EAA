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
## Paso 02

La inspección estructural fue ejecutada. Sus tablas, métricas y el histograma se encuentran en `outputs/tables/`, `outputs/metrics/` y `outputs/figures/`. Todavía no se ha realizado imputación, encoding ni modelado.
## Paso 03

Se completó la estadística descriptiva de variables numéricas. Aún no se han eliminado variables; la siguiente decisión dependerá del análisis de correlaciones y missing. Los artefactos están en `outputs/tables/`, `outputs/metrics/` y `outputs/figures/`.
## Paso 04

Se completó el análisis categórico y de frecuencias. Aún no se realiza encoding ni imputación de missing.
## Paso 05

Se calcularon las correlaciones numéricas y se documentó la redundancia preliminar. Todavía no se eliminan variables; el siguiente paso será el tratamiento de missing.
## Paso 06

Se trataron los missing y está disponible el dataset derivado imputado en `outputs/tables/housing_train_imputado.csv`. El dataset original fue preservado; todavía no se ha realizado encoding ni modelado.
## Paso 07

Se entrenó el primer modelo supervisado: un árbol de regresión baseline con preprocesamiento dentro de un pipeline sin leakage. La poda aún no se ha realizado.
## Paso 08

Se evaluó la poda del árbol de regresión y se comparó contra el baseline y la referencia del profesor. Random Forest todavía no se ha ejecutado.
## Paso 09

Se evaluó Random Forest para regresión y se completó la comparación con los árboles. Boosting todavía no se ha ejecutado.
