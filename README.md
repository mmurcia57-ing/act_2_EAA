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
## Paso 10

Se evaluó XGBoost para regresión y se completó la comparación con árboles y Random Forest. MLP todavía está pendiente.
## Paso 11

Se evaluó una MLP de regresión y se completó la comparación de los cinco modelos de regresión. La clasificación aún no se ha iniciado.
## Paso 12

El bloque de regresión fue cerrado y el ranking final está disponible en `outputs/tables/`. El mejor modelo fue XGBoost. El siguiente bloque será clasificación.
## Paso 13

Se inició el bloque de clasificación: se crearon los grupos de precio y se definió el split estratificado. Todavía no se ha entrenado ningún clasificador.
## Paso 14

Se evaluó el árbol de clasificación y se generó la matriz de confusión. El siguiente modelo será Random Forest.
## Paso 15

Se evaluó Random Forest para clasificación y se comparó con el árbol. El siguiente clasificador será Boosting.

## Paso 16

Se ejecutó XGBoost para clasificación multiclase y se comparó con árbol y Random Forest. El siguiente paso será SVM.

## Paso 17

Se evaluaron SVM lineal y RBF con C=0.1, 1 y 10. El siguiente paso será el cierre comparativo de clasificación.

## Paso 18

Se completó el cierre comparativo de clasificación. XGBoost obtuvo la mejor accuracy; SVM lineal con C=1 obtuvo la mejor balanced accuracy y macro F1. El siguiente bloque será clustering.

## Paso 19

Se ejecutó la selección de K mediante codo y silhouette usando variables numéricas escaladas. El clustering definitivo aún no se ha ejecutado; el siguiente paso será K-Means.

## Paso 20

Se completó el K-Means definitivo con K=2 y se caracterizaron los perfiles de vivienda, diferencias entre centroides y distribución descriptiva de `SalePrice`. El siguiente paso será clustering jerárquico.
