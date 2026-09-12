# Guía paso a paso - Actividad 2

## Paso 01 - Preparación del proyecto y validación del dataset

### Objetivo
Preparar una estructura reproducible, crear un entorno virtual y validar el dataset sin modificarlo ni entrenar modelos.

### Dataset y ubicación
Se utiliza `housing_train.csv` del [USA Housing Dataset de Kaggle](https://www.kaggle.com/gpandi007/usa-housing-dataset). La copia local debe ubicarse en `data/housing_train.csv`.

### Resultados
La ejecución de `src/paso01_validar_dataset.py` produjo:

- Registros: 1460.
- Columnas: 81.
- Target `SalePrice`: presente.
- Duplicados completos: 0.
- SHA-256: `ed142e0a97bd49fc3b46c930209c55750fc4e806f5430c7d7409cb4cf25149cb`.

El número de registros coincide con la referencia esperada.

### Entorno y dependencias
El entorno virtual es `.venv`, con Python 3.13.9 y pip 26.2.1. Las dependencias definidas son `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `jupyter`, `ipykernel`, `xgboost`, `scipy`, `joblib`, `tensorflow` y `gymnasium`; la instalación terminó correctamente.

### Decisiones de Git
Se versionan documentación, configuración y script. El dataset original, `.venv/` y los logs quedan excluidos mediante `.gitignore`.
## Paso 02 - Inspección estructural del dataset

### Objetivo

Inspeccionar y documentar la estructura del dataset para orientar el tratamiento posterior, sin imputar, eliminar columnas, aplicar encoding ni entrenar modelos.

### Resultados estructurales

El dataset tiene shape `(1460, 81)`: 38 variables numéricas y 43 categóricas. Entre las numéricas se encuentran `LotArea`, `OverallQual`, `YearBuilt`, `GrLivArea` y `GarageArea`. Entre las categóricas se encuentran `MSZoning`, `Neighborhood`, `HouseStyle`, `Exterior1st` y `KitchenQual`.

Las cinco categóricas con mayor cardinalidad son `Neighborhood` (25), `Exterior2nd` (16), `Exterior1st` (15), `Condition1` (9) y `SaleType` (9). Hay 19 columnas con valores faltantes; las primeras señales se observan en `PoolQC` (99.52%), `MiscFeature` (96.30%), `Alley` (93.77%), `Fence` (80.75%) y `MasVnrType` (59.73%). Esto es una descripción inicial, no una decisión de imputación.

`SalePrice` es el target de regresión: mínimo 34900, media 180921.20, mediana 163000 y máximo 755000. Como la media supera a la mediana por 17921.20 y el histograma muestra una cola hacia valores altos, la distribución parece sesgada a la derecha. `Id` se identifica preliminarmente como identificador y no como predictor, porque es único y no representa la magnitud del inmueble.

### Respuesta académica equivalente a la Figura 1

De manera general, el dataset combina variables numéricas de tamaño, calidad y antigüedad con variables categóricas de zona, estilo, barrio y características físicas. Presenta categóricas de baja y alta cardinalidad y valores faltantes en varias variables, por lo que requiere un tratamiento mixto antes del modelado. `SalePrice` es el objetivo de regresión e `Id` es un identificador. En Python/scikit-learn, las variables categóricas deberán codificarse antes de entrenar modelos; en este paso todavía no se aplica encoding.
