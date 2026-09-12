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
