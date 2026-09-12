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
## Paso 03 - Estadística descriptiva de variables numéricas

### Objetivo

Describir las variables numéricas del dataset mediante medidas de tendencia central, dispersión, asimetría y concentración de ceros, sin eliminar ni imputar variables.

Se analizaron 38 variables numéricas en 1460 observaciones. La tabla completa incluye `count`, missing, mínimo, Q1, mediana, media, Q3, máximo, desviación estándar, IQR, skewness y porcentaje de ceros.

### SalePrice

`SalePrice` presenta mínimo 34900, Q1 129975, mediana 163000, media 180921.20, Q3 214000, máximo 755000, desviación estándar 79442.50 y skewness 1.88. La media supera la mediana en 17921.20 y la asimetría es positiva, por lo que la distribución está sesgada a la derecha.

### Ceros, sesgo y variabilidad

Hay 13 variables numéricas con mediana cero. Las mayores proporciones de ceros corresponden a `PoolArea` (99.52%), `3SsnPorch` (98.36%), `LowQualFinSF` (98.22%), `MiscVal` (96.44%), `BsmtHalfBath` (94.38%), `ScreenPorch` (92.05%), `BsmtFinSF2` (88.56%) y `EnclosedPorch` (85.75%). Esto sugiere características ausentes en muchas viviendas o variables con concentración estructural en cero.

Las variables más sesgadas por valor absoluto son `MiscVal`, `PoolArea`, `LotArea`, `3SsnPorch`, `LowQualFinSF`, `KitchenAbvGr`, `BsmtFinSF2`, `ScreenPorch`, `BsmtHalfBath` y `EnclosedPorch`. Las candidatas preliminares de baja variabilidad, usando una concentración dominante superior al 95% o un porcentaje de ceros de al menos 95%, son `PoolArea`, `MiscVal`, `LowQualFinSF`, `3SsnPorch` y `KitchenAbvGr`. Pueden considerarse transformaciones, indicadores binarios o una exclusión posterior, pero por ahora la decisión es `revisar`.

### Respuesta académica

Si se tienen 1460 observaciones, constituyen una muestra suficiente para el laboratorio académico. Los descriptivos muestran variables con fuerte dispersión y sesgo; `SalePrice` está sesgada a la derecha; varias características tienen alta concentración en cero; y algunas podrían aportar poca información por baja variabilidad.

Sí, podríamos eliminar algunas variables, pero existen únicamente candidatas preliminares y todavía no se elimina ninguna. `Id` se excluye del modelado por ser identificador. Las variables con casi todos sus valores en cero podrían eliminarse o transformarse en indicadores binarios. La decisión definitiva debe considerar también correlaciones, missing y desempeño predictivo, además de completar la preparación del modelado.

La guía del profesor coincide aproximadamente con el dataset: `SalePrice` tiene mínimo 34900, mediana 163000, media 180921.20 y máximo 755000; `LotFrontage`, `MasVnrArea` y `GarageYrBlt` tienen respectivamente 259, 8 y 81 missing. No se forzaron coincidencias ni se alteraron los datos.
## Paso 04 - Variables categóricas y frecuencias

### Objetivo y resultados

Se identificaron automáticamente 43 variables categóricas y se calcularon sus frecuencias absolutas y relativas, cardinalidad y missing, sin imputar, eliminar columnas ni aplicar encoding.

Las variables de mayor cardinalidad son `Neighborhood` (25 categorías), `Exterior2nd` (16) y `Exterior1st` (15), seguidas por `Condition1` y `SaleType` (9 cada una). `Neighborhood` está encabezada por `NAmes` (225), `CollgCr` (150), `OldTown` (113), `Edwards` (100) y `Somerst` (86). `Street` tiene `Pave` (1454) y `Grvl` (6).

Se identificaron 11 variables dominadas por una categoría con al menos 90%; entre las más marcadas están `Utilities`/`AllPub` (99.93%), `Street`/`Pave` (99.59%), `Condition2`/`Norm` (98.97%), `RoofMatl`/`CompShg` (98.22%) y `Heating`/`GasA` (97.81%). También se encontraron categorías raras en 31 variables; se identificaron, pero todavía no se agrupan.

Hay 16 variables categóricas con missing. Los porcentajes más altos están en `PoolQC` (99.52%), `MiscFeature` (96.30%), `Alley` (93.77%), `Fence` (80.75%) y `MasVnrType` (59.73%). Se marcaron como `por_revisar`, porque un missing puede ser estructural o un dato faltante y no se debe afirmar su significado sin revisar la semántica de cada variable.

### Respuesta académica

Las variables categóricas muestran que algunas tienen pocas categorías, mientras otras, como `Neighborhood`, tienen muchas. Algunas están fuertemente dominadas por una sola categoría y otras presentan niveles muy poco frecuentes. Varias tienen missing que puede ser estructural. Estas características condicionarán el preprocesamiento: las categóricas deberán codificarse antes de entrenar modelos en Python/scikit-learn y OneHotEncoder puede aumentar considerablemente la dimensionalidad cuando hay alta cardinalidad.

No se agrupan categorías raras ni se eliminan variables todavía: primero debe revisarse su significado y su aporte junto con missing, correlaciones y desempeño predictivo. La comparación con la referencia del profesor coincide: `Neighborhood` tiene 25 categorías y `Street` tiene 2.
## Paso 05 - Matriz de correlaciones y redundancia numérica

### Objetivo y metodología

Se calculó la matriz de correlaciones de Pearson con las 37 variables numéricas interpretadas como predictoras, excluyendo `Id` y manteniendo `SalePrice` como objetivo. No se imputaron valores: pandas calculó cada correlación con los datos disponibles por pares.

### Variables más correlacionadas con SalePrice

Las diez variables con mayor correlación absoluta con `SalePrice` fueron `OverallQual` (0.791), `GrLivArea` (0.709), `GarageCars` (0.640), `GarageArea` (0.623), `TotalBsmtSF` (0.614), `1stFlrSF` (0.606), `FullBath` (0.561), `TotRmsAbvGrd` (0.534), `YearBuilt` (0.523) y `YearRemodAdd` (0.507). En particular, `OverallQual` y `GrLivArea` presentan relaciones positivas fuertes con el precio.

### Pares de predictores fuertemente correlacionados

Se encontraron cuatro pares con `|r| >= 0.70`: `GarageCars`-`GarageArea` (0.882), `YearBuilt`-`GarageYrBlt` (0.826), `GrLivArea`-`TotRmsAbvGrd` (0.825) y `TotalBsmtSF`-`1stFlrSF` (0.820). Estos pares sugieren redundancia potencial y fueron marcados para revisión preliminar.

### Respuesta académica a la Figura 3

Las variables más correlacionadas con `SalePrice` son principalmente indicadores de calidad, superficie, garaje, sótano, baños y antigüedad. Separadamente, los pares de predictores con mayor correlación entre sí describen aspectos relacionados del mismo inmueble y pueden introducir multicolinealidad o información redundante.

Sí, se puede considerar eliminar alguna columna o simplificar grupos de variables, pero todavía no se elimina ninguna solo por correlación. Los árboles y random forest suelen tolerar mejor predictores correlacionados, mientras que los modelos lineales y otros sensibles a redundancia pueden beneficiarse de simplificación. La decisión definitiva requiere completar el tratamiento de missing, preparar el pipeline y comparar modelos.

La correlación no implica causalidad y puede estar afectada por relaciones no lineales, valores atípicos y datos faltantes. Por ello, estos resultados son una guía exploratoria: `GarageCars`/`GarageArea`, `TotalBsmtSF`/`1stFlrSF`, `YearBuilt`/`GarageYrBlt` y `GrLivArea`/`TotRmsAbvGrd` son candidatos preliminares a revisar, no variables eliminadas.

La comparación conceptual coincide con la referencia del profesor: `OverallQual` y `GrLivArea` aparecen entre las variables más relacionadas con `SalePrice`, y los cuatro pares de referencia muestran correlaciones altas o relevantes.
## Paso 06 - Tratamiento de valores faltantes

### Estrategia aplicada

Se detectaron 19 columnas con 7829 valores missing. Se distinguió entre missing real, cuando el dato debería existir pero no fue registrado, y missing estructural, cuando la característica no existe o no aplica. La imputación se aplicó únicamente sobre `df_original.copy()` y el CSV original permaneció intacto.

Para el missing real numérico `LotFrontage` se utilizó la mediana calculada del dataset (69.0), por ser robusta frente a outliers. `Electrical`, con un solo missing y sin significado estructural, se imputó con su moda real (`SBrkr`).

Para variables numéricas estructurales, `MasVnrArea` se imputó con 0: su mediana es 0 y 859 de los 864 casos comparables con `MasVnrType` missing tienen área 0. `GarageYrBlt` se imputó con 0 cuando faltaba; 0 no representa un año real, sino “sin garaje / no aplica”. No se imputó con `YearBuilt`.

Los missing estructurales de `Alley`, variables de sótano (`BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`), `FireplaceQu`, variables de garaje (`GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`), `PoolQC`, `Fence` y `MiscFeature` se reemplazaron por la categoría explícita `None`. En `MasVnrType` se eligió también `None`, porque la evidencia disponible sugiere ausencia de revestimiento. No se eliminaron `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType` ni `FireplaceQu`.

### Diagnóstico y consistencia

`MasVnrType` tiene 872 missing (59.73%); 859 coinciden con `MasVnrArea == 0` y 5 con `MasVnrArea > 0` entre los casos comparables. Los 8 casos restantes tienen `MasVnrArea` missing. La discrepancia frente a la referencia complementaria del profesor se conserva y se documenta, sin alterar los datos.

Las reglas estructurales no detectaron inconsistencias: `GarageType == None` fue coherente con `GarageYrBlt == 0` en 81 de 81 casos; `BsmtQual == None` fue compatible con las variables numéricas de sótano en 37 de 37; y `PoolQC == None` fue coherente con `PoolArea == 0` en 1453 de 1453. Después del tratamiento quedaron 0 valores missing y el dataset derivado conserva shape `(1460, 81)`.

### Respuesta académica

La mejor manera de llenar los valores faltantes depende de su significado. Para missing reales, los numéricos pueden imputarse con la mediana cuando sea adecuada y los categóricos con la moda si son pocos y no representan ausencia estructural. Para missing estructurales conviene crear la categoría explícita `None`; en variables numéricas asociadas a ausencia, usar 0 cuando tenga sentido semántico.

Imputar la moda indiscriminadamente sería incorrecto: asignar a una vivienda sin garaje el tipo de garaje más frecuente inventaría una característica inexistente. Por eso se conservaron las variables y se distinguió la ausencia de un dato no registrado. Todavía no se realiza encoding ni modelado.
## Paso 07 - Árbol de decisión para regresión

### Metodología

Se construyó un problema de regresión para predecir `SalePrice`, el precio de venta de cada vivienda. Se excluyeron `SalePrice` de los predictores e `Id` por ser un identificador, quedando 79 predictores originales. Se dividieron los datos antes de ajustar cualquier transformación: 1168 observaciones para train y 292 para test, con `random_state=42` y sin estratificación.

Esto evita data leakage: el modelo no usa `outputs/tables/housing_train_imputado.csv`, cuyos valores fueron calculados sobre todo el dataset. En su lugar, el pipeline lee `data/housing_train.csv`, ajusta la imputación únicamente sobre `X_train` y luego transforma test. Las variables numéricas usan mediana y las categóricas usan `None` seguido de `OneHotEncoder(handle_unknown="ignore")`; los árboles no manejan strings categóricos directamente en scikit-learn.

### Resultado del baseline

El `DecisionTreeRegressor(random_state=42)` sin `max_depth`, `min_samples_leaf` ni `ccp_alpha` obtuvo en test MAE 27494.58, RMSE 42453.07 y R² 0.7650. La referencia del profesor es RMSE 41002.2; la diferencia absoluta es 1450.87 (3.54%). Se documenta la diferencia sin alterar split, semilla, preprocesamiento ni hiperparámetros. Puede deberse a diferencias de implementación, tratamiento de categóricas, missing o versiones.

El árbol produjo 2245 nodos, profundidad máxima 24 y 1123 hojas. En train el RMSE fue 0.00, frente a 42453.07 en test; esta brecha constituye un indicio claro de posible sobreajuste en este baseline sin restricciones. Las importancias principales fueron `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `2ndFlrSF` y `BsmtFinSF1`. Las categóricas aparecen descompuestas en features por categoría tras el OneHotEncoder, por lo que sus importancias deben interpretarse con esa limitación.

### Respuesta académica

El árbol de decisión sin podar obtuvo un RMSE de 42453.07. La referencia proporcionada por el profesor es 41002.2; el valor del dataset es 1450.87 mayor, una diferencia de 3.54%, y no se forzó coincidencia.

Un árbol sin restricciones puede ajustarse fuertemente al conjunto de entrenamiento, por lo que la diferencia entre el error de entrenamiento y de prueba permite evaluar indicios de sobreajuste. En el siguiente paso se evaluará la poda y se comprobará empíricamente si simplificar el árbol mejora o empeora su capacidad de generalización.
## Paso 08 - Poda del árbol de regresión

### Metodología

La poda reduce ramas de un árbol para controlar su complejidad y disminuir el riesgo de sobreajuste. Se usó cost-complexity pruning: `ccp_alpha` penaliza la complejidad del árbol, de modo que valores mayores producen árboles más simples. Se recreó el split externo de los pasos anteriores (1168 train y 292 test), usando `data/housing_train.csv` y el mismo pipeline de imputación y OneHotEncoder.

El camino de poda se calculó solo sobre los datos de entrenamiento. Para elegir `ccp_alpha` se dividió `X_train` en train interno y validación interna, ambos con `random_state=42`; el test externo no participó en la selección. Se evaluaron 40 candidatos reproducibles y se eligió el alpha con menor RMSE de validación, con preferencia por el árbol más simple en caso de empate.

### Resultados

El RMSE baseline sin poda se reprodujo como 42453.07. El alpha seleccionado fue 6135101.075147216. El árbol podado obtuvo RMSE train 19979.70, RMSE test 40386.27, MAE test 25776.51 y R² test 0.7874.

La complejidad se redujo de 2245 a 127 nodos, de profundidad 24 a 9 y de 1123 a 64 hojas. El RMSE test disminuyó en 2066.81 (4.87%) respecto al baseline, por lo que en este experimento la poda mejoró la generalización. La brecha train-test pasó de 42453.07 a 20406.57, aunque el RMSE train sigue siendo menor que el de test.

Frente a la referencia del profesor, el baseline difiere en 1450.87 (3.54%) respecto a 41002.2 y el árbol podado difiere en 2540.98 (5.92%) respecto a 42927.25. En la referencia del profesor, 41002.2 pasa a 42927.25: la poda empeora aproximadamente 1925.05 (4.70%). No se reprodujo artificialmente ese comportamiento; se respetó el alpha seleccionado por validación interna en este dataset y pipeline.

### Respuesta académica

Una menor complejidad no garantiza automáticamente un menor error de test. Si el RMSE podado aumenta, la poda pudo ser demasiado agresiva o eliminar divisiones útiles, introduciendo mayor sesgo y posible underfitting. Si disminuye, como en nuestro resultado, la poda redujo parte del sobreajuste y mejoró la generalización. La decisión debe basarse en validación y test, no en asumir que podar siempre mejora.

El árbol sin poda obtuvo RMSE 42453.07 y el podado 40386.27, por lo que la poda mejoró en este experimento. La referencia del profesor muestra el caso contrario (41002.2 a 42927.25); esa diferencia confirma que el efecto depende de los datos, el preprocesamiento, la selección de alpha y la evaluación. Todavía no se ha ejecutado Random Forest ni se avanza al Paso 09.
