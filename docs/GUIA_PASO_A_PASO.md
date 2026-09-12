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
## Paso 09 - Random Forest para regresión

### Metodología y modelo

Random Forest combina múltiples árboles de decisión. Cada árbol se entrena sobre una muestra bootstrap y, en cada división, considera un subconjunto aleatorio de variables. Esto reduce la correlación entre árboles y, al promediar sus predicciones, suele reducir la varianza frente a un árbol individual. No garantiza ser siempre mejor, pero normalmente generaliza de forma más estable.

Se mantuvo el split externo de 1168 train y 292 test, con `random_state=42`, se excluyeron `SalePrice` e `Id`, y se conservaron los demás 79 predictores originales. El preprocesamiento se ajustó solo sobre `X_train` dentro del pipeline: mediana para numéricas y categoría `None` más `OneHotEncoder(handle_unknown="ignore")` para categóricas. Se obtuvieron 301 features transformadas.

La configuración fue `RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)`, sin búsqueda agresiva de hiperparámetros. El entrenamiento tomó 12.96 segundos.

### Resultados y comparación

En train, Random Forest obtuvo MAE 6471.35, RMSE 11063.98 y R² 0.9795. En test obtuvo MAE 17410.11, RMSE 28929.45 y R² 0.8909, con una brecha RMSE train-test de 17865.47.

Frente al árbol sin poda, redujo el RMSE test en 13523.62 (31.86%). Frente al árbol podado, lo redujo en 11456.82 (28.37%). Su RMSE fue menor que el de ambos árboles en este experimento. Comparado con la referencia del profesor (27588.9), fue 1340.55 mayor (4.86%). La diferencia puede explicarse por split, implementación R/Python, encoding, tratamiento de missing, parámetros del bosque o versiones de librerías; no se forzó coincidencia.

Las diez importancias transformadas principales fueron `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `2ndFlrSF`, `BsmtFinSF1`, `1stFlrSF`, `LotArea`, `GarageArea`, `GarageCars` y `YearBuilt`. La agregación por variable original suma correctamente los niveles one-hot de cada categórica; las features categóricas aparecen descompuestas por nivel en la tabla no agregada. La importancia no implica causalidad y puede variar con el split y la configuración del bosque.

### Respuesta académica

El Random Forest combina múltiples árboles, cada uno entrenado con bootstrap y con subconjuntos aleatorios de variables en las divisiones. Al promediar, reduce la varianza y la sensibilidad de un árbol individual a pequeñas variaciones del dataset. En nuestros resultados, el RMSE fue 28929.45, mejor que 42453.07 del árbol sin poda y 40386.27 del árbol podado, con mejoras de 31.86% y 28.37%, respectivamente.

La referencia del profesor es 27588.9; nuestro valor es 1340.55 mayor, una diferencia de 4.86%. No debe concluirse que Random Forest siempre es mejor ni que la diferencia sea un error: pueden cambiar el split, la implementación, el encoding, el tratamiento de missing, los parámetros y las versiones. El gap train-test de 17865.47 muestra cierta diferencia de generalización, aunque menor que el sobreajuste extremo del árbol sin poda. Boosting todavía no se ha ejecutado.
## Paso 10 - XGBoost para regresión

### Metodología y configuración

Gradient boosting construye árboles secuencialmente: cada árbol nuevo intenta corregir los errores de los anteriores. Esto contrasta con Random Forest, que construye árboles principalmente en paralelo sobre muestras bootstrap y subconjuntos aleatorios de variables, y después promedia sus predicciones. `learning_rate` controla cuánto aporta cada árbol, `n_estimators` define cuántos árboles se construyen, `max_depth` limita la profundidad y `subsample` y `colsample_bytree` introducen aleatoriedad usando fracciones de observaciones y variables.

Se mantuvo el split externo 1168/292, se excluyeron `SalePrice` e `Id`, y el preprocesamiento se ajustó únicamente dentro del pipeline sobre train. Las variables numéricas usan mediana; las categóricas usan `None` y `OneHotEncoder(handle_unknown="ignore")`. La configuración fue `XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)`, con XGBoost 3.4.1.

### Resultados

XGBoost obtuvo en train MAE 8697.48, RMSE 11960.51 y R² 0.9760. En test obtuvo MAE 15852.43, RMSE 25564.80 y R² 0.9148. El gap RMSE train-test fue 13604.29, menor que el de Random Forest (17865.47), aunque todavía existe diferencia entre entrenamiento y prueba y debe interpretarse con prudencia.

Frente al árbol sin poda, XGBoost mejoró el RMSE en 16888.27 (39.78%); frente al árbol podado, en 14821.47 (36.70%); y frente a Random Forest, en 3364.65 (11.63%). Comparado con la referencia del profesor (27588.9), el RMSE fue 2024.10 mayor (7.34%). Las diferencias pueden deberse al split, R/Python, encoding, tratamiento de missing, parámetros y versiones de librerías; no se forzó coincidencia.

Las principales importancias transformadas fueron `OverallQual`, `ExterQual_TA`, `GarageCars`, `BsmtQual_Ex`, `FullBath`, `FireplaceQu_None`, `GarageType_Attchd`, `GrLivArea`, `KitchenQual_Ex` y `GarageFinish_Unf`. La tabla agregada suma los niveles one-hot por variable original y muestra como principales `OverallQual`, `ExterQual`, `GarageCars`, `KitchenQual`, `BsmtQual`, `GarageType`, `FullBath`, `CentralAir`, `Neighborhood` y `FireplaceQu`. Las importancias no implican causalidad y las categóricas aparecen descompuestas por nivel en la tabla no agregada.

### Respuesta académica

El modelo de gradient boosting obtuvo un RMSE de 25564.80. Frente al árbol sin poda (42453.07), representa una mejora de 39.78%; frente al árbol podado (40386.27), una mejora de 36.70%; y frente a Random Forest (28929.45), una mejora de 11.63%.

La diferencia conceptual principal es que Random Forest construye múltiples árboles diversificados y promedia sus resultados, mientras que XGBoost construye árboles secuencialmente para corregir los residuos de los modelos anteriores. En este experimento XGBoost obtuvo el menor RMSE, pero no debe afirmarse que siempre sea mejor: su desempeño depende de la configuración, los datos y la validación. El gap train-test de 13604.29 indica una diferencia de generalización que debe vigilarse, aunque es menor que la de Random Forest. MLP queda pendiente para el Paso 11.
## Paso 11 - Red neuronal MLP para regresión

### Arquitectura y entrenamiento

Una MLP es una red neuronal formada por neuronas conectadas mediante pesos y sesgos. Las capas ocultas transforman las entradas y aprenden relaciones no lineales. La arquitectura utilizada fue `Dense(128)-Dense(64)-Dense(32)-Dense(1)`: tres capas ocultas de 128, 64 y 32 neuronas con activación ReLU y una neurona de salida lineal para regresión.

Se usó TensorFlow/Keras 2.21.0 porque es compatible con Python 3.13.9. El optimizador fue Adam y la función de pérdida MSE. El entrenamiento usó batch size 32, máximo de 500 epochs y early stopping sobre `val_loss` con paciencia 30 y restauración de los mejores pesos; se detuvo después de 55 epochs. Las variables numéricas se imputaron con mediana y escalaron con `StandardScaler`; las categóricas se imputaron con `None` y se codificaron con `OneHotEncoder`. El target `SalePrice` se escaló con un `StandardScaler` ajustado solo sobre `y_train` y las predicciones se regresaron a dólares antes de calcular métricas.

Durante cada epoch, la red realiza un forward pass, calcula el error, y backpropagation obtiene los gradientes del error respecto a pesos y sesgos. Adam usa esos gradientes para actualizar los parámetros. Early stopping ayuda a detener el aprendizaje cuando la pérdida de validación deja de mejorar.

### Resultados

En train la MLP obtuvo MAE 9191.99, RMSE 12679.09 y R² 0.9730. En test obtuvo MAE 17913.03, RMSE 29390.27 y R² 0.8874. El gap RMSE train-test fue 16711.19. Las métricas están expresadas en dólares reales después de invertir el escalado del target.

El ranking por RMSE test fue: XGBoost (25564.80), Random Forest (28929.45), MLP (29390.27), árbol podado (40386.27) y árbol sin poda (42453.07). La MLP empeoró frente a Random Forest en 460.82 (1.59%) y frente a XGBoost en 3825.47 (14.96%), aunque superó a ambos árboles. En un dataset tabular de 1460 observaciones, este resultado es razonable: los modelos basados en árboles pueden ser muy competitivos frente a redes neuronales.

### Respuesta académica

La MLP utilizada estuvo formada por `Dense(128)-Dense(64)-Dense(32)-Dense(1)` con activación ReLU en las capas ocultas, salida lineal y optimización mediante Adam. El aprendizaje se realizó mediante backpropagation, proceso por el cual el error calculado en la salida se propaga hacia atrás para actualizar los pesos de la red.

El modelo obtuvo un RMSE de 29390.27 en test. Comparado con Random Forest (28929.45), la MLP empeoró el desempeño en 1.59%; comparada con XGBoost (25564.80), empeoró en 14.96%. Este comportamiento es razonable en un dataset tabular de 1460 observaciones, donde modelos basados en árboles pueden ser muy competitivos frente a redes neuronales. La MLP no debe considerarse inválida: su desempeño depende de arquitectura, escalado, regularización, entrenamiento y características del dataset.
## Paso 12 - Cierre comparativo del problema de regresión

### Comparación final

Se cerró el bloque de regresión utilizando las métricas ya validadas de los pasos 07 a 11, sin entrenar modelos nuevos ni recalcular predicciones. La métrica principal fue RMSE, complementada con MAE y R². El ranking por RMSE test fue:

| Modelo | RMSE | MAE | R² | Comentario |
|---|---:|---:|---:|---|
| XGBoost | 25564.80 | 15852.43 | 0.9148 | Mejor desempeño |
| Random Forest | 28929.45 | 17410.11 | 0.8909 | Segundo |
| MLP | 29390.27 | 17913.03 | 0.8874 | Tercero y cercano a RF |
| Árbol podado | 40386.27 | 25776.51 | 0.7874 | Menor complejidad |
| Árbol sin poda | 42453.07 | 27494.58 | 0.7650 | Sobreajuste fuerte |

XGBoost fue el mejor modelo. Mejoró frente a Random Forest en 3364.65 de RMSE (11.63%) y frente a MLP en 3825.47 (13.02%). Random Forest mejora al árbol al reducir varianza mediante un ensamble; XGBoost mejora aún más mediante aprendizaje secuencial. La MLP funciona bien, pero no supera a los ensambles de árboles en este dataset tabular.

### Generalización y efecto de la poda

El árbol sin poda presenta el mayor indicio de sobreajuste: RMSE train 0 frente a RMSE test 42453.07. El árbol podado reduce la complejidad y mejora la generalización en nuestro experimento. Random Forest tiene gap train-test existente, pero buen desempeño test; XGBoost logra el menor RMSE test y un gap menor que RF; MLP también tiene buen desempeño, aunque queda por debajo de RF y XGBoost.

En nuestro experimento, la poda mejora: el RMSE pasa de 42453.07 a 40386.27. En la referencia del profesor ocurre lo contrario: pasa de 41002.2 a 42927.25, un empeoramiento de 1925.05. Esto no se presenta como error: las diferencias pueden deberse al split, criterio de poda, preprocesamiento, implementación R/Python, codificación o hiperparámetros.

### Ventajas, desventajas y utilidad

El árbol es sencillo e interpretable, pero tiene alta varianza. El árbol podado es más interpretable, aunque una poda excesiva puede producir underfitting. Random Forest es robusto y reduce varianza, con mayor costo e interpretabilidad menor. XGBoost ofrece alto desempeño y modela relaciones complejas, pero requiere más control de hiperparámetros y puede sobreajustar. La MLP es flexible y no lineal, pero es sensible al escalado, más compleja de entrenar y no necesariamente supera a los modelos de árboles en datasets tabulares pequeños.

Los modelos son realmente útiles para este conjunto de datos: especialmente Random Forest, XGBoost y MLP muestran capacidad predictiva relevante. XGBoost alcanza R²=0.9148, lo que significa que explica aproximadamente el 91.5% de la variabilidad observada en `SalePrice` en test bajo este split; no significa que prediga correctamente el 91.5% de los casos. Su RMSE de aproximadamente 25.6 mil dólares representa un error típico penalizado por cuadrados, no un error promedio exacto. Estas conclusiones se limitan a este dataset, split, configuración y laboratorio; no prueban causalidad, desempeño productivo ni superioridad universal.
## Paso 13 - Creación de grupos para clasificación

### Creación de la variable objetivo

Se creó `PriceGroup` a partir de `SalePrice` para convertir el problema de regresión en uno de clasificación. Regresión predice un valor numérico continuo; clasificación predice una clase discreta. `SalePrice` original se conservó sin modificar.

Las reglas exactas fueron: `grupo1` si `SalePrice <= 100000`; `grupo2` si `100001 <= SalePrice <= 500000`; y `grupo3` si `SalePrice >= 500001`. La implementación con condiciones consecutivas no deja huecos ni solapamientos.

### Distribución y desbalance

El dataset contiene 123 viviendas en grupo1 (8.42%), 1328 en grupo2 (90.96%) y 9 en grupo3 (0.62%). La clase mayoritaria es grupo2 y la minoritaria grupo3, con una razón de 147.56 a 1. El problema está fuertemente desbalanceado.

Este desbalance implica que una accuracy global elevada puede ocultar un mal desempeño en la clase minoritaria. Por esta razón, además de la accuracy global, será necesario analizar la matriz de confusión y la exactitud/recall por grupo.

### Split estratificado

Se definió `X` excluyendo `SalePrice`, `PriceGroup` e `Id`; `SalePrice` no puede ser predictor porque define directamente la clase y produciría target leakage. Se utilizó `train_test_split` con `test_size=0.20`, `random_state=42` y `stratify=y`. El resultado fue train 1168 y test 292: train tiene grupo1=98 (8.39%), grupo2=1063 (91.01%) y grupo3=7 (0.60%); test tiene grupo1=25 (8.56%), grupo2=265 (90.75%) y grupo3=2 (0.68%).

La estratificación preserva aproximadamente la distribución original y el mismo split debe reutilizarse en los clasificadores posteriores para que sus comparaciones sean justas y reproducibles. La distribución confirma conceptualmente la referencia del profesor: la mayoría pertenece a grupo2, grupo3 es extremadamente pequeño y el problema está fuertemente desbalanceado.
## Paso 14 - Árbol de decisión para clasificación

### Objetivo y metodología

Se entrenó un árbol de decisión multiclase para predecir `PriceGroup`, reutilizando exactamente los índices del split estratificado del Paso 13: 1168 registros de train y 292 de test, con test grupo1=25, grupo2=265 y grupo3=2. `SalePrice`, `PriceGroup` e `Id` se excluyeron de `X`; de esta forma se evita target leakage, porque `SalePrice` es precisamente la variable que define la clase.

El árbol baseline usa `DecisionTreeClassifier(random_state=42)` sin `class_weight`, profundidad máxima, poda ni tuning. El preprocesamiento se ajustó únicamente sobre train: mediana para numéricas y `None` más `OneHotEncoder(handle_unknown="ignore")` para categóricas. La matriz de confusión se interpreta con filas reales y columnas predichas.

### Resultados

La matriz de confusión fue:

| Real / predicho | grupo1 | grupo2 | grupo3 |
|---|---:|---:|---:|
| grupo1 | 15 | 10 | 0 |
| grupo2 | 13 | 252 | 0 |
| grupo3 | 0 | 2 | 0 |

La accuracy global fue 0.9144 y la balanced accuracy 0.5170. El macro F1 fue 0.5063. Por grupo, grupo1 tuvo soporte 25, 15 aciertos y recall 0.6000; grupo2 tuvo soporte 265, 252 aciertos y recall 0.9509; grupo3 tuvo soporte 2, 0 aciertos y recall 0.0000. La “exactitud por grupo” solicitada equivale aquí al recall por clase: aciertos de esa fila divididos entre el total real de esa fila.

El árbol obtuvo accuracy train 1.0 frente a 0.9144 en test, con 131 nodos, profundidad 16 y 66 hojas. Esta diferencia sugiere posible sobreajuste, pero debe analizarse junto con las métricas por clase y el fuerte desbalance. Las principales importancias fueron `OverallQual`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`, `GrLivArea`, `LotFrontage`, `YearBuilt`, `2ndFlrSF`, `KitchenQual_Gd` y `BsmtFinType1_LwQ`.

### Comparación y respuestas académicas

Frente a la referencia del profesor, el soporte coincide en los tres grupos. Para grupo1 obtuvimos recall 60.00% frente a 44.00%; para grupo2, 95.09% frente a 96.23%; y para grupo3, 0% frente a 0%. La matriz no es idéntica, pero el comportamiento de grupo3 coincide: no se acertó ningún caso.

La exactitud por grupo se calcula como aciertos de la fila entre el total real de la fila. Por tanto, fue 60.00% para grupo1, 95.09% para grupo2 y 0% para grupo3.

La conclusión principal es que la accuracy global de 91.44% está dominada por grupo2, que representa 90.75% del test, y oculta el mal desempeño de grupo3. Por eso se deben revisar matriz de confusión, precision, recall, F1 y balanced accuracy, no solo accuracy.

Grupo3 tiene únicamente 2 casos en test y 7 en train, por lo que no es posible aprender un patrón robusto con suficiente confianza. Para pasos posteriores podrían considerarse aumentar ejemplos, `class_weight`, sobremuestreo solo en train, SMOTE con cuidado, revisar los límites si el contexto de negocio lo permite y evaluar siempre métricas por clase. Estas técnicas no se aplicaron en este paso.
## Paso 15 - Random Forest para clasificación

### Metodología

Se entrenó un `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)` para predecir `PriceGroup`. Se reutilizó exactamente el split del Paso 13: 1168 registros de train y 292 de test, con distribución test 25/265/2. `SalePrice`, `PriceGroup` e `Id` se excluyeron de `X`, y el preprocesamiento se ajustó exclusivamente con train mediante imputación y `OneHotEncoder`.

Random Forest combina árboles construidos con muestras bootstrap y subconjuntos aleatorios de variables. Frente a un árbol individual, el ensamble suele reducir la varianza, aunque no garantiza mejorar todas las métricas ni resolver por sí mismo el desbalance.

### Resultados y comparación con el árbol

Random Forest obtuvo accuracy 0.9418, balanced accuracy 0.4908 y macro F1 0.5280 en test. El árbol del Paso 14 obtuvo respectivamente 0.9144, 0.5170 y 0.5063. Por tanto, Random Forest mejoró accuracy en 0.0274 y macro F1 en 0.0218, pero redujo balanced accuracy en 0.0262.

La matriz de confusión de Random Forest fue:

| Real / predicho | grupo1 | grupo2 | grupo3 |
|---|---:|---:|---:|
| grupo1 | 12 | 13 | 0 |
| grupo2 | 2 | 263 | 0 |
| grupo3 | 0 | 2 | 0 |

Por grupo, grupo1 tuvo soporte 25, 12 aciertos y recall 0.48; grupo2 tuvo soporte 265, 263 aciertos y recall 0.9925; grupo3 tuvo soporte 2, 0 aciertos y recall 0. La “exactitud por grupo” corresponde al recall de cada clase: aciertos de la fila divididos entre su soporte real.

El árbol tenía recalls 0.60, 0.9509 y 0 para grupo1, grupo2 y grupo3. El ensamble mejoró grupo2, empeoró grupo1 y no cambió grupo3. La accuracy elevada está dominada por grupo2, que representa aproximadamente el 91% del test; por eso deben priorizarse balanced accuracy, macro F1 y recall por grupo.

### Comparación con el profesor y grupo3

La referencia del profesor usa soportes ligeramente distintos: grupo1=24, grupo2=266 y grupo3=2. En nuestro split, los soportes son 25, 265 y 2 y no se modificaron. El profesor reporta recalls de 54.17%, 99.62% y 50%; nuestro modelo obtuvo 48%, 99.25% y 0%, respectivamente. La comparación de grupo3 no debe generalizarse: hay solo 7 ejemplos en train y 2 en test, de modo que una observación cambia el recall en 50 puntos porcentuales.

El modelo alcanzó accuracy train 1.0 frente a 0.9418 en test, lo que sugiere posible sobreajuste y debe analizarse junto con las métricas por clase. No se aplicaron `class_weight`, SMOTE, oversampling ni tuning. Para grupo3 podrían considerarse aumentar ejemplos, ponderación de clases, sobremuestreo solo en train, SMOTE con cuidado, revisar límites si el negocio lo permite y evaluar siempre métricas por clase. Estas técnicas quedan para pasos posteriores.

Las importancias principales fueron `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`, `OverallQual`, `LotArea`, `YearRemodAdd`, `OverallCond`, `LotFrontage` y `YearBuilt`. Las categóricas codificadas aparecen descompuestas por nivel; la importancia no implica causalidad.

### Respuesta académica

Random Forest mejoró la accuracy y el macro F1 respecto al árbol, pero no la balanced accuracy. El ensamble concentró aún más su buen desempeño en grupo2: el recall pasó a 99.25%, mientras grupo1 bajó a 48% y grupo3 permaneció en 0%. Por ello, no basta con observar la accuracy global; la clasificación sigue limitada por el fuerte desbalance.

Comparado con el árbol de decisión, Random Forest es más robusto frente a variaciones del dataset y reduce varianza al promediar muchos árboles, pero puede ser menos interpretable. En este experimento obtuvo mejor accuracy y macro F1, pero peor balanced accuracy, y no resolvió la clase extremadamente minoritaria. No se balancearon las clases todavía.
