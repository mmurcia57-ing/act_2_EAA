# Índice de evidencias - Actividad 2

| Requisito | Paso | Script | Tabla principal | Figura | Conclusión |
|---|---:|---|---|---|---|
| EDA y estructura | 02–05 | `src/paso02_inspeccion_estructura.py` | `outputs/tables/paso02_missing_resumen.csv` | `outputs/figures/paso05_matriz_correlacion.png` | Se identificaron distribución, correlaciones y multicolinealidades relevantes. |
| Valores faltantes | 06 | `src/paso06_tratamiento_missing.py` | `outputs/tables/paso06_consistencia_estructural.csv` | `outputs/figures/paso06_missing_antes_despues.png` | Se imputó sin leakage; el CSV imputado completo fue solo evidencia. |
| Regresión | 07–12 | `src/paso10_xgboost_regresion.py`, `src/paso11_mlp_regresion.py` | `outputs/tables/paso26_resumen_regresion.csv` | `outputs/figures/paso26_resumen_regresion.png` | XGBoost obtuvo el menor RMSE bajo el split evaluado. |
| Clasificación | 13–18 | `src/paso17_svm_clasificacion.py` | `outputs/tables/paso26_resumen_clasificacion.csv` | `outputs/figures/paso26_resumen_clasificacion.png` | XGBoost lideró accuracy; SVM lideró métricas balanceadas. |
| Selección de K | 19 | `src/paso19_seleccion_k_clustering.py` | `outputs/tables/paso19_metricas_k.csv` | `outputs/figures/paso19_silhouette_por_k.png` | K=2 maximizó silhouette. |
| K-Means y jerárquico | 20–22 | `src/paso20_kmeans_segmentacion.py`, `src/paso21_clustering_jerarquico.py` | `outputs/tables/paso22_resumen_kmeans_vs_jerarquico.csv` | `outputs/figures/paso22_saleprice_perfiles.png` | Ambos métodos hallaron estructura relacionada; K-Means tuvo mayor silhouette. |
| Anomalías | 23 | `src/paso23_isolation_forest.py` | `outputs/tables/paso23_variables_diferenciadoras.csv` | `outputs/figures/paso23_distribucion_anomaly_score.png` | Se detectaron 70 casos atípicos; anomalía no equivale a fraude. |
| Q-learning | 24 | `src/paso24_q_learning.py` | `outputs/metrics/paso24_evaluacion.csv` | `outputs/figures/paso24_politica_grid.png` | La política alcanzó la meta en 100% de evaluaciones. |
| RLHF | 25 | `src/paso25_rlhf_reflexion.py` | `outputs/tables/paso25_qlearning_vs_rlhf.csv` | `outputs/figures/paso25_flujo_rlhf.png` | Reflexión conceptual; no hubo implementación real. |
| Conclusiones y rúbrica | 26–27 | `src/paso26_conclusiones_globales.py`, `src/paso27_entrega_final.py` | `outputs/metrics/paso27_check_final.csv` | — | La entrega integra resultados, decisiones y limitaciones. |
