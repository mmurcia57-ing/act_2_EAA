"""Consolidación global de la actividad; lee resultados y no entrena modelos."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def main() -> None:
    root = Path(__file__).resolve().parents[1]; td, md, fd = [root / d for d in ("outputs/tables", "outputs/metrics", "outputs/figures")]
    # Los valores se leen de los artefactos ya producidos en pasos anteriores.
    reg_sources = [("Árbol sin poda", "paso07_arbol_regresion_metricas.csv", "rmse", "mae", "r2"), ("Árbol podado", "paso08_poda_resumen.csv", "rmse_podado", "mae_podado", "r2_podado"), ("Random Forest", "paso09_random_forest_regresion_metricas.csv", "rmse_test", "mae_test", "r2_test"), ("MLP", "paso11_mlp_regresion_metricas.csv", "rmse_test", "mae_test", "r2_test"), ("XGBoost", "paso10_xgboost_regresion_metricas.csv", "rmse_test", "mae_test", "r2_test")]
    reg_rows=[]
    for model, file, rmse, mae, r2 in reg_sources:
        row=pd.read_csv(md/file).iloc[0]; reg_rows.append({"modelo":model,"rmse":float(row[rmse]),"mae":float(row[mae]),"r2":float(row[r2])})
    reg=pd.DataFrame(reg_rows).sort_values("rmse").reset_index(drop=True); reg.insert(4,"ranking_rmse",range(1,len(reg)+1)); reg.to_csv(td/"paso26_resumen_regresion.csv",index=False)
    clf=pd.read_csv(td/"paso18_comparacion_final_clasificacion.csv"); clf["modelo"]=clf.modelo.replace({"svm_linear_c1":"SVM lineal C=1","arbol":"Árbol","random_forest":"Random Forest","xgboost":"XGBoost"}); clf.to_csv(td/"paso26_resumen_clasificacion.csv",index=False)
    cluster=pd.DataFrame([{ "metodo":"K-Means", "k":2,"silhouette":float(pd.read_csv(md/"paso20_resumen_kmeans.csv").iloc[0].silhouette),"perfil_bajo":769,"perfil_alto":691},{"metodo":"Jerárquico Ward","k":2,"silhouette":float(pd.read_csv(md/"paso21_resumen_jerarquico.csv").iloc[0].silhouette),"perfil_bajo":904,"perfil_alto":556}]); cluster["ARI_kmeans_vs_jerarquico"]=float(pd.read_csv(md/"paso21_agreement.csv").iloc[0].adjusted_rand_index); cluster.to_csv(td/"paso26_resumen_clustering.csv",index=False)
    anomaly=pd.read_csv(md/"paso23_resumen.csv").rename(columns={"n_anomaly":"n_anomalies"}); anomaly.to_csv(td/"paso26_resumen_anomalias.csv",index=False)
    q=pd.read_csv(md/"paso24_resumen.csv"); q.to_csv(td/"paso26_resumen_qlearning.csv",index=False)
    rlhf=pd.read_csv(md/"paso25_resumen_rlhf.csv"); rlhf.to_csv(td/"paso26_resumen_rlhf.csv",index=False)
    findings=[("EDA","Dataset de housing con 1460 filas y 81 columnas; se revisaron estructura, tipos, faltantes y relaciones.","outputs/tables/paso02_* y paso03_*","La inspección inicial orientó el modelado"),("missing","Se documentaron faltantes y se imputaron dentro de pipelines o con reglas reproducibles.","outputs/tables/paso06_*","Se evitó leakage en supervisado"),("regresión","XGBoost obtuvo RMSE 25564.80 y R2 0.9148.","paso26_resumen_regresion.csv","Mejor desempeño predictivo bajo este split"),("clasificación","XGBoost tuvo mayor accuracy; SVM C=1 mayor balanced accuracy 0.7916 y macro F1 0.8015.","paso26_resumen_clasificacion.csv","El desbalance exige métricas complementarias"),("clustering","K=2; K-Means silhouette 0.1424 y jerárquico 0.1150; ARI 0.5735.","paso26_resumen_clustering.csv","Estructura exploratoria y separación limitada"),("anomalías","Isolation Forest detectó 70 anomalías (4.79%).","paso26_resumen_anomalias.csv","Casos estructuralmente inusuales, no fraude"),("reinforcement_learning","Q-learning logró 100% de éxito, 6 pasos y recompensa media 5.","paso26_resumen_qlearning.csv","Política eficiente en Gridworld simple"),("RLHF","Se analizó conceptualmente; no hubo implementación real.","paso26_resumen_rlhf.csv","La recompensa puede derivarse de preferencias humanas")]
    pd.DataFrame(findings,columns=["area","hallazgo","evidencia","implicacion"]).to_csv(td/"paso26_hallazgos_principales.csv",index=False)
    limitations=["dataset relativamente pequeño y único","split único en modelos supervisados","grupo3 extremadamente pequeño","tuning limitado y sin validación cruzada exhaustiva","silhouette bajo y separación limitada","sin ground truth de anomalías","entorno Q-learning artificial","RLHF tratado solo conceptualmente","resultados no deben generalizarse sin validación externa"]
    pd.DataFrame({"limitacion":limitations}).to_csv(td/"paso26_limitaciones.csv",index=False)
    decisions=["evitar target leakage","split antes de imputar/escalar en supervisado","SalePrice fuera de clustering","SalePrice fuera de Isolation Forest","StandardScaler para SVM y clustering","no usar accuracy sola con desbalance","no interpretar clusters como clases","no interpretar anomalías como fraude","no presentar RLHF conceptual como implementación real"]
    pd.DataFrame({"decision_metodologica":decisions}).to_csv(td/"paso26_decisiones_metodologicas.csv",index=False)
    rubric=[("EDA","si","inspección estructural, estadística y correlaciones","02-05"),("missing","si","tratamiento e imputación documentados","06"),("regresión tree/RF/boosting/NN","si","métricas de árbol, RF, XGBoost y MLP","07-12"),("clasificación tree/RF/boosting/SVM","si","comparación final de cuatro modelos","14-18"),("clustering","si","K-Means y jerárquico con silhouette/ARI","19-22"),("anomalías","si","Isolation Forest y perfiles","23"),("RL","si","Q-learning Gridworld y evaluación","24"),("RLHF","si","reflexión conceptual y tablas","25"),("comentarios/documentación","si","guía paso a paso y README","01-26")]
    pd.DataFrame(rubric,columns=["criterio","cubierto","evidencia","paso"]).to_csv(td/"paso26_check_rubrica.csv",index=False)
    priority=[("metodología","alta","paso26_mapa_tecnicas.png","paso26_decisiones_metodologicas.csv","Presentar el flujo y decisiones contra leakage"),("regresión","alta","paso26_resumen_regresion.png","paso26_resumen_regresion.csv","Comparar RMSE y R2"),("clasificación","alta","paso26_resumen_clasificacion.png","paso26_resumen_clasificacion.csv","Priorizar balanced accuracy y macro F1"),("clustering","media","—","paso26_resumen_clustering.csv","Reportar silhouette y ARI"),("anomalías/RL/RLHF","media","—","paso26_resumen_anomalias.csv; paso26_resumen_qlearning.csv; paso26_resumen_rlhf.csv","Sintetizar sin sobrecargar el informe"),("limitaciones","alta","—","paso26_limitaciones.csv","Cerrar con alcance y generalización")]
    pd.DataFrame(priority,columns=["seccion","prioridad","figura_recomendada","tabla_recomendada","nota"]).to_csv(td/"paso26_prioridad_informe.csv",index=False)
    plt.figure(figsize=(9,5)); plt.bar(reg.modelo,reg.rmse); plt.ylabel("RMSE"); plt.title("RMSE por modelo de regresión"); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(fd/"paso26_resumen_regresion.png",dpi=150); plt.close()
    plot_clf=clf.copy(); x=range(len(plot_clf)); width=.25; plt.figure(figsize=(10,5));
    for i,m in enumerate(["accuracy","balanced_accuracy","macro_f1"]): plt.bar([v+(i-1)*width for v in x],plot_clf[m],width,label=m)
    plt.xticks(list(x),plot_clf.modelo,rotation=20); plt.ylim(0,1); plt.ylabel("Métrica"); plt.title("Métricas de clasificación"); plt.legend(); plt.tight_layout(); plt.savefig(fd/"paso26_resumen_clasificacion.png",dpi=150); plt.close()
    fig,ax=plt.subplots(figsize=(10,4)); ax.axis("off"); boxes=[("Supervisado","Regresión\nClasificación"),("No supervisado","Clustering\nAnomalías"),("Reinforcement learning","Q-learning\nRLHF conceptual")]
    for i,(title,body) in enumerate(boxes): ax.text(i,0.5,f"{title}\n\n{body}",ha="center",va="center",bbox={"boxstyle":"round,pad=1"},fontsize=11); 
    fig.tight_layout(); fig.savefig(fd/"paso26_mapa_tecnicas.png",dpi=150); plt.close(fig)
    print("Conclusiones globales consolidadas sin entrenar modelos.")

if __name__ == "__main__": main()
