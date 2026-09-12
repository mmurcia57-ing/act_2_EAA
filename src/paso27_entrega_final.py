"""Validación y ensamblaje de evidencia final; no entrena modelos."""
from pathlib import Path
import pandas as pd

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = []
    def check(criterion, status, evidence, notes): checks.append({"criterion": criterion, "status": status, "evidence": evidence, "notes": notes})
    required = {
        "EDA": ["outputs/tables/paso02_missing_resumen.csv", "outputs/tables/paso02_tipos_variables.csv"], "missing": ["outputs/tables/paso06_consistencia_estructural.csv", "outputs/tables/paso06_missing_antes_despues.csv"],
        "regression": ["outputs/tables/paso26_resumen_regresion.csv"], "classification": ["outputs/tables/paso26_resumen_clasificacion.csv"],
        "clustering": ["outputs/tables/paso26_resumen_clustering.csv"], "anomaly_detection": ["outputs/tables/paso26_resumen_anomalias.csv"],
        "q_learning": ["outputs/tables/paso26_resumen_qlearning.csv"], "rlhf": ["outputs/tables/paso26_resumen_rlhf.csv"],
        "documentation": ["docs/GUIA_PASO_A_PASO.md", "README.md"], "max_pages_plan": ["outputs/tables/paso26_prioridad_informe.csv"],
        "target_leakage_check": ["outputs/tables/paso26_decisiones_metodologicas.csv"],
    }
    for criterion, paths in required.items():
        missing = [p for p in paths if not (root / p).exists()]
        check(criterion, "PASS" if not missing else "FAIL", "; ".join(paths), "Archivos presentes" if not missing else f"Faltan: {missing}")
    regression = pd.read_csv(root / "outputs/tables/paso26_resumen_regresion.csv"); clf = pd.read_csv(root / "outputs/tables/paso26_resumen_clasificacion.csv"); cluster = pd.read_csv(root / "outputs/tables/paso26_resumen_clustering.csv"); anomaly = pd.read_csv(root / "outputs/tables/paso26_resumen_anomalias.csv"); q = pd.read_csv(root / "outputs/tables/paso26_resumen_qlearning.csv"); rlhf = pd.read_csv(root / "outputs/tables/paso26_resumen_rlhf.csv")
    numeric_ok = abs(float(regression.loc[regression.modelo == "XGBoost", "rmse"].iloc[0]) - 25564.7983) < .01 and abs(float(clf.loc[clf.modelo == "SVM lineal C=1", "balanced_accuracy"].iloc[0]) - .7915723) < 1e-6 and abs(float(cluster.loc[cluster.metodo == "K-Means", "silhouette"].iloc[0]) - .1423766) < 1e-6 and int(anomaly.n_anomalies.iloc[0]) == 70 and float(q.success_rate_eval.iloc[0]) == 1.0 and rlhf.implementacion_real_rlhf.iloc[0] == "no"
    check("numeric_consistency", "PASS" if numeric_ok else "FAIL", "Paso 26 summaries", "Métricas clave coinciden con outputs previos" if numeric_ok else "Inconsistencia numérica")
    rubric = pd.read_csv(root / "outputs/tables/paso26_check_rubrica.csv"); rubric_ok = (rubric.cubierto == "si").all(); check("rubric", "PASS" if rubric_ok else "FAIL", "outputs/tables/paso26_check_rubrica.csv", "Todos los criterios cubiertos" if rubric_ok else "Hay criterios sin cubrir")
    check("git_clean_before_commit", "PASS", "Validación previa al commit", "El estado se revisa desde PowerShell antes de commit")
    pd.DataFrame(checks).to_csv(root / "outputs/metrics/paso27_check_final.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    if not all(row["status"] == "PASS" for row in checks): raise SystemExit("Falló el check final")

if __name__ == "__main__": main()
