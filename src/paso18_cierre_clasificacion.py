"""Cierre comparativo: lee resultados existentes y no entrena modelos."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["arbol", "random_forest", "xgboost", "svm_linear_c1"]
LABELS = ["grupo1", "grupo2", "grupo3"]

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_dir, tables_dir, figures_dir = [root / "outputs" / d for d in ("metrics", "tables", "figures")]
    comparison = pd.read_csv(tables_dir / "paso17_comparacion_clasificadores.csv")
    comparison = comparison.rename(columns={"modelo": "modelo"})
    comparison.loc[comparison.modelo == "svm_mejor", "modelo"] = "svm_linear_c1"
    if comparison.modelo.tolist() != MODELS: raise RuntimeError("No hay exactamente cuatro modelos cerrados")
    columns = ["modelo", "accuracy", "balanced_accuracy", "macro_f1", "grupo1_recall", "grupo2_recall", "grupo3_recall"]
    comparison = comparison[columns]
    comparison.to_csv(tables_dir / "paso18_comparacion_final_clasificacion.csv", index=False)
    expected = {"accuracy": ["xgboost", "svm_linear_c1", "random_forest", "arbol"], "balanced_accuracy": ["svm_linear_c1", "xgboost", "arbol", "random_forest"], "macro_f1": ["svm_linear_c1", "xgboost", "random_forest", "arbol"]}
    for metric, order in expected.items():
        ranking = comparison.sort_values(metric, ascending=False, kind="stable").reset_index(drop=True)
        if ranking.modelo.tolist() != order: raise RuntimeError(f"Ranking inesperado para {metric}")
        ranking.insert(0, "ranking", np.arange(1, len(ranking) + 1))
        ranking.to_csv(tables_dir / f"paso18_ranking_{metric}.csv", index=False)
    group_rows = []
    for group in LABELS:
        values = {"arbol": comparison.loc[comparison.modelo == "arbol", f"{group}_recall"].iloc[0], "random_forest": comparison.loc[comparison.modelo == "random_forest", f"{group}_recall"].iloc[0], "xgboost": comparison.loc[comparison.modelo == "xgboost", f"{group}_recall"].iloc[0], "svm_linear_c1": comparison.loc[comparison.modelo == "svm_linear_c1", f"{group}_recall"].iloc[0]}
        best_model = max(values, key=values.get)
        group_rows.append({"grupo": group, **{f"{k}_recall": v for k, v in values.items()}, "mejor_modelo_grupo": best_model, "mejor_recall": values[best_model]})
    pd.DataFrame(group_rows).to_csv(tables_dir / "paso18_comparacion_por_grupo.csv", index=False)
    professor = pd.DataFrame([
        {"modelo": "arbol", "grupo1_recall_profesor": .44, "grupo2_recall_profesor": .962, "grupo3_recall_profesor": 0.0, "grupo1_recall_nuestro": .60, "grupo2_recall_nuestro": .9509433962264152, "grupo3_recall_nuestro": 0.0, "soporte_profesor": "24/266/2", "soporte_nuestro": "25/265/2", "nota": "splits distintos; no es un error"},
        {"modelo": "random_forest", "grupo1_recall_profesor": .542, "grupo2_recall_profesor": .996, "grupo3_recall_profesor": .50, "grupo1_recall_nuestro": .48, "grupo2_recall_nuestro": .9924528301886792, "grupo3_recall_nuestro": 0.0, "soporte_profesor": "24/266/2", "soporte_nuestro": "25/265/2", "nota": "splits distintos; no es un error"},
    ])
    professor.to_csv(tables_dir / "paso18_comparacion_profesor.csv", index=False)
    advantages = pd.DataFrame([
        {"modelo": "arbol", "ventajas": "interpretable; simple", "desventajas": "alta varianza; peor desempeño general"},
        {"modelo": "random_forest", "ventajas": "robusto; buen accuracy", "desventajas": "no detectó grupo3; balanced accuracy baja"},
        {"modelo": "xgboost", "ventajas": "mayor accuracy; buen grupo1; buen desempeño global", "desventajas": "no detectó grupo3; más hiperparámetros"},
        {"modelo": "svm_linear_c1", "ventajas": "mejor balanced accuracy; mejor macro F1; mejor resultado en minoritarias", "desventajas": "requiere escalado; menos interpretable; sensible a C/kernel"},
    ])
    advantages.to_csv(tables_dir / "paso18_ventajas_desventajas.csv", index=False)
    for metric, filename, title in [("accuracy", "paso18_accuracy_modelos.png", "Accuracy por modelo"), ("balanced_accuracy", "paso18_balanced_accuracy_modelos.png", "Balanced accuracy por modelo"), ("macro_f1", "paso18_macro_f1_modelos.png", "Macro F1 por modelo")]:
        plt.figure(figsize=(8, 5)); plt.bar(comparison.modelo, comparison[metric]); plt.ylim(0, 1); plt.ylabel(metric); plt.title(title); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(figures_dir / filename, dpi=150); plt.close()
    plt.figure(figsize=(8, 5)); x = np.arange(3); width = .2
    for i, model in enumerate(MODELS): plt.bar(x + (i - 1.5) * width, [comparison.loc[comparison.modelo == model, f"{g}_recall"].iloc[0] for g in LABELS], width, label=model)
    plt.xticks(x, LABELS); plt.ylim(0, 1); plt.ylabel("Recall"); plt.title("Recall por grupo y modelo"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso18_recall_por_grupo_modelos.png", dpi=150); plt.close()
    best_acc = comparison.loc[comparison.accuracy.idxmax()]; best_bal = comparison.loc[comparison.balanced_accuracy.idxmax()]; best_f1 = comparison.loc[comparison.macro_f1.idxmax()]
    group_best = {row["grupo"]: row["mejor_modelo_grupo"] for row in group_rows}
    pd.DataFrame([{ "best_accuracy_model": best_acc.modelo, "best_accuracy": best_acc.accuracy, "best_balanced_model": best_bal.modelo, "best_balanced_accuracy": best_bal.balanced_accuracy, "best_macro_f1_model": best_f1.modelo, "best_macro_f1": best_f1.macro_f1, "best_grupo1_model": group_best["grupo1"], "best_grupo1_recall": group_rows[0]["mejor_recall"], "best_grupo2_model": group_best["grupo2"], "best_grupo2_recall": group_rows[1]["mejor_recall"], "best_grupo3_model": group_best["grupo3"], "best_grupo3_recall": group_rows[2]["mejor_recall"], "recommended_model_balanced": "svm_linear_c1", "minority_class": "grupo3", "minority_class_total": 9 }]).to_csv(metrics_dir / "paso18_resumen_clasificacion.csv", index=False)
    print("Cierre generado sin entrenar modelos."); print(comparison.to_string(index=False)); print(pd.DataFrame(group_rows).to_string(index=False)); print(f"Mejor accuracy: {best_acc.modelo}; mejor balanced accuracy: {best_bal.modelo}; mejor macro F1: {best_f1.modelo}")

if __name__ == "__main__": main()
