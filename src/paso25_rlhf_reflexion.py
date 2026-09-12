"""Artefactos conceptuales sobre RLHF; no entrena modelos ni usa servicios externos."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    for directory in (tables_dir, metrics_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    comparison = [
        ("objetivo", "aprender valores de acción para alcanzar una meta", "optimizar el comportamiento según preferencias humanas"),
        ("tipo_recompensa", "numérica y definida por reglas", "señal aprendida o derivada de preferencias"),
        ("fuente_recompensa", "diseñador del entorno", "evaluadores humanos, directa o indirectamente"),
        ("estado", "posición del agente en el Gridworld", "contexto, prompt y conversación"),
        ("accion", "arriba, derecha, abajo o izquierda", "token, secuencia o respuesta generada"),
        ("politica", "acción con mayor valor Q en cada estado", "modelo que decide qué respuesta producir"),
        ("aprendizaje", "actualización de Q-table por experiencia", "preferencias → reward model → optimización de política"),
        ("retroalimentacion_humana", "no utilizada", "comparaciones, rankings, puntuaciones, correcciones o demostraciones, según el sistema"),
        ("escala", "entorno pequeño y explícito", "modelos y datos potencialmente mucho más grandes"),
        ("casos_de_uso", "control y navegación en entornos discretos", "alineación y utilidad de modelos generativos"),
        ("riesgos", "exploración ineficiente y Q-table poco escalable", "sesgo, reward hacking, errores del reward model y sobreoptimización"),
    ]
    pd.DataFrame(comparison, columns=["aspecto", "q_learning", "rlhf"]).to_csv(tables_dir / "paso25_qlearning_vs_rlhf.csv", index=False)
    flow = [
        (1, "modelo base", "modelo preentrenado", "modelo inicial"),
        (2, "generación de respuestas", "prompts/contextos", "respuestas candidatas"),
        (3, "evaluación/preferencias humanas", "pares o listas de respuestas", "preferencias o rankings"),
        (4, "entrenamiento de modelo de recompensa", "preferencias humanas", "reward model"),
        (5, "optimización de política", "modelo base y reward model", "política ajustada"),
        (6, "evaluación/alineación", "política y criterios de evaluación", "modelo evaluado y posibles iteraciones"),
    ]
    pd.DataFrame(flow, columns=["etapa", "descripcion", "entrada", "salida"]).to_csv(tables_dir / "paso25_flujo_rlhf.csv", index=False)
    relation = [
        ("agente", "agente en Gridworld", "modelo generativo"), ("entorno", "cuadrícula 4x4 determinista", "contexto de interacción y distribución de tareas"), ("estado", "posición actual", "prompt + conversación"), ("accion", "movimiento discreto", "token, secuencia o respuesta"), ("recompensa", "-1 por paso y +10 en la meta", "señal influenciada por preferencias humanas"), ("politica", "tabla de acciones con mayor Q", "modelo que selecciona respuestas"), ("exploracion", "epsilon-greedy", "búsqueda de comportamientos y evaluación de alternativas, de forma conceptual"), ("objetivo", "llegar a la meta con pocos pasos", "producir respuestas útiles y preferidas"),
    ]
    pd.DataFrame(relation, columns=["concepto", "paso24_qlearning", "rlhf"]).to_csv(tables_dir / "paso25_relacion_paso24_rlhf.csv", index=False)
    benefits = ["incorpora preferencias humanas", "sirve cuando la recompensa es difícil de programar", "puede mejorar utilidad y seguimiento de instrucciones", "puede reducir ciertos comportamientos no deseados", "permite optimizar criterios cualitativos"]
    risks = ["sesgo de evaluadores", "preferencias inconsistentes", "coste de anotación", "reward hacking", "sobreoptimización del reward model", "generalización imperfecta", "subjetividad", "dificultad de representar valores diversos", "errores del reward model"]
    pd.DataFrame({"beneficio": benefits}).to_csv(tables_dir / "paso25_beneficios_rlhf.csv", index=False); pd.DataFrame({"riesgo_o_limitacion": risks}).to_csv(tables_dir / "paso25_riesgos_rlhf.csv", index=False)
    supervised = [("tipo de señal", "etiquetas o respuestas objetivo", "preferencias humanas y señal de recompensa"), ("objetivo", "ajustar una salida a una referencia", "optimizar comportamiento preferido"), ("datos", "pares entrada-salida", "comparaciones, rankings u otras señales"), ("feedback", "generalmente incorporado como etiqueta", "evaluación humana explícita o indirecta"), ("uso típico", "predicción y clasificación", "alineación de políticas o modelos"), ("limitaciones", "coste y rigidez de etiquetas", "sesgos, inconsistencias y reward hacking")]
    pd.DataFrame(supervised, columns=["aspecto", "supervisado", "rlhf"]).to_csv(tables_dir / "paso25_supervisado_vs_rlhf.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 3)); ax.axis("off"); stages = ["Modelo base", "Respuestas", "Preferencias\nhumanas", "Reward\nmodel", "Optimización\nde política", "Política\najustada"]
    for i, stage in enumerate(stages):
        ax.text(i, 0.5, stage, ha="center", va="center", bbox={"boxstyle":"round,pad=0.7"}, fontsize=11)
        if i < len(stages)-1: ax.annotate("", xy=(i+0.75,0.5), xytext=(i+0.25,0.5), arrowprops={"arrowstyle":"->"})
    ax.set_xlim(-.5, 5.5); ax.set_ylim(0, 1); ax.set_title("Flujo conceptual de RLHF"); fig.tight_layout(); fig.savefig(figures_dir / "paso25_flujo_rlhf.png", dpi=150); plt.close(fig)
    pd.DataFrame([{ "tipo_paso":"conceptual", "implementacion_real_rlhf":"no", "usa_modelo_recompensa_real":"no", "usa_feedback_humano_real":"no", "relacion_paso24":"Q-learning usa recompensa fija; RLHF deriva la señal de preferencias humanas", "principal_diferencia":"origen de la señal de recompensa", "principal_beneficio":"incorporar criterios cualitativos difíciles de programar", "principal_riesgo":"optimizar una señal imperfecta y producir reward hacking" }]).to_csv(metrics_dir / "paso25_resumen_rlhf.csv", index=False)
    print("Reflexión RLHF generada; no se entrenaron modelos ni se realizaron llamadas externas.")

if __name__ == "__main__": main()
