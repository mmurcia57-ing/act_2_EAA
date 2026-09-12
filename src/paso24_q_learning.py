"""Ejemplo pedagógico y reproducible de Q-learning en un Gridworld 4x4."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N = 4; N_STATES = 16; N_ACTIONS = 4; START = 0; GOAL = 15
ACTION_NAMES = ["up", "right", "down", "left"]

def transition(state: int, action: int) -> tuple[int, float, bool]:
    row, col = divmod(state, N); next_row, next_col = row, col
    if action == 0: next_row -= 1
    elif action == 1: next_col += 1
    elif action == 2: next_row += 1
    elif action == 3: next_col -= 1
    if not (0 <= next_row < N and 0 <= next_col < N):
        next_state = state
    else: next_state = next_row * N + next_col
    if next_state == GOAL: return next_state, 10.0, True
    return next_state, -1.0, False

def choose_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon: return int(rng.integers(N_ACTIONS))
    best = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(rng.choice(best))

def greedy_action(Q: np.ndarray, state: int, rng: np.random.Generator) -> int:
    best = np.flatnonzero(np.isclose(Q[state], Q[state].max()))
    return int(rng.choice(best))

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    for directory in (tables_dir, metrics_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    alpha, gamma = 0.1, 0.95; epsilon_start, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995; episodes, max_steps, random_state = 2000, 100, 42
    Q = np.zeros((N_STATES, N_ACTIONS)); rng = np.random.default_rng(random_state); epsilon = epsilon_start; history = []
    for episode in range(1, episodes + 1):
        state = START; total_reward = 0.0; reached = False
        for step in range(1, max_steps + 1):
            action = choose_action(Q[state], epsilon, rng); next_state, reward, terminal = transition(state, action)
            target = reward if terminal else reward + gamma * Q[next_state].max()
            Q[state, action] += alpha * (target - Q[state, action]); total_reward += reward; state = next_state
            if terminal: reached = True; break
        history.append({"episode": episode, "total_reward": total_reward, "steps": step, "epsilon": epsilon, "reached_goal": reached})
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    history_df = pd.DataFrame(history); history_df.to_csv(tables_dir / "paso24_qlearning_episodios.csv", index=False)
    policy_rows = []
    for state in range(N_STATES):
        row, col = divmod(state, N); action = greedy_action(Q, state, rng)
        policy_rows.append({"state": state, "row": row, "col": col, "best_action": action, "best_action_name": ACTION_NAMES[action], "best_q_value": Q[state, action]})
    policy_df = pd.DataFrame(policy_rows); policy_df.to_csv(tables_dir / "paso24_politica_final.csv", index=False)
    pd.DataFrame(Q, columns=ACTION_NAMES).assign(state=np.arange(N_STATES)).loc[:, ["state", *ACTION_NAMES]].to_csv(tables_dir / "paso24_q_table.csv", index=False)
    eval_rng = np.random.default_rng(random_state + 1); eval_rewards=[]; success_steps=[]; successes=0
    for _ in range(100):
        state=START; total=0.0
        for step in range(1, max_steps + 1):
            action=greedy_action(Q, state, eval_rng); state, reward, terminal=transition(state, action); total += reward
            if terminal: successes += 1; success_steps.append(step); break
        eval_rewards.append(total)
    evaluation = {"n_eval_episodes":100, "success_rate":successes/100, "mean_steps_success":float(np.mean(success_steps)) if success_steps else np.nan, "mean_reward":float(np.mean(eval_rewards))}
    if not np.isfinite([evaluation["success_rate"], evaluation["mean_steps_success"], evaluation["mean_reward"]]).all(): raise RuntimeError("Evaluación no finita")
    pd.DataFrame([evaluation]).to_csv(metrics_dir / "paso24_evaluacion.csv", index=False)
    trajectory=[]; state=START
    for step in range(1, max_steps + 1):
        action=greedy_action(Q, state, eval_rng); next_state, reward, terminal=transition(state, action); row,col=divmod(state,N); trajectory.append({"step":step, "state":state, "row":row, "col":col, "action":action, "next_state":next_state, "reward":reward}); state=next_state
        if terminal: break
    if state != GOAL: raise RuntimeError("La trayectoria final no llegó a la meta")
    pd.DataFrame(trajectory).to_csv(tables_dir / "paso24_trayectoria_final.csv", index=False)
    plt.figure(figsize=(9,4)); plt.plot(history_df.episode, history_df.total_reward, alpha=.35); plt.plot(history_df.episode, history_df.total_reward.rolling(50, min_periods=1).mean(), label="media móvil (50)"); plt.xlabel("Episodio"); plt.ylabel("Recompensa total"); plt.title("Recompensa por episodio"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso24_reward_por_episodio.png", dpi=150); plt.close()
    plt.figure(figsize=(9,4)); plt.plot(history_df.episode, history_df.steps, alpha=.35); plt.plot(history_df.episode, history_df.steps.rolling(50, min_periods=1).mean(), label="media móvil (50)"); plt.xlabel("Episodio"); plt.ylabel("Pasos"); plt.title("Pasos por episodio"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso24_steps_por_episodio.png", dpi=150); plt.close()
    plt.figure(figsize=(9,4)); plt.plot(history_df.episode, history_df.epsilon); plt.xlabel("Episodio"); plt.ylabel("Epsilon"); plt.title("Descenso de epsilon"); plt.tight_layout(); plt.savefig(figures_dir / "paso24_epsilon_decay.png", dpi=150); plt.close()
    arrows={"up":"↑","right":"→","down":"↓","left":"←"}; plt.figure(figsize=(6,6)); plt.xlim(-.5,3.5); plt.ylim(3.5,-.5); plt.xticks(range(4)); plt.yticks(range(4)); plt.grid(True)
    for _, item in policy_df.iterrows():
        label="S" if item.state == START else ("G" if item.state == GOAL else arrows[item.best_action_name]); plt.text(item.col, item.row, label, ha="center", va="center", fontsize=18)
    plt.title("Política final Q-learning"); plt.tight_layout(); plt.savefig(figures_dir / "paso24_politica_grid.png", dpi=150); plt.close()
    pd.DataFrame([{ "n_states":N_STATES, "n_actions":N_ACTIONS, "episodes":episodes, "alpha":alpha, "gamma":gamma, "epsilon_start":epsilon_start, "epsilon_min":epsilon_min, "epsilon_decay":epsilon_decay, "success_rate_eval":evaluation["success_rate"], "mean_steps_eval":evaluation["mean_steps_success"], "mean_reward_eval":evaluation["mean_reward"], "final_epsilon":epsilon, "random_state":random_state }]).to_csv(metrics_dir / "paso24_resumen.csv", index=False)
    print(f"Grid {N}x{N}; episodios={episodes}; epsilon_final={epsilon:.6f}; success_rate={evaluation['success_rate']:.3f}; mean_steps={evaluation['mean_steps_success']:.2f}; mean_reward={evaluation['mean_reward']:.2f}"); print(pd.DataFrame(trajectory).to_string(index=False)); print(policy_df.to_string(index=False))

if __name__ == "__main__": main()
