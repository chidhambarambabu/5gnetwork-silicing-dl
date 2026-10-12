import os
import sys
import numpy as np
import pandas as pd
import torch

# Add src folder to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import RANSlicingEnvironment
from cvar_ppo_agent import CVaRPPOAgent


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/cvar_ppo_initial_test.pth"

NUM_EPISODES = 10
MAX_STEPS = 200
SEED = 42

CVAR_CONFIDENCE = 0.95
CVAR_THRESHOLD = 18.0


# ============================================================
# CVaR CALCULATION
# ============================================================

def calculate_cvar(delays, confidence=0.95):
    """
    Calculate empirical upper-tail CVaR.

    CVaR95 is the average of delay observations
    in the worst 5% tail.
    """

    delays = np.asarray(delays, dtype=np.float64)
    delays = delays[np.isfinite(delays)]

    if len(delays) == 0:
        return np.nan

    var_value = np.quantile(delays, confidence)

    # Include observations at or above the 95th percentile
    tail = delays[delays >= var_value]

    if len(tail) == 0:
        return float(var_value)

    return float(np.mean(tail))


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\nStarting CVaR-PPO evaluation...\n")

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = RANSlicingEnvironment()

    # --------------------------------------------------------
    # Create CVaR-PPO agent
    # Use the same architecture and hyperparameters
    # as the trained model.
    # --------------------------------------------------------

    agent = CVaRPPOAgent(
        state_dim=8,
        action_dim=3,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        update_epochs=10,
        cvar_alpha=0.05,
        cvar_threshold=CVAR_THRESHOLD,
        lambda_learning_rate=0.01,
        initial_lambda=1.0,
        max_lambda=100.0
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    agent.load(MODEL_PATH)
    agent.policy.eval()

    print(f"\nLoaded model: {MODEL_PATH}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Maximum steps per episode: {MAX_STEPS}")
    print(f"CVaR threshold: {CVAR_THRESHOLD}")

    all_results = []

    # ========================================================
    # EPISODE LOOP
    # ========================================================

    for episode in range(NUM_EPISODES):

        reset_result = env.reset(seed=SEED + episode)

        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result

        delays = []
        utilizations = []
        fairness_values = []
        constraint_flags = []

        # ----------------------------------------------------
        # STEP LOOP
        # ----------------------------------------------------

        for step in range(MAX_STEPS):

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=agent.device
            ).unsqueeze(0)

            # Deterministic action from the learned policy
            with torch.no_grad():

                distribution = agent.policy.get_distribution(
                    state_tensor
                )

                action = (
                    distribution.mean
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

            # Normalize allocation
            action = np.clip(action, 1e-6, None)
            action = action / action.sum()

            # Step environment
            step_result = env.step(action)

            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = (
                    step_result
                )
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            # Collect metrics
            delays.append(float(info.get("delay", np.nan)))

            utilizations.append(
                float(info.get("utilization", np.nan))
            )

            fairness_values.append(
                float(info.get("fairness", np.nan))
            )

            # Environment constraint flag
            constraint_flags.append(
                bool(info.get("constraint_satisfied", False))
            )

            state = next_state

            if done:
                break

        # ====================================================
        # EPISODE SUMMARY
        # ====================================================

        episode_cvar = calculate_cvar(
            delays,
            confidence=CVAR_CONFIDENCE
        )

        mean_delay = np.nanmean(delays)
        mean_utilization = np.nanmean(utilizations)
        mean_fairness = np.nanmean(fairness_values)

        # Episode-level CVaR threshold check
        cvar_satisfied = (
            episode_cvar <= CVAR_THRESHOLD
            if np.isfinite(episode_cvar)
            else False
        )

        # Count step-level constraint satisfaction
        step_satisfaction = (
            np.mean(constraint_flags)
            if constraint_flags
            else np.nan
        )

        episode_result = {
            "episode": episode + 1,
            "steps": len(delays),
            "mean_delay": mean_delay,
            "episode_cvar95": episode_cvar,
            "mean_utilization": mean_utilization,
            "mean_fairness": mean_fairness,
            "cvar_satisfied": cvar_satisfied,
            "step_constraint_satisfaction": step_satisfaction
        }

        all_results.append(episode_result)

        print(
            f"Episode {episode + 1}/{NUM_EPISODES} | "
            f"Steps: {len(delays)} | "
            f"Mean Delay: {mean_delay:.4f} | "
            f"CVaR95: {episode_cvar:.4f} | "
            f"Utilization: {mean_utilization:.4f} | "
            f"Fairness: {mean_fairness:.4f} | "
            f"CVaR Satisfied: {cvar_satisfied}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    output_path = "cvar_ppo_evaluation_results.csv"

    results_df.to_csv(output_path, index=False)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 65)
    print("CVaR-PPO EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Episodes evaluated: {NUM_EPISODES}")

    print(
        f"Mean Delay: "
        f"{results_df['mean_delay'].mean():.4f}"
    )

    print(
        f"Mean Episode CVaR95: "
        f"{results_df['episode_cvar95'].mean():.4f}"
    )

    print(
        f"Mean Utilization: "
        f"{results_df['mean_utilization'].mean():.4f}"
    )

    print(
        f"Mean Fairness: "
        f"{results_df['mean_fairness'].mean():.4f}"
    )

    satisfied_count = int(
        results_df["cvar_satisfied"].sum()
    )

    print(
        f"Episode CVaR Constraint Satisfaction: "
        f"{satisfied_count}/{NUM_EPISODES} "
        f"({100 * satisfied_count / NUM_EPISODES:.1f}%)"
    )

    print(
        f"Mean Step Constraint Satisfaction: "
        f"{100 * results_df['step_constraint_satisfaction'].mean():.2f}%"
    )

    print("=" * 65)

    print(f"\nResults saved to: {output_path}")
    print("\nCVaR-PPO evaluation completed!")

    if hasattr(env, "close"):
        env.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()