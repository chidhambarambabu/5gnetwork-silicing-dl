import os
import sys
import numpy as np
import pandas as pd
import torch

# Add src folder to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import RANSlicingEnvironment
from ppo_agent import PPOAgent


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/ppo_initial_test.pth"

NUM_EPISODES = 10
MAX_STEPS = 200
SEED = 42

# 95% CVaR: average of the worst 5% delay observations
CVAR_CONFIDENCE = 0.95


# ============================================================
# CVaR CALCULATION
# ============================================================

def calculate_cvar(delays, confidence=0.95):
    """
    Calculate empirical upper-tail CVaR.

    VaR is the confidence-level quantile.
    CVaR is the mean delay at or above VaR.

    Higher CVaR means worse tail-delay risk.
    """

    delays = np.asarray(delays, dtype=np.float64)
    delays = delays[np.isfinite(delays)]

    if len(delays) == 0:
        return np.nan

    var_value = np.quantile(delays, confidence)

    tail_delays = delays[delays >= var_value]

    if len(tail_delays) == 0:
        return float(var_value)

    return float(np.mean(tail_delays))


# ============================================================
# SAFE METRIC EXTRACTION
# ============================================================

def get_metric(info, key):
    """
    Extract a metric from info.
    Return NaN if the key is missing or invalid.
    """

    value = info.get(key, np.nan)

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\nStarting PPO evaluation...\n")

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Check whether the PPO model was saved correctly.")
        return

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = RANSlicingEnvironment()

    # --------------------------------------------------------
    # Create PPO agent
    # --------------------------------------------------------

    agent = PPOAgent(
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
    )

    # --------------------------------------------------------
    # Load trained PPO model
    # --------------------------------------------------------

    agent.load(MODEL_PATH)
    agent.policy.eval()

    print(f"\nLoaded model: {MODEL_PATH}")
    print(f"Number of evaluation episodes: {NUM_EPISODES}")
    print(f"Maximum steps per episode: {MAX_STEPS}")
    print("\nBeginning evaluation...\n")

    all_results = []

    # ========================================================
    # EPISODE LOOP
    # ========================================================

    for episode in range(NUM_EPISODES):

        # Reset environment
        reset_result = env.reset(seed=SEED + episode)

        # Support both reset return formats
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result

        # Store episode metrics
        delays = []
        cvars = []
        utilizations = []
        fairness_values = []

        # ----------------------------------------------------
        # STEP LOOP
        # ----------------------------------------------------

        for step in range(MAX_STEPS):

            # Convert state to tensor
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            # ------------------------------------------------
            # Deterministic PPO action
            # ------------------------------------------------

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

            # Ensure valid resource allocation
            action = np.clip(action, 1e-6, None)
            action = action / np.sum(action)

            # ------------------------------------------------
            # Take environment step
            # ------------------------------------------------

            step_result = env.step(action)

            # Support Gymnasium 5-value API
            if len(step_result) == 5:

                next_state, reward, terminated, truncated, info = (
                    step_result
                )

                done = terminated or truncated

            # Support 4-value API
            else:

                next_state, reward, done, info = step_result

            # ------------------------------------------------
            # Debug first step only
            # ------------------------------------------------

            if episode == 0 and step == 0:

                print("DEBUG info keys:", list(info.keys()))
                print("DEBUG info values:", info)

            # ------------------------------------------------
            # Collect metrics
            # ------------------------------------------------

            delay = get_metric(info, "delay")
            cvar = get_metric(info, "cvar")
            utilization = get_metric(info, "utilization")
            fairness = get_metric(info, "fairness")

            delays.append(delay)
            cvars.append(cvar)
            utilizations.append(utilization)
            fairness_values.append(fairness)

            # Update state
            state = next_state

            if done:
                break

        # ====================================================
        # EPISODE METRICS
        # ====================================================

        # Calculate episode CVaR from delay history.
        # This avoids NaN when the environment doesn't return
        # a usable "cvar" field in info.

        episode_cvar = calculate_cvar(
            delays,
            confidence=CVAR_CONFIDENCE
        )

        # Prefer environment-provided final CVaR if available
        valid_cvars = [
            value for value in cvars
            if np.isfinite(value)
        ]

        if valid_cvars:
            final_cvar = valid_cvars[-1]
        else:
            final_cvar = episode_cvar

        episode_result = {
            "episode": episode + 1,
            "steps": len(delays),
            "mean_delay": np.nanmean(delays),
            "episode_cvar_95": episode_cvar,
            "final_cvar": final_cvar,
            "mean_utilization": np.nanmean(utilizations),
            "mean_fairness": np.nanmean(fairness_values),
        }

        all_results.append(episode_result)

        # ----------------------------------------------------
        # Print episode results
        # ----------------------------------------------------

        print(
            f"Episode {episode + 1}/{NUM_EPISODES} | "
            f"Steps: {episode_result['steps']} | "
            f"Mean Delay: {episode_result['mean_delay']:.4f} | "
            f"Episode CVaR95: {episode_cvar:.4f} | "
            f"Final CVaR: {final_cvar:.4f} | "
            f"Utilization: "
            f"{episode_result['mean_utilization']:.4f} | "
            f"Fairness: "
            f"{episode_result['mean_fairness']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    output_path = "ppo_evaluation_results.csv"

    results_df.to_csv(output_path, index=False)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 65)
    print("PPO EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Number of episodes: {NUM_EPISODES}")

    print(
        f"Mean Delay: "
        f"{results_df['mean_delay'].mean():.4f}"
    )

    print(
        f"Mean Episode CVaR95: "
        f"{results_df['episode_cvar_95'].mean():.4f}"
    )

    print(
        f"Mean Utilization: "
        f"{results_df['mean_utilization'].mean():.4f}"
    )

    print(
        f"Mean Fairness: "
        f"{results_df['mean_fairness'].mean():.4f}"
    )

    print("=" * 65)

    print(f"\nResults saved to: {output_path}")
    print("\nPPO evaluation completed successfully!")

    # Close environment if supported
    if hasattr(env, "close"):
        env.close()


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()