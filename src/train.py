import os
import sys
import numpy as np
import torch

# Allow importing files from the same src directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import RANSlicingEnvironment
from ppo_agent import PPOAgent


def train_ppo(
    num_episodes=10,
    episode_length=200,
    update_interval=200
):
    """
    Train PPO agent on the 5G RAN slicing environment.
    """

    print("=" * 70)
    print("PPO TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Create environment
    # ---------------------------------------------------------
    env = RANSlicingEnvironment(
        episode_length=episode_length
    )

    # ---------------------------------------------------------
    # 2. Create PPO agent
    # ---------------------------------------------------------
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
        update_epochs=10
    )

    # ---------------------------------------------------------
    # 3. Storage for PPO rollout
    # ---------------------------------------------------------
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []

    # ---------------------------------------------------------
    # 4. Training statistics
    # ---------------------------------------------------------
    episode_rewards = []
    episode_delays = []
    episode_cvars = []
    episode_utilizations = []
    episode_fairness = []

    total_steps = 0

    # ---------------------------------------------------------
    # 5. Training loop
    # ---------------------------------------------------------
    for episode in range(1, num_episodes + 1):

        state = env.reset(seed=episode)

        episode_reward = 0.0
        episode_delay = []
        episode_cvar = []
        episode_utilization = []
        episode_fairness_values = []

        done = False

        while not done:

            # -------------------------------------------------
            # Select action
            # -------------------------------------------------
            action, log_prob, value = agent.select_action(state)

            # -------------------------------------------------
            # Environment step
            # -------------------------------------------------
            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            done = terminated or truncated

            # -------------------------------------------------
            # Store transition
            # -------------------------------------------------
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------
            episode_reward += reward

            episode_delay.append(info["delay"])
            episode_cvar.append(info["CVaR"])
            episode_utilization.append(info["utilization"])
            episode_fairness_values.append(info["fairness"])

            state = next_state

            total_steps += 1

            # -------------------------------------------------
            # PPO update
            # -------------------------------------------------
            if len(states) >= update_interval or done:

                # Get value of final state
                if done:
                    next_value = 0.0
                else:
                    _, _, next_value = agent.select_action(state)

                # Compute GAE
                advantages, returns = agent.compute_gae(
                    rewards,
                    values,
                    dones,
                    next_value
                )

                # Update PPO
                policy_loss, value_loss, entropy = agent.update(
                    states,
                    actions,
                    log_probs,
                    returns,
                    advantages
                )

                # Clear rollout buffer
                states.clear()
                actions.clear()
                log_probs.clear()
                rewards.clear()
                values.clear()
                dones.clear()

        # -----------------------------------------------------
        # Episode statistics
        # -----------------------------------------------------
        avg_delay = float(np.mean(episode_delay))
        avg_cvar = float(np.mean(episode_cvar))
        avg_utilization = float(np.mean(episode_utilization))
        avg_fairness = float(np.mean(episode_fairness_values))

        episode_rewards.append(episode_reward)
        episode_delays.append(avg_delay)
        episode_cvars.append(avg_cvar)
        episode_utilizations.append(avg_utilization)
        episode_fairness.append(avg_fairness)

        # -----------------------------------------------------
        # Print episode result
        # -----------------------------------------------------
        print(
            f"Episode {episode:03d} | "
            f"Reward: {episode_reward:10.3f} | "
            f"Delay: {avg_delay:8.3f} | "
            f"CVaR: {avg_cvar:8.3f} | "
            f"Utilization: {avg_utilization:.3f} | "
            f"Fairness: {avg_fairness:.3f}"
        )

    # ---------------------------------------------------------
    # 6. Training summary
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Total steps: {total_steps}")

    print(
        f"Average episode reward: "
        f"{np.mean(episode_rewards):.4f}"
    )

    print(
        f"Average delay: "
        f"{np.mean(episode_delays):.4f}"
    )

    print(
        f"Average CVaR: "
        f"{np.mean(episode_cvars):.4f}"
    )

    print(
        f"Average utilization: "
        f"{np.mean(episode_utilizations):.4f}"
    )

    print(
        f"Average fairness: "
        f"{np.mean(episode_fairness):.4f}"
    )

    # ---------------------------------------------------------
    # 7. Save model
    # ---------------------------------------------------------
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_dir = os.path.join(
        project_root,
        "models"
    )

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        "ppo_initial_test.pth"
    )

    agent.save(model_path)

    print()
    print(f"Model saved to:")
    print(model_path)

    print("=" * 70)

    return agent


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    train_ppo(
        num_episodes=10,
        episode_length=200,
        update_interval=200
    )