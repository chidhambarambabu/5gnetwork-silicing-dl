import os
import sys
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from environment import RANSlicingEnvironment
from cvar_ppo_agent import CVaRPPOAgent


def train_cvar_ppo(
    num_episodes=10,
    episode_length=200,
    update_interval=200
):

    print("=" * 70)
    print("CORRECTED CVaR-PPO TRAINING")
    print("=" * 70)

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = RANSlicingEnvironment(
        episode_length=episode_length
    )

    # ========================================================
    # CVaR-PPO AGENT
    # ========================================================

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

        # CVaR parameters
        cvar_alpha=0.05,
        cvar_threshold=18.0,

        # Lagrange multiplier
        lambda_learning_rate=0.01,
        initial_lambda=1.0,
        max_lambda=100.0
    )

    # ========================================================
    # ROLLOUT STORAGE
    # ========================================================

    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    risks = []

    # ========================================================
    # STATISTICS
    # ========================================================

    episode_rewards = []
    episode_delays = []
    episode_cvars = []
    episode_vars = []
    episode_utilizations = []
    episode_fairness = []
    episode_lambdas = []
    episode_violations = []

    total_steps = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(
        1,
        num_episodes + 1
    ):

        state = env.reset(
            seed=episode
        )

        done = False

        episode_reward = 0.0

        episode_delay = []
        episode_cvar = []
        episode_utilization = []
        episode_fairness_values = []

        while not done:

            # ------------------------------------------------
            # Select action
            # ------------------------------------------------

            action, log_prob, value = (
                agent.select_action(
                    state
                )
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = (
                terminated
                or truncated
            )

            # ------------------------------------------------
            # Store transition
            # ------------------------------------------------

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            # Delay is the risk variable
            risks.append(
                info["delay"]
            )

            # ------------------------------------------------
            # Episode statistics
            # ------------------------------------------------

            episode_reward += reward

            episode_delay.append(
                info["delay"]
            )

            episode_cvar.append(
                info["CVaR"]
            )

            episode_utilization.append(
                info["utilization"]
            )

            episode_fairness_values.append(
                info["fairness"]
            )

            state = next_state

            total_steps += 1

            # ------------------------------------------------
            # PPO UPDATE
            # ------------------------------------------------

            if (
                len(states) >= update_interval
                or done
            ):

                # Value of next state
                if done:

                    next_value = 0.0

                else:

                    _, _, next_value = (
                        agent.select_action(
                            state
                        )
                    )

                # ------------------------------------------------
                # GAE
                # ------------------------------------------------

                advantages, returns = (
                    agent.compute_gae(
                        rewards,
                        values,
                        dones,
                        next_value
                    )
                )

                # ------------------------------------------------
                # CVaR-PPO UPDATE
                # ------------------------------------------------

                update_info = agent.update(
                    states,
                    actions,
                    log_probs,
                    returns,
                    advantages,
                    risks
                )

                # ------------------------------------------------
                # Clear rollout
                # ------------------------------------------------

                states.clear()
                actions.clear()
                log_probs.clear()
                rewards.clear()
                values.clear()
                dones.clear()
                risks.clear()

        # ====================================================
        # EPISODE STATISTICS
        # ====================================================

        avg_delay = float(
            np.mean(
                episode_delay
            )
        )

        avg_environment_cvar = float(
            np.mean(
                episode_cvar
            )
        )

        avg_utilization = float(
            np.mean(
                episode_utilization
            )
        )

        avg_fairness = float(
            np.mean(
                episode_fairness_values
            )
        )

        # ----------------------------------------------------
        # CVaR from episode delays
        # ----------------------------------------------------

        episode_cvar_value = (
            agent.calculate_cvar(
                np.asarray(
                    episode_delay
                )
            )
        )

        episode_var_value = (
            agent.calculate_var(
                np.asarray(
                    episode_delay
                )
            )
        )

        violation = max(
            0.0,
            episode_cvar_value
            - agent.cvar_threshold
        )

        # ====================================================
        # Store statistics
        # ====================================================

        episode_rewards.append(
            episode_reward
        )

        episode_delays.append(
            avg_delay
        )

        episode_cvars.append(
            episode_cvar_value
        )

        episode_vars.append(
            episode_var_value
        )

        episode_utilizations.append(
            avg_utilization
        )

        episode_fairness.append(
            avg_fairness
        )

        episode_lambdas.append(
            agent.lambda_value
        )

        episode_violations.append(
            violation
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Episode {episode:03d} | "
            f"Reward: {episode_reward:10.3f} | "
            f"Delay: {avg_delay:8.3f} | "
            f"VaR: {episode_var_value:8.3f} | "
            f"CVaR: {episode_cvar_value:8.3f} | "
            f"Lambda: {agent.lambda_value:8.3f} | "
            f"Violation: {violation:8.3f} | "
            f"Util: {avg_utilization:.3f} | "
            f"Fairness: {avg_fairness:.3f}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CORRECTED CVaR-PPO TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Total steps: {total_steps}"
    )

    print(
        f"Average episode reward: "
        f"{np.mean(episode_rewards):.4f}"
    )

    print(
        f"Average delay: "
        f"{np.mean(episode_delays):.4f}"
    )

    print(
        f"Average VaR: "
        f"{np.mean(episode_vars):.4f}"
    )

    print(
        f"Average CVaR: "
        f"{np.mean(episode_cvars):.4f}"
    )

    print(
        f"CVaR threshold: "
        f"{agent.cvar_threshold:.4f}"
    )

    print(
        f"Final lambda: "
        f"{agent.lambda_value:.4f}"
    )

    print(
        f"Average CVaR violation: "
        f"{np.mean(episode_violations):.4f}"
    )

    print(
        f"Average utilization: "
        f"{np.mean(episode_utilizations):.4f}"
    )

    print(
        f"Average fairness: "
        f"{np.mean(episode_fairness):.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    model_dir = os.path.join(
        project_root,
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        model_dir,
        "cvar_ppo_initial_test.pth"
    )

    agent.save(
        model_path
    )

    print()
    print("Model saved to:")
    print(model_path)

    print("=" * 70)

    return agent


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_cvar_ppo(
    num_episodes=100,
    episode_length=200,
    update_interval=200
)