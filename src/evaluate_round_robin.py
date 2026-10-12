import numpy as np
import pandas as pd

from environment import RANSlicingEnvironment


NUM_EPISODES = 10
EPISODE_LENGTH = 200
CVAR_THRESHOLD = 18.0
CVAR_CONFIDENCE = 0.95


def calculate_episode_cvar(delays, confidence=0.95):
    """
    Calculate empirical CVaR at the given confidence level.

    For CVaR95, take the average of the highest 5% of delay
    observations in the episode.
    """
    delays = np.asarray(delays, dtype=np.float64)

    if len(delays) == 0:
        return 0.0

    var = np.quantile(delays, confidence)
    tail_delays = delays[delays >= var]

    return float(np.mean(tail_delays))


print("Starting Round Robin evaluation...", flush=True)

env = RANSlicingEnvironment(
    episode_length=EPISODE_LENGTH
)

episode_results = []

for episode in range(NUM_EPISODES):

    state = env.reset(seed=episode + 100)

    done = False
    delays = []
    utilizations = []
    fairness_scores = []

    while not done:

        # Equal allocation to all three slices
        action = np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=np.float32
        )

        state, reward, terminated, truncated, info = env.step(action)

        delays.append(info["delay"])
        utilizations.append(info["utilization"])
        fairness_scores.append(info["fairness"])

        done = terminated or truncated

    # Calculate CVaR95 from all delays in this episode
    episode_cvar95 = calculate_episode_cvar(
        delays,
        confidence=CVAR_CONFIDENCE
    )

    result = {
        "episode": episode + 1,
        "mean_delay": float(np.mean(delays)),
        "episode_cvar95": episode_cvar95,
        "mean_utilization": float(np.mean(utilizations)),
        "mean_fairness": float(np.mean(fairness_scores)),
        "constraint_satisfied": episode_cvar95 <= CVAR_THRESHOLD
    }

    episode_results.append(result)

    print(
        f"Episode {episode + 1:02d}/{NUM_EPISODES} | "
        f"Mean Delay: {result['mean_delay']:.4f} | "
        f"Episode CVaR95: {result['episode_cvar95']:.4f} | "
        f"Utilization: {result['mean_utilization']:.4f} | "
        f"Fairness: {result['mean_fairness']:.4f} | "
        f"Constraint: {result['constraint_satisfied']}",
        flush=True
    )


# Convert results to a DataFrame
results_df = pd.DataFrame(episode_results)

# Save individual episode results
results_df.to_csv(
    "round_robin_evaluation_results.csv",
    index=False
)


print("\n" + "=" * 60)
print("ROUND ROBIN BASELINE - SUMMARY")
print("=" * 60)

for metric in [
    "mean_delay",
    "episode_cvar95",
    "mean_utilization",
    "mean_fairness"
]:
    values = results_df[metric].to_numpy()

    print(
        f"{metric}: {np.mean(values):.4f} "
        f"+/- {np.std(values):.4f}"
    )


satisfaction_rate = results_df["constraint_satisfied"].mean()

print(
    f"CVaR95 constraint satisfaction: "
    f"{satisfaction_rate * 100:.1f}% "
    f"({int(results_df['constraint_satisfied'].sum())}/{NUM_EPISODES} episodes)"
)

print("\nSaved results to round_robin_evaluation_results.csv")
print("Round Robin evaluation completed.")