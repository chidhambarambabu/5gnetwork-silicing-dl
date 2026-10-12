from environment import RANSlicingEnvironment
import numpy as np


# Create environment
env = RANSlicingEnvironment(episode_length=20)

# Reset environment
state = env.reset(seed=42)

done = False
step_count = 0

while not done:

    # Equal allocation baseline
    action = np.array(
        [1 / 3, 1 / 3, 1 / 3],
        dtype=np.float32
    )

    state, reward, terminated, truncated, info = env.step(action)

    step_count += 1
    done = terminated or truncated

    print(
        f"Step: {step_count:02d} | "
        f"Delay: {info['delay']:.4f} | "
        f"CVaR: {info['CVaR']:.4f} | "
        f"Violation: {info['CVaR_violation']:.4f}"
    )


print("\nEpisode test completed.")
print("Total steps:", step_count)
print("Risk history length:", len(env.risk_history))
print("Final CVaR:", info["CVaR"])
print("Constraint satisfied:", info["constraint_satisfied"])