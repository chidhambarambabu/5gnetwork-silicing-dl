import numpy as np


def calculate_var(values, alpha=0.05):
    """
    Calculate VaR using the upper-tail quantile.

    alpha = 0.05 means the worst 5% of observations
    are considered the risk tail.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return 0.0

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    return float(np.quantile(values, 1.0 - alpha))


def calculate_cvar(values, alpha=0.05):
    """
    Calculate Rockafellar-Uryasev CVaR.

    CVaR = VaR + E[max(X - VaR, 0)] / (1 - alpha)

    This measures the expected risk in the upper tail.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return 0.0

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    var = calculate_var(values, alpha)

    excess = np.maximum(values - var, 0.0)

    cvar = var + np.mean(excess) / (1.0 - alpha)

    return float(cvar)


def calculate_cvar_violation(values, threshold, alpha=0.05):
    """
    Calculate how much CVaR exceeds the allowed threshold.
    """
    cvar = calculate_cvar(values, alpha)

    return max(0.0, cvar - threshold)


def is_constraint_satisfied(values, threshold, alpha=0.05):
    """
    Check whether the CVaR constraint is satisfied.
    """
    cvar = calculate_cvar(values, alpha)

    return cvar <= threshold


if __name__ == "__main__":

    print("=" * 70)
    print("CVaR MODULE TEST")
    print("=" * 70)

    delays = np.array([
        5, 6, 7, 8, 8,
        9, 10, 10, 11, 12,
        12, 13, 14, 15, 16,
        17, 18, 20, 25, 40
    ])

    alpha = 0.05
    threshold = 18.0

    var = calculate_var(delays, alpha)
    cvar = calculate_cvar(delays, alpha)
    violation = calculate_cvar_violation(
        delays,
        threshold,
        alpha
    )
    satisfied = is_constraint_satisfied(
        delays,
        threshold,
        alpha
    )

    print()
    print("Risk samples:")
    print(delays)

    print()
    print(f"Alpha: {alpha}")
    print(f"VaR: {var:.4f}")
    print(f"CVaR: {cvar:.4f}")
    print(f"CVaR threshold: {threshold:.4f}")
    print(f"CVaR violation: {violation:.4f}")
    print(f"Constraint satisfied: {satisfied}")

    print()
    print("=" * 70)
    print("CVaR MODULE TEST COMPLETED")
    print("=" * 70)