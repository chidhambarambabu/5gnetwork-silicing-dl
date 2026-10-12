import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class DirichletPolicy(nn.Module):
    """
    Dirichlet policy network for 3-slice
    5G RAN resource allocation.

    Input:
        8-dimensional state

    Output:
        3-dimensional allocation vector

        [eMBB, URLLC, mMTC]

    The output always satisfies:

        allocation[i] > 0
        sum(allocation) = 1
    """

    def __init__(
        self,
        state_dim=8,
        action_dim=3,
        hidden_dim=128
    ):

        super().__init__()

        self.state_dim = state_dim

        self.action_dim = action_dim

        # --------------------------------------------------
        # Neural network
        # --------------------------------------------------

        self.network = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh()
        )

        # --------------------------------------------------
        # Dirichlet concentration parameters
        # --------------------------------------------------

        self.alpha_layer = nn.Linear(
            hidden_dim,
            action_dim
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(self, state):

        features = self.network(
            state
        )

        # Raw values
        raw_alpha = self.alpha_layer(
            features
        )

        # Dirichlet concentration parameters
        #
        # softplus guarantees positive values.

        alpha = torch.nn.functional.softplus(
            raw_alpha
        ) + 1e-3

        return alpha

    # ======================================================
    # DISTRIBUTION
    # ======================================================

    def get_distribution(self, state):

        alpha = self.forward(
            state
        )

        distribution = Dirichlet(
            alpha
        )

        return distribution

    # ======================================================
    # SAMPLE ACTION
    # ======================================================

    def sample_action(self, state):

        distribution = self.get_distribution(
            state
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action,
            log_probability,
            distribution
        )

    # ======================================================
    # DETERMINISTIC ACTION
    # ======================================================

    def deterministic_action(self, state):

        alpha = self.forward(
            state
        )

        # Mean of Dirichlet distribution:
        #
        # alpha_i / sum(alpha)

        action = (
            alpha /
            torch.sum(
                alpha,
                dim=-1,
                keepdim=True
            )
        )

        return action


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DIRICHLET POLICY TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Create policy
    # ------------------------------------------------------

    policy = DirichletPolicy(
        state_dim=8,
        action_dim=3,
        hidden_dim=128
    )

    print()
    print("Policy created successfully.")

    # ------------------------------------------------------
    # Example state
    # ------------------------------------------------------

    state = torch.tensor(
        [
            -0.011,
            -0.014,
            -0.014,

            -0.007,
            -0.016,
            -0.015,

            -0.354,
            -0.653
        ],
        dtype=torch.float32
    )

    # Add batch dimension

    state = state.unsqueeze(0)

    print()
    print("State:")
    print(state)

    print()
    print("State shape:")
    print(state.shape)

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------

    alpha = policy(
        state
    )

    print()
    print("Dirichlet alpha parameters:")
    print(alpha)

    # ------------------------------------------------------
    # Sample action
    # ------------------------------------------------------

    (
        action,
        log_probability,
        distribution
    ) = policy.sample_action(
        state
    )

    print()
    print("Sampled action:")
    print(action)

    print()
    print("Action shape:")
    print(action.shape)

    print()
    print("Action sum:")
    print(
        torch.sum(
            action,
            dim=-1
        )
    )

    print()
    print("Log probability:")
    print(log_probability)

    # ------------------------------------------------------
    # Deterministic action
    # ------------------------------------------------------

    deterministic = (
        policy.deterministic_action(
            state
        )
    )

    print()
    print("Deterministic action:")
    print(deterministic)

    print()
    print("Deterministic action sum:")

    print(
        torch.sum(
            deterministic,
            dim=-1
        )
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    action_sum = torch.sum(
        action,
        dim=-1
    ).item()

    if abs(action_sum - 1.0) < 1e-6:

        print()
        print(
            "✓ Action sums to 1."
        )

    else:

        print()
        print(
            "✗ Action does not sum to 1."
        )

    if torch.all(action > 0):

        print(
            "✓ All allocations are positive."
        )

    else:

        print(
            "✗ Invalid negative allocation."
        )

    print()
    print("=" * 70)
    print("DIRICHLET POLICY TEST COMPLETED")
    print("=" * 70)