import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet

from policy import DirichletPolicy


class ValueNetwork(nn.Module):
    """
    Critic network.

    Estimates V(s), the expected future reward
    from a given state.
    """

    def __init__(
        self,
        state_dim=8,
        hidden_dim=128
    ):

        super().__init__()

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

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(self, state):

        return self.network(
            state
        ).squeeze(-1)


class PPOAgent:

    def __init__(
        self,
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
    ):

        self.state_dim = state_dim

        self.action_dim = action_dim

        self.gamma = gamma

        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon

        self.value_coefficient = (
            value_coefficient
        )

        self.entropy_coefficient = (
            entropy_coefficient
        )

        self.update_epochs = update_epochs

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"PPO device: {self.device}"
        )

        # --------------------------------------------------
        # Actor
        # --------------------------------------------------

        self.policy = DirichletPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # --------------------------------------------------
        # Critic
        # --------------------------------------------------

        self.value_network = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # --------------------------------------------------
        # Optimizers
        # --------------------------------------------------

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate
        )

    # ======================================================
    # SELECT ACTION
    # ======================================================

    def select_action(
        self,
        state
    ):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            distribution = (
                self.policy.get_distribution(
                    state_tensor
                )
            )

            action = (
                distribution.sample()
            )

            log_probability = (
                distribution.log_prob(
                    action
                )
            )

            value = (
                self.value_network(
                    state_tensor
                )
            )

        action = action.squeeze(0).cpu().numpy()

        log_probability = (
            log_probability.item()
        )

        value = value.item()

        return (
            action,
            log_probability,
            value
        )

    # ======================================================
    # COMPUTE GAE
    # ======================================================

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        next_value
    ):

        advantages = np.zeros(
            len(rewards),
            dtype=np.float32
        )

        gae = 0.0

        for step in reversed(
            range(len(rewards))
        ):

            if step == len(rewards) - 1:

                next_non_terminal = (
                    1.0 - dones[step]
                )

                next_value_step = (
                    next_value
                )

            else:

                next_non_terminal = (
                    1.0 - dones[step + 1]
                )

                next_value_step = (
                    values[step + 1]
                )

            delta = (
                rewards[step]
                +
                self.gamma
                *
                next_value_step
                *
                next_non_terminal
                -
                values[step]
            )

            gae = (
                delta
                +
                self.gamma
                *
                self.gae_lambda
                *
                next_non_terminal
                *
                gae
            )

            advantages[step] = gae

        returns = (
            advantages
            +
            np.asarray(
                values,
                dtype=np.float32
            )
        )

        return (
            advantages,
            returns
        )

    # ======================================================
    # UPDATE
    # ======================================================

    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages
    ):

        states = torch.tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            np.asarray(actions),
            dtype=torch.float32,
            device=self.device
        )

        old_log_probs = torch.tensor(
            np.asarray(old_log_probs),
            dtype=torch.float32,
            device=self.device
        )

        returns = torch.tensor(
            np.asarray(returns),
            dtype=torch.float32,
            device=self.device
        )

        advantages = torch.tensor(
            np.asarray(advantages),
            dtype=torch.float32,
            device=self.device
        )

        # Normalize advantages

        advantages = (
            advantages
            -
            advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        policy_losses = []

        value_losses = []

        entropy_values = []

        for _ in range(
            self.update_epochs
        ):

            # --------------------------------------------------
            # Actor
            # --------------------------------------------------

            distribution = (
                self.policy.get_distribution(
                    states
                )
            )

            new_log_probs = (
                distribution.log_prob(
                    actions
                )
            )

            entropy = (
                distribution.entropy()
            )

            # --------------------------------------------------
            # PPO ratio
            # --------------------------------------------------

            ratios = torch.exp(
                new_log_probs
                -
                old_log_probs
            )

            # --------------------------------------------------
            # Clipped objective
            # --------------------------------------------------

            unclipped = (
                ratios
                *
                advantages
            )

            clipped = (
                torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                )
                *
                advantages
            )

            policy_loss = -torch.min(
                unclipped,
                clipped
            ).mean()

            # --------------------------------------------------
            # Entropy bonus
            # --------------------------------------------------

            entropy_bonus = (
                entropy.mean()
            )

            total_policy_loss = (
                policy_loss
                -
                self.entropy_coefficient
                *
                entropy_bonus
            )

            # --------------------------------------------------
            # Update actor
            # --------------------------------------------------

            self.policy_optimizer.zero_grad()

            total_policy_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=0.5
            )

            self.policy_optimizer.step()

            # --------------------------------------------------
            # Critic
            # --------------------------------------------------

            predicted_values = (
                self.value_network(
                    states
                )
            )

            value_loss = (
                (predicted_values - returns) ** 2
            ).mean()

            total_value_loss = (
                self.value_coefficient
                *
                value_loss
            )

            self.value_optimizer.zero_grad()

            total_value_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                max_norm=0.5
            )

            self.value_optimizer.step()

            # --------------------------------------------------
            # Store metrics
            # --------------------------------------------------

            policy_losses.append(
                policy_loss.item()
            )

            value_losses.append(
                value_loss.item()
            )

            entropy_values.append(
                entropy_bonus.item()
            )

        return {

            "policy_loss":
                float(np.mean(policy_losses)),

            "value_loss":
                float(np.mean(value_losses)),

            "entropy":
                float(np.mean(entropy_values))
        }

    # ======================================================
    # SAVE
    # ======================================================

    def save(
        self,
        path
    ):

        torch.save(
            {
                "policy":
                    self.policy.state_dict(),

                "value_network":
                    self.value_network.state_dict(),

                "policy_optimizer":
                    self.policy_optimizer.state_dict(),

                "value_optimizer":
                    self.value_optimizer.state_dict()
            },
            path
        )

        print(
            f"Model saved to: {path}"
        )

    # ======================================================
    # LOAD
    # ======================================================

    def load(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.policy.load_state_dict(
            checkpoint["policy"]
        )

        self.value_network.load_state_dict(
            checkpoint["value_network"]
        )

        self.policy_optimizer.load_state_dict(
            checkpoint["policy_optimizer"]
        )

        self.value_optimizer.load_state_dict(
            checkpoint["value_optimizer"]
        )

        print(
            f"Model loaded from: {path}"
        )


# ==========================================================
# TEST PPO AGENT
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PPO AGENT TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Create agent
    # ------------------------------------------------------

    agent = PPOAgent(
        state_dim=8,
        action_dim=3
    )

    # ------------------------------------------------------
    # Example state
    # ------------------------------------------------------

    state = np.array(
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
        dtype=np.float32
    )

    # ------------------------------------------------------
    # Select action
    # ------------------------------------------------------

    (
        action,
        log_probability,
        value
    ) = agent.select_action(
        state
    )

    print()
    print("State:")
    print(state)

    print()
    print("Action:")
    print(action)

    print()
    print("Action sum:")
    print(np.sum(action))

    print()
    print("Log probability:")
    print(log_probability)

    print()
    print("State value:")
    print(value)

    print()
    print("=" * 70)
    print("PPO AGENT TEST COMPLETED")
    print("=" * 70)