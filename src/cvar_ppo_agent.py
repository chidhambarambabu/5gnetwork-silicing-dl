import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from policy import DirichletPolicy


# ============================================================
# VALUE NETWORK
# ============================================================

class ValueNetwork(nn.Module):

    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)


# ============================================================
# CVaR-PPO AGENT
# ============================================================

class CVaRPPOAgent:

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
        update_epochs=10,
        cvar_alpha=0.05,
        cvar_threshold=18.0,
        lambda_learning_rate=0.01,
        initial_lambda=1.0,
        max_lambda=100.0
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

        self.update_epochs = update_epochs

        # CVaR parameters
        self.cvar_alpha = cvar_alpha
        self.cvar_threshold = cvar_threshold

        # Lagrange multiplier
        self.lambda_learning_rate = lambda_learning_rate
        self.lambda_value = initial_lambda
        self.max_lambda = max_lambda

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"CVaR-PPO device: {self.device}")

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------

        self.policy = DirichletPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # ----------------------------------------------------
        # Critic
        # ----------------------------------------------------

        self.value_network = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # ----------------------------------------------------
        # Optimizers
        # ----------------------------------------------------

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate
        )

    # ========================================================
    # SELECT ACTION
    # ========================================================

    def select_action(self, state):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            distribution = self.policy.get_distribution(
                state_tensor
            )

            action = distribution.sample()

            log_probability = distribution.log_prob(
                action
            )

            value = self.value_network(
                state_tensor
            )

        return (
            action.squeeze(0).cpu().numpy(),
            log_probability.item(),
            value.item()
        )

    # ========================================================
    # GAE
    # ========================================================

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        next_value
    ):

        rewards = np.asarray(
            rewards,
            dtype=np.float32
        )

        values = np.asarray(
            values,
            dtype=np.float32
        )

        dones = np.asarray(
            dones,
            dtype=np.float32
        )

        advantages = np.zeros_like(
            rewards
        )

        gae = 0.0

        for step in reversed(
            range(len(rewards))
        ):

            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]

            not_done = 1.0 - dones[step]

            delta = (
                rewards[step]
                + self.gamma
                * next_val
                * not_done
                - values[step]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * not_done
                * gae
            )

            advantages[step] = gae

        returns = advantages + values

        return advantages, returns

    # ========================================================
    # CVaR CALCULATION
    # ========================================================

    def calculate_var(
        self,
        risks
    ):

        risks = np.asarray(
            risks,
            dtype=np.float32
        )

        if len(risks) == 0:
            return 0.0

        # Upper-tail VaR
        return float(
            np.quantile(
                risks,
                1.0 - self.cvar_alpha
            )
        )

    # ========================================================
    # CVaR USING ROCKAFELLAR-URYASEV FORMULATION
    # ========================================================

    def calculate_cvar(
        self,
        risks
    ):

        risks = np.asarray(
            risks,
            dtype=np.float32
        )

        if len(risks) == 0:
            return 0.0

        eta = self.calculate_var(
            risks
        )

        excess = np.maximum(
            risks - eta,
            0.0
        )

        cvar = (
            eta
            + np.mean(excess)
            / (1.0 - self.cvar_alpha)
        )

        return float(cvar)

    # ========================================================
    # TAIL RISK PENALTY
    # ========================================================

    def calculate_tail_penalty(
        self,
        risks
    ):

        risks = np.asarray(
            risks,
            dtype=np.float32
        )

        if len(risks) == 0:
            return np.zeros(0), 0.0

        eta = self.calculate_var(
            risks
        )

        excess = np.maximum(
            risks - eta,
            0.0
        )

        normalized_tail_cost = (
            excess
            / (1.0 - self.cvar_alpha)
        )

        return normalized_tail_cost, eta

    # ========================================================
    # UPDATE LAMBDA
    # ========================================================

    def update_lambda(
        self,
        cvar_value
    ):

        violation = (
            cvar_value
            - self.cvar_threshold
        )

        old_lambda = self.lambda_value

        self.lambda_value += (
            self.lambda_learning_rate
            * violation
        )

        self.lambda_value = max(
            0.0,
            self.lambda_value
        )

        self.lambda_value = min(
            self.max_lambda,
            self.lambda_value
        )

        return {
            "old_lambda": old_lambda,
            "new_lambda": self.lambda_value,
            "cvar_violation": violation
        }

    # ========================================================
    # CVaR-PPO UPDATE
    # ========================================================

    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
        risks
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

        # ----------------------------------------------------
        # Risk calculations
        # ----------------------------------------------------

        risks = np.asarray(
            risks,
            dtype=np.float32
        )

        cvar_value = self.calculate_cvar(
            risks
        )

        tail_penalty, var_value = (
            self.calculate_tail_penalty(
                risks
            )
        )

        # ----------------------------------------------------
        # Adaptive lambda
        # ----------------------------------------------------

        lambda_info = self.update_lambda(
            cvar_value
        )

        # ----------------------------------------------------
        # Convert tail penalty to tensor
        # ----------------------------------------------------

        tail_penalty_tensor = torch.tensor(
            tail_penalty,
            dtype=torch.float32,
            device=self.device
        )

        # ----------------------------------------------------
        # Risk-adjusted advantage
        # ----------------------------------------------------

        risk_adjusted_advantages = (
            advantages
            - self.lambda_value
            * tail_penalty_tensor
        )

        # Normalize risk-adjusted advantage
        risk_adjusted_advantages = (
            risk_adjusted_advantages
            - risk_adjusted_advantages.mean()
        ) / (
            risk_adjusted_advantages.std()
            + 1e-8
        )

        # ----------------------------------------------------
        # PPO update
        # ----------------------------------------------------

        policy_losses = []
        value_losses = []
        entropy_values = []

        for _ in range(
            self.update_epochs
        ):

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
                distribution.entropy().mean()
            )

            # PPO probability ratio
            ratios = torch.exp(
                new_log_probs
                - old_log_probs
            )

            # PPO clipped objective
            surr1 = (
                ratios
                * risk_adjusted_advantages
            )

            surr2 = (
                torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                )
                * risk_adjusted_advantages
            )

            policy_objective = torch.min(
                surr1,
                surr2
            ).mean()

            # Actor loss
            policy_loss = (
                -policy_objective
                - self.entropy_coefficient
                * entropy
            )

            # ------------------------------------------------
            # Critic
            # ------------------------------------------------

            predicted_values = (
                self.value_network(states)
            )

            value_loss = nn.functional.mse_loss(
                predicted_values,
                returns
            )

            # ------------------------------------------------
            # Actor update
            # ------------------------------------------------

            self.policy_optimizer.zero_grad()

            policy_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=0.5
            )

            self.policy_optimizer.step()

            # ------------------------------------------------
            # Critic update
            # ------------------------------------------------

            self.value_optimizer.zero_grad()

            critic_loss = (
                self.value_coefficient
                * value_loss
            )

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                max_norm=0.5
            )

            self.value_optimizer.step()

            policy_losses.append(
                policy_loss.item()
            )

            value_losses.append(
                value_loss.item()
            )

            entropy_values.append(
                entropy.item()
            )

        return {
            "policy_loss":
                float(np.mean(policy_losses)),

            "value_loss":
                float(np.mean(value_losses)),

            "entropy":
                float(np.mean(entropy_values)),

            "var":
                var_value,

            "cvar":
                cvar_value,

            "cvar_violation":
                max(
                    0.0,
                    cvar_value
                    - self.cvar_threshold
                ),

            "lambda":
                self.lambda_value
        }

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path
    ):

        checkpoint = {

            "policy_state_dict":
                self.policy.state_dict(),

            "value_state_dict":
                self.value_network.state_dict(),

            "policy_optimizer":
                self.policy_optimizer.state_dict(),

            "value_optimizer":
                self.value_optimizer.state_dict(),

            "lambda_value":
                self.lambda_value,

            "cvar_alpha":
                self.cvar_alpha,

            "cvar_threshold":
                self.cvar_threshold
        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"CVaR-PPO model saved to: {path}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.value_network.load_state_dict(
            checkpoint["value_state_dict"]
        )

        self.policy_optimizer.load_state_dict(
            checkpoint["policy_optimizer"]
        )

        self.value_optimizer.load_state_dict(
            checkpoint["value_optimizer"]
        )

        self.lambda_value = checkpoint.get(
            "lambda_value",
            self.lambda_value
        )

        self.cvar_alpha = checkpoint.get(
            "cvar_alpha",
            self.cvar_alpha
        )

        self.cvar_threshold = checkpoint.get(
            "cvar_threshold",
            self.cvar_threshold
        )

        print(
            f"CVaR-PPO model loaded from: {path}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CORRECTED CVaR-PPO AGENT TEST")
    print("=" * 70)

    agent = CVaRPPOAgent(
        state_dim=8,
        action_dim=3,
        hidden_dim=128,
        cvar_alpha=0.05,
        cvar_threshold=18.0,
        lambda_learning_rate=0.01,
        initial_lambda=1.0
    )

    state = np.array([
        -0.011,
        -0.014,
        -0.014,
        -0.007,
        -0.016,
        -0.015,
        -0.354,
        -0.653
    ], dtype=np.float32)

    # --------------------------------------------------------
    # Action test
    # --------------------------------------------------------

    action, log_probability, value = (
        agent.select_action(state)
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

    # --------------------------------------------------------
    # CVaR test
    # --------------------------------------------------------

    risks = np.array([
        5, 6, 7, 8, 8,
        9, 10, 10, 11, 12,
        12, 13, 14, 15, 16,
        17, 18, 20, 25, 40
    ], dtype=np.float32)

    var = agent.calculate_var(
        risks
    )

    cvar = agent.calculate_cvar(
        risks
    )

    tail_penalty, eta = (
        agent.calculate_tail_penalty(
            risks
        )
    )

    print()
    print("Risk samples:")
    print(risks)

    print()
    print("VaR:")
    print(var)

    print()
    print("CVaR:")
    print(cvar)

    print()
    print("CVaR threshold:")
    print(agent.cvar_threshold)

    print()
    print("Tail penalty:")
    print(tail_penalty)

    # --------------------------------------------------------
    # Lambda test
    # --------------------------------------------------------

    print()
    print("Initial lambda:")
    print(agent.lambda_value)

    lambda_info = agent.update_lambda(
        cvar
    )

    print()
    print("Lambda update:")
    print(lambda_info)

    print()
    print("Updated lambda:")
    print(agent.lambda_value)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert np.isclose(
        np.sum(action),
        1.0,
        atol=1e-5
    )

    assert np.all(
        action > 0
    )

    assert cvar > agent.cvar_threshold

    assert agent.lambda_value > 1.0

    assert np.any(
        tail_penalty > 0
    )

    print()
    print("=" * 70)
    print("CORRECTED CVaR-PPO TEST COMPLETED")
    print("=" * 70)