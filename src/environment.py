import numpy as np
import pandas as pd
from pathlib import Path


class RANSlicingEnvironment:

    def __init__(self, episode_length=200):

        # --------------------------------------------------
        # Project paths
        # --------------------------------------------------

        project_root = Path(__file__).resolve().parent.parent

        self.data_path = (
            project_root
            / "dataset"
            / "processed"
        )

        # --------------------------------------------------
        # Slice names
        # --------------------------------------------------

        self.slices = [
            "eMBB",
            "URLLC",
            "mMTC"
        ]

        self.num_slices = 3

        self.episode_length = episode_length

        # --------------------------------------------------
        # Load datasets
        # --------------------------------------------------

        self.data = {}

        for slice_name in self.slices:

            file_path = (
                self.data_path
                / f"{slice_name}_processed.csv"
            )

            print(f"Loading {slice_name}...")
            print(f"File: {file_path}")

            if not file_path.exists():

                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            self.data[slice_name] = (
                df.reset_index(drop=True)
            )

            print(
                f"{slice_name} rows: {len(df)}"
            )

        # --------------------------------------------------
        # Common number of samples
        # --------------------------------------------------

        self.num_samples = min(
            len(self.data[name])
            for name in self.slices
        )

        # --------------------------------------------------
        # Environment variables
        # --------------------------------------------------

        self.current_step = 0

        self.start_index = 0

        self.indices = None

        self.risk_history = []

        # Equal allocation initially
        self.previous_action = np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=np.float32
        )

        print()
        print("=" * 70)
        print("5G RAN SLICING ENVIRONMENT")
        print("=" * 70)

        print(
            f"Number of slices: {self.num_slices}"
        )

        print(
            f"Common samples available: "
            f"{self.num_samples}"
        )

        print(
            f"Episode length: "
            f"{self.episode_length}"
        )

        print("=" * 70)

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None):

        if seed is not None:

            np.random.seed(seed)

        self.current_step = 0

        # Make sure episode fits inside dataset

        if self.num_samples <= self.episode_length:

            self.start_index = 0

        else:

            self.start_index = np.random.randint(
                0,
                self.num_samples
                - self.episode_length
            )

        self.indices = np.arange(
            self.start_index,
            self.start_index
            + self.episode_length
        )

        self.previous_action = np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=np.float32
        )

        self.risk_history = []

        state = self._get_state()

        return state

    # ======================================================
    # GET STATE
    # ======================================================

    def _get_state(self):

        index = self.indices[
            self.current_step
        ]

        embb = self.data[
            "eMBB"
        ].iloc[index]

        urllc = self.data[
            "URLLC"
        ].iloc[index]

        mmtc = self.data[
            "mMTC"
        ].iloc[index]

        state = np.array(
            [
                embb["Load"],
                urllc["Load"],
                mmtc["Load"],

                embb["Rate"],
                urllc["Rate"],
                mmtc["Rate"],

                urllc["TcpRtt"],
                urllc["pLoss"]
            ],
            dtype=np.float32
        )

        # Remove invalid values

        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0
        )

        # Protect neural network from extreme outliers

        state = np.clip(
            state,
            -5.0,
            5.0
        )

        return state

    # ======================================================
    # ACTION NORMALIZATION
    # ======================================================

    def _normalize_action(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        if action.shape != (3,):

            raise ValueError(
                "Action must contain "
                "3 allocation values."
            )

        action = np.nan_to_num(
            action,
            nan=1 / 3,
            posinf=1.0,
            neginf=0.0
        )

        # No negative allocation

        action = np.maximum(
            action,
            0.000001
        )

        # Make allocation sum equal to 1

        action = action / np.sum(action)

        return action

    # ======================================================
    # DELAY / RISK
    # ======================================================

    def _calculate_delay(self, action):

        index = self.indices[
            self.current_step
        ]

        urllc = self.data[
            "URLLC"
        ].iloc[index]

        tcp_rtt = float(
            urllc["TcpRtt"]
        )

        packet_loss = float(
            urllc["pLoss"]
        )

        # Protect against extreme outliers

        tcp_rtt = np.clip(
            tcp_rtt,
            -5,
            5
        )

        packet_loss = np.clip(
            packet_loss,
            -5,
            5
        )

        # Convert normalized values
        # into positive indicators

        rtt_indicator = max(
            0.0,
            tcp_rtt + 1.0
        )

        loss_indicator = max(
            0.0,
            packet_loss + 1.0
        )

        # URLLC allocation

        urllc_allocation = action[1]

        # Lower allocation means higher risk

        allocation_factor = (
            1.0 /
            max(
                float(urllc_allocation),
                0.05
            )
        )

        delay = (
            (
                5.0 * rtt_indicator
            )
            +
            (
                2.0 * loss_indicator
            )
        ) * allocation_factor

        return float(delay)

    # ======================================================
    # RESOURCE UTILIZATION
    # ======================================================

    def _calculate_utilization(self, action):

        index = self.indices[
            self.current_step
        ]

        total = 0.0

        for i, slice_name in enumerate(
            self.slices
        ):

            row = self.data[
                slice_name
            ].iloc[index]

            load = float(
                row["Load"]
            )

            rate = float(
                row["Rate"]
            )

            load = np.clip(
                load,
                -5,
                5
            )

            rate = np.clip(
                rate,
                -5,
                5
            )

            demand = (
                max(0.0, load + 1.0)
                +
                max(0.0, rate + 1.0)
            ) / 2.0

            total += (
                action[i] * demand
            )

        return float(total)

    # ======================================================
    # FAIRNESS
    # ======================================================

    def _calculate_fairness(self, action):

        numerator = (
            np.sum(action) ** 2
        )

        denominator = (
            self.num_slices
            *
            np.sum(action ** 2)
        )

        if denominator == 0:

            return 0.0

        return float(
            numerator / denominator
        )

    # ======================================================
    # CVaR
    # ======================================================

        # ======================================================
    # CVaR - Rockafellar-Uryasev Method
    # ======================================================

    @staticmethod
    def calculate_cvar(values, alpha=0.05):

        values = np.asarray(values, dtype=float)

        # Keep only finite values
        values = values[np.isfinite(values)]

        # Handle empty input
        if len(values) == 0:
            return 0.0

        # Validate alpha
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        # VaR at the (1 - alpha) quantile
        var = np.quantile(values, 1.0 - alpha)

        # Rockafellar-Uryasev CVaR
        excess_losses = np.maximum(values - var, 0.0)

        cvar = var + np.mean(excess_losses) / alpha

        return float(cvar)
    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        # Normalize action

        action = self._normalize_action(
            action
        )

        # Calculate metrics

        delay = self._calculate_delay(
            action
        )

        utilization = (
            self._calculate_utilization(
                action
            )
        )

        fairness = (
            self._calculate_fairness(
                action
            )
        )

        # Store risk

        self.risk_history.append(
            delay
        )

        # Calculate CVaR

        cvar = self.calculate_cvar(
            self.risk_history,
            alpha=0.05
        )

        # Reference CVaR threshold

        cvar_threshold = 18.0

        cvar_violation = max(
            0.0,
            cvar - cvar_threshold
        )

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = (

            2.0 * utilization

            +

            0.5 * fairness

            -

            0.10 * delay

            -

            0.20 * cvar_violation
        )

        # --------------------------------------------------
        # Update step
                # Update step
        self.previous_action = action.copy()

        self.current_step += 1

        terminated = self.current_step >= len(self.indices)
        truncated = False

        # Avoid indexing beyond the final sample
        if terminated:
            next_state = np.zeros(8, dtype=np.float32)
        else:
            next_state = self._get_state()
        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        info = {

            "eMBB_allocation":
                float(action[0]),

            "URLLC_allocation":
                float(action[1]),

            "mMTC_allocation":
                float(action[2]),

            "delay":
                float(delay),

            "CVaR":
                float(cvar),

            "CVaR_violation":
                float(cvar_violation),

            "utilization":
                float(utilization),

            "fairness":
                float(fairness),

            "constraint_satisfied":
                bool(
                    cvar <= cvar_threshold
                )
        }

        return (
            next_state,
            float(reward),
            terminated,
            truncated,
            info
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("STARTING ENVIRONMENT TEST")
    print("=" * 70)
    print()

    # Create environment

    env = RANSlicingEnvironment(
        episode_length=20
    )

    print()
    print("Resetting environment...")

    state = env.reset(
        seed=42
    )

    print()
    print("Initial state:")
    print(state)

    print()
    print("State shape:")
    print(state.shape)

    # ------------------------------------------------------
    # Test action
    # ------------------------------------------------------

    action = np.array(
        [
            0.40,
            0.45,
            0.15
        ],
        dtype=np.float32
    )

    print()
    print("Test action:")
    print(action)

    print()
    print(
        "Action sum:",
        np.sum(action)
    )

    # ------------------------------------------------------
    # Execute environment step
    # ------------------------------------------------------

    print()
    print("Executing environment step...")

    (
        next_state,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    print()
    print("Next state:")
    print(next_state)

    print()
    print("Reward:")
    print(reward)

    print()
    print("Terminated:")
    print(terminated)

    print()
    print("Info:")

    for key, value in info.items():

        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 70)
    print("ENVIRONMENT TEST COMPLETED")
    print("=" * 70)