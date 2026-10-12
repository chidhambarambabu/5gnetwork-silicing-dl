import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset"
PROCESSED_PATH = DATASET_PATH / "processed"


# Create processed folder if it doesn't exist
PROCESSED_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET FILES
# ============================================================

SLICE_FILES = {
    "eMBB": "eMBB.csv",
    "URLLC": "URLLC.csv",
    "mMTC": "mMTC.csv"
}


# ============================================================
# FEATURES USED BY OUR RL MODEL
# ============================================================

FEATURES = [
    "Load",
    "Rate",
    "TotPkts",
    "TotBytes",
    "pLoss",
    "TcpRtt",
    "Dur",
    "RunTime"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_slice_data():

    data = {}

    print("\nDataset location:")
    print(DATASET_PATH)

    print("\nChecking dataset folder...")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    print("Dataset folder found.")

    print("\nFiles available:")

    for file in DATASET_PATH.iterdir():

        print(" -", file.name)

    for slice_name, filename in SLICE_FILES.items():

        path = DATASET_PATH / filename

        print("\n" + "=" * 60)
        print(f"Loading {slice_name}")
        print("=" * 60)

        print("File:", path)

        if not path.exists():

            raise FileNotFoundError(
                f"\nCould not find:\n{path}\n\n"
                f"Please check that {filename} exists inside:\n"
                f"{DATASET_PATH}"
            )

        df = pd.read_csv(path)

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        # Check required features
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing columns in {filename}: "
                f"{missing_features}"
            )

        # Keep required columns
        df = df[
            FEATURES + ["Label"]
        ].copy()

        # Add slice information
        df["Slice"] = slice_name

        data[slice_name] = df

    return data


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(data):

    cleaned_data = {}

    for slice_name, df in data.items():

        print("\n" + "=" * 60)
        print(f"Cleaning {slice_name}")
        print("=" * 60)

        df = df.copy()

        # Convert selected features to numeric
        for feature in FEATURES:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # Replace infinity
        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        before = len(df)

        # Remove rows with missing values
        df.dropna(
            subset=FEATURES,
            inplace=True
        )

        after = len(df)

        print("Rows before cleaning:", before)
        print("Rows after cleaning:", after)

        print(
            "Rows removed:",
            before - after
        )

        cleaned_data[slice_name] = df

    return cleaned_data


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(data):

    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    # Combine all slices temporarily
    combined = pd.concat(
        data.values(),
        ignore_index=True
    )

    print(
        "Total rows used for scaler:",
        len(combined)
    )

    # Create scaler
    scaler = StandardScaler()

    # Fit scaler using all slices
    scaler.fit(
        combined[FEATURES]
    )

    normalized_data = {}

    for slice_name, df in data.items():

        df = df.copy()

        df[FEATURES] = scaler.transform(
            df[FEATURES]
        )

        normalized_data[slice_name] = df

        print(
            f"{slice_name} normalization completed."
        )

    return normalized_data, scaler


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(data):

    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    for slice_name, df in data.items():

        output_path = (
            PROCESSED_PATH /
            f"{slice_name}_processed.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"{slice_name} saved to:"
        )

        print(output_path)


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(data):

    for slice_name, df in data.items():

        print("\n" + "=" * 60)
        print(f"STATISTICS — {slice_name}")
        print("=" * 60)

        print(
            "\nShape:",
            df.shape
        )

        print("\nFeature statistics:")

        print(
            df[FEATURES]
            .describe()
            .round(4)
        )

        print("\nLabel distribution:")

        print(
            df["Label"]
            .value_counts()
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("5G-SliciNdd PREPROCESSING")
    print("=" * 70)

    # 1. Load
    data = load_slice_data()

    # 2. Clean
    data = clean_data(data)

    # 3. Normalize
    data, scaler = normalize_data(data)

    # 4. Save
    save_processed_data(data)

    # 5. Display statistics
    print_statistics(data)

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)