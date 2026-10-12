import pandas as pd
import os

DATASET_PATH = "../dataset"

for file in os.listdir(DATASET_PATH):

    print("\n==============================")
    print("FILE:", file)
    print("==============================")

    path = os.path.join(DATASET_PATH, file)

    if file.endswith(".csv"):

        df = pd.read_csv(path)

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nData types:")
        print(df.dtypes)