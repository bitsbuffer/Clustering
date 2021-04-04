import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class RandomProjection(GaussianRandomProjection):

    def inverse_transform(self, X):
        _pseudoinverse = np.linalg.pinv(self.components_.T)
        return X.dot(_pseudoinverse)


def run_rp(df):
    scalar = StandardScaler()
    X = scalar.fit_transform(df.values)
    max_component = X.shape[1]
    mean_errors = []
    for num_component in range(2, max_component + 1):
        rp = RandomProjection(n_components=num_component)
        X_transform = rp.fit_transform(X)
        X_recostruct = rp.inverse_transform(X_transform)
        mean_squared_error(X, X_recostruct)
        mean_errors.append(mean_squared_error(X, X_recostruct))

    mean_errors = np.array(mean_errors)
    num_component = (mean_errors <= 0.05).argmax()
    rp = RandomProjection(n_components=num_component)
    X_transform = rp.fit_transform(X)
    # plot the data
    plt.figure(figsize=(12, 10))
    plt.bar(np.arange(2, max_component + 1), mean_errors)
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.xlabel("No. of components")
    plt.ylabel("Reconstruction Error (MSE)")
    fname = os.path.join(args.output, "random_projections.png")
    plt.savefig(fname)

    return X_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-o', '--output', type=str, help="Output dir for images and compressed data", required=True)
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_data_name, compression="zip")
    y = df.pop('target')
    X_transform = run_rp(df)

    cols_names = []
    for i in range(X_transform.shape[1]):
        cols_names.append(f"component_{i}")
    df_transform = pd.DataFrame(X_transform, columns=cols_names)
    df_transform['target'] = y
    df_transform.to_csv(os.path.join(args.output, "data_rp.csv.zip"), compression="zip", index=False)

# -t ./dataset/bnp/data_cleaned.csv.zip -o ./dataset/bnp