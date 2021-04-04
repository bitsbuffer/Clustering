import os
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


def run_ica(df):
    X = df.values
    max_component = 70
    mean_kurtosis = []
    for num_component in range(2, max_component + 1):
        ica = FastICA(n_components=num_component, max_iter=500, random_state=1234)
        X_transform = ica.fit_transform(X)
        Xk = pd.DataFrame(X_transform)
        mean_kurtosis.append(Xk.kurt().mean())
    mean_kurtosis = np.absolute(mean_kurtosis)

    num_component = mean_kurtosis.argmax() + 2
    print(f"Max Kurtosis for {num_component}")
    ica = FastICA(n_components=num_component, max_iter=500, random_state=1234)
    X_transform = ica.fit_transform(X)

    # plot data
    plt.figure(figsize=(12, 10))
    plt.bar(np.arange(2, max_component + 1), mean_kurtosis)
    plt.xlabel("No. of components")
    plt.ylabel("Mean Kurtosis")
    fname = os.path.join(args.output, "ica_nc.png")
    plt.savefig(fname)

    return X_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-o', '--output', type=str, help="Output dir for images and compressed data", required=True)
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_data_name, compression="zip")
    y = df.pop('target')

    # run ica
    X_transform = run_ica(df)
    cols_names = []
    for i in range(X_transform.shape[1]):
        cols_names.append(f"component_{i}")
    df_transform = pd.DataFrame(X_transform, columns=cols_names)
    df_transform['target'] = y
    df_transform.to_csv(os.path.join(args.output, "data_ica.csv.zip"), compression="zip", index=False)

# -t ./dataset/bnp/data_cleaned.csv.zip -o ./dataset/bnp