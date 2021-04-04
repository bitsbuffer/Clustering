import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(df):
    scalar = StandardScaler()
    X = scalar.fit_transform(df)
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    plt.subplot(axes[0])
    plt.plot(np.arange(1, pca.n_components_ + 1), cum_var)
    plt.axhline(y=0.98, color='r', linestyle='-')
    plt.xlabel("No. of components")
    plt.ylabel("Cumulative Explained Variance")
    plt.subplot(axes[1])
    plt.bar(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
    plt.xlabel("No. of components")
    plt.ylabel("Explained Variance Ratio")
    fname = os.path.join(args.output, "pca.png")
    plt.savefig(fname)

    num_components = (cum_var < 0.98).argmin()
    pca = PCA(n_components=num_components)
    X_transform = pca.fit_transform(X)

    return X_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-o', '--output', type=str, help="Output dir for images and compressed data", required=True)
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_data_name, compression="zip")
    y = df.pop('target')
    X_transform = run_pca(df)
    cols_names = []
    for i in range(X_transform.shape[1]):
        cols_names.append(f"component_{i}")
    df_transform = pd.DataFrame(X_transform, columns=cols_names)
    df_transform['target'] = y
    df_transform.to_csv(os.path.join(args.output, "data_pca.csv.zip"), compression="zip", index=False)

# -t ./dataset/bnp/data_cleaned.csv.zip -o ./dataset/bnp