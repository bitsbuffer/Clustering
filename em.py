import os
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def run_gmm(df):
    scalar = StandardScaler()
    X = scalar.fit_transform(df)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(2, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.aic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    bars = []
    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    fname = os.path.join(args.output, args.prefix + "-em-bic.png")
    plt.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KMeans')
    parser.add_argument('-o', '--output', type=str, help="Output dir for images and compressed data", required=True)
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    parser.add_argument('-pre', '--prefix', type=str, help='Prefix to append to saved images fname', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_data_name, compression="zip")
    y = df.pop('target')
    run_gmm(df)
