import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cluster
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def calculate_silhouette(X, cluster_labels, n_clusters):
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    fname = os.path.join(args.output, args.prefix + "-kmedoids-silhouette-" + str(n_clusters) + ".png")
    plt.savefig(fname)


def plot_elbow(range_n_clusters, inertias):
    # plot elbow
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters.astype(str), inertias, color='red')
    plt.xlabel('No. of Clusters', fontsize=15)
    plt.ylabel('Inertia', fontsize=15)
    plt.title('No. Of Clusters vs Inertia', fontsize=15)
    plt.grid()
    fname = os.path.join(args.output, args.prefix + "-kmedoids-elbow.png")
    plt.savefig(fname)


def plot_purity(range_n_clusters, purity):
    # plot elbow
    plt.figure(figsize=(8, 6))
    plt.bar(range_n_clusters.astype(str), purity)
    plt.xlabel('No. of Clusters', fontsize=15)
    plt.ylabel('Purity', fontsize=15)
    plt.title('No. Of Clusters vs Purity', fontsize=15)
    plt.grid()
    fname = os.path.join(args.output, args.prefix + "-kmedoids-purity.png")
    plt.savefig(fname)


def run_knn(df, y):
    scalar = MinMaxScaler()
    X = scalar.fit_transform(df)
    range_n_clusters = np.array([2, 3, 4, 5, 6])
    inertias = []
    purity = []
    for n_clusters in range_n_clusters:
        clusterer = KMedoids(n_clusters=n_clusters,
                             metric="manhattan",
                             init='k-medoids++',
                             max_iter=100,
                             random_state=1234)
        cluster_labels = clusterer.fit_predict(X)
        inertias.append(clusterer.inertia_)
        calculate_silhouette(X, cluster_labels, n_clusters)
        purity.append(purity_score(y, cluster_labels))
    plot_purity(range_n_clusters, purity)
    plot_elbow(range_n_clusters, inertias)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KMeans')
    parser.add_argument('-o', '--output', type=str, help="Output dir for images and compressed data", required=True)
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    parser.add_argument('-pre', '--prefix', type=str, help='Prefix to append to saved images fname', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_data_name, compression="zip")
    y = df.pop('target')
    run_knn(df, y)




