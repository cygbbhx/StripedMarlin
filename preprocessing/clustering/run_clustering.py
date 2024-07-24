import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import pandas as pd
import csv

def get_optimal_clusters(embeddings):
    min_clusters = 2
    max_clusters = 9
    silhouette_scores = []
    inertias = []

    # Calculate silhouette scores for different numbers of clusters
    for n_clusters in tqdm(range(min_clusters, max_clusters + 1)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        print(f'Number of clusters: {n_clusters}, {inertia}, Silhouette score: {silhouette_avg}')

    # Plot silhouette scores to find the optimal number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette scores for different numbers of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.savefig('Sillhouette_Anlaysis.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia score')
    plt.savefig('Kmeans_Inertia.png')
    plt.show()

    # Choose the optimal number of clusters based on the silhouette scores
    optimal_clusters = np.argmax(silhouette_scores) + min_clusters
    print(f'Optimal number of clusters: {optimal_clusters}')
    return optimal_clusters


def visualize_gmm_clustering(emb, n_clusters):
    embeddings_arr = np.array(emb)

    # Fit the Gaussian Mixture Model
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(embeddings_arr)
    cluster_labels = gmm.predict(embeddings_arr)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_arr)

    gmm_centers = gmm.means_
    reduced_centers = pca.transform(gmm_centers)

    def plot_gaussian_ellipse(mean, cov, ax, **kwargs):
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = Ellipse(mean, v[0], v[1], 180.0 + angle, **kwargs)
        ax.add_patch(ell)

    # Plot the results
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()

    # Plot data points
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=1, c=cluster_labels, label='Data points')
    ax.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', marker='x', s=100, label='Cluster centers')

    # Plot Gaussian ellipses
    for i in range(gmm.n_components):
        cov_matrix = gmm.covariances_[i]
        reduced_cov_matrix = pca.components_ @ cov_matrix @ pca.components_.T
        plot_gaussian_ellipse(reduced_centers[i], reduced_cov_matrix, ax, edgecolor='blue', alpha=0.5)

    ax.set_title('GMM Clustering Results')
    ax.legend()
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('Clustering_Results.png')
    plt.show()
    
    return cluster_labels

def write_labels(cluster_labels, inst_ids, filename):
    original_data = pd.read_csv('../../data/train.csv')
    file = open(f'{filename}.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(['id', 'cluster', 'label'])

    for i, inst_id in enumerate(tqdm(inst_ids)):
        voice_id = inst_id.split('.ogg')[0]
        inst_row = original_data[original_data['id']==voice_id]
        if len(inst_row) != 0:
            label = inst_row.iloc[0]['label']
            writer.writerow([voice_id, cluster_labels[i], label])

with open('train_embedding.pkl', 'rb') as file:
    embeddings = pickle.load(file)
with open('train_ids.pkl', 'rb') as file:
    ids = pickle.load(file)

assert len(embeddings) == len(ids)
# cluster_num = get_optimal_clusters(embeddings)
our_labels = visualize_gmm_clustering(embeddings, 3)
write_labels(our_labels, ids, 'cluster_labels')