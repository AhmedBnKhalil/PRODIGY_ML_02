import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(filepath)


def preprocess_data(data):
    """
    Encode categorical data and scale numerical data.
    """
    encoder = LabelEncoder()
    data['Gender'] = encoder.fit_transform(data['Gender'])
    features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler, features.columns


def find_optimal_clusters(data):
    """
    Use the elbow method to determine the optimal number of clusters.
    """
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.grid(True)
    plt.show()


def perform_clustering(data, n_clusters):
    """
    Apply K-means clustering with the given number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(data), kmeans


def analyze_clusters(data, labels, scaler, feature_names):
    """
    Analyze the resulting clusters.
    """
    data['Cluster'] = labels
    cluster_centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_names)
    cluster_distribution = data['Cluster'].value_counts()
    return cluster_centroids, cluster_distribution


# Main execution
if __name__ == "__main__":
    # Load the data
    data = load_data('Mall_Customers.csv')

    # Preprocess the data
    scaled_features, scaler, feature_names = preprocess_data(data)

    # Determine the optimal number of clusters
    find_optimal_clusters(scaled_features)

    # Perform clustering
    n_clusters = 5  # This can be adjusted based on the elbow plot
    labels, kmeans = perform_clustering(scaled_features, n_clusters)

    # Analyze the clusters
    centroids, distribution = analyze_clusters(data, labels, scaler, feature_names)

    print("Cluster Centroids:\n", centroids)
    print("\nCluster Distribution:\n", distribution)
