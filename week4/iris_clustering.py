import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:")
print(X.shape)

print("\nFeature Names:")
print(iris.feature_names)

print("\nTrue Labels:")
print(y)


# 2. Standardize the data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# 3. Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_scaled)

print("\nPredicted Clusters:")
print(clusters)


# 4. Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Data:")
print(X_pca)


# 5. Visualize K-Means clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering of Iris Dataset")

plt.show()


# 6. Compare predicted clusters with true labels

comparison = pd.DataFrame({
    "True_Label": y,
    "Predicted_Cluster": clusters
})

print("\nTrue Labels vs Predicted Clusters:")
print(comparison.head(20))
