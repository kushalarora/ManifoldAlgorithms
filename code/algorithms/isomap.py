from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import floyd_warshall
from sklearn.decomposition import KernelPCA
import numpy as np
from sklearn import datasets
from plot import plot_artificial_dataset
class Isomap:
    """
        Isomap Algorithm from Tenenbaum, J.B.; De Silva, V.; & Langford, J.C.

        References
        ---------
        [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
                framework for nonlinear dimensionality reduction. Science 290 (5500)
    """

    def __init__(self, n_neighbors, n_components):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def run(self, X):
        """
        Return the embedding.
        """

        #   Compute the nearest neighbor graphs
        nearestNeighbors = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(X)
        k_neighbors_array = kneighbors_graph(X, self.n_neighbors, mode='distance').toarray()


        numNodes = k_neighbors_array.shape[0]
        # making graph symmetric
        for i in range(numNodes):
            for j in range(numNodes):
                if k_neighbors_array[j, i] <= k_neighbors_array[i, j]:
                    k_neighbors_array[i, j] = k_neighbors_array[j, i]
                else:
                    k_neighbors_array[j, i] = k_neighbors_array[i, j]

        #   Compute the all pair shortest path distance.
        dist_matrix = floyd_warshall(k_neighbors_array, directed=False)

        dist_matrix[np.isinf(dist_matrix)] = 0
        # Do MDS or learn embedding
        # MDS can also be seen as a case of Kernel PCA
        # using data dependent kernel
        # So using K = 1/2 D^2,
        # we generate projections along principal components

        kernel = dist_matrix ** 2

        kernel *= -0.5

        kernelPCA = KernelPCA(n_components=self.n_components, kernel='precomputed')

        return kernelPCA.fit_transform(kernel)

if __name__ == "__main__":
    isomap = Isomap(10, 3)
    X, color = datasets.make_swiss_roll(n_samples=3000)
    X_r = isomap.run(X)
    plot_artificial_dataset(X, X_r, color, "Swiss Roll")
