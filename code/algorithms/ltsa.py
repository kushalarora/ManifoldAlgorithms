from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
import numpy as np
class LocalTangentSpaceAnalysis:
    """
    References:
    [1] Zhang, Zhen-yue, and Hong-yuan Zha. "Principal manifolds and nonlinear dimensionality reduction via tangent space alignment."
        Journal of Shanghai University (English Edition) 8.4 (2004): 406-424.

    [2] Sui, Yuelei. "Local Tangent Space Alignment." (2012).
    """
    def __init__(self, n_neighbors=10, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components


    def run(self, X):
        knn = NearestNeighbors(self.n_neighbors + 1).fit(X)
        neighbors = knn.kneighbors(X, return_distance=False)[:, 1:]
        N = X.shape[0]

        B = np.zeros((N, N))

        # Get the projections along the orthogonal basis
        # of tanget plane.
        # This is achieved by minimizing the reconstruction error.
        # E = min (Z) || X_i - UZn_i + C1||
        #
        # The solution to this is Zo = mean(Z).
        # TODO:: Complete the definition

        for i in xrange(N):
            Xi = X[neighbors[i]]
            Xi -= Xi.mean(0)        # Xni - mean(Xni)

            Ci = Xi.dot(Xi.T)

            eigvects = eigh(Ci)[1][:, ::-1]


            # Now Gi = [1/\sqrt(K) g1, g2, ... gd]

            Gi = np.zeros((self.n_neighbors, self.n_components + 1))

            Gi[:, 0] = 1.0/np.sqrt(self.n_neighbors)

            Gi[:, 1:] = eigvects[:, :self.n_components]

            GiGiT = Gi.dot(Gi.T)

            #B[N_i, N_i] = I - GiGiT
            #where N_i are neighbors of ith data point.
            for k, n_y in enumerate(neighbors[i]):
                for j, n_x in enumerate(neighbors[i]):
                    if (k == j):
                        B[n_x, n_y] += 1

                    B[n_x, n_y] -= GiGiT[k, j]

            #nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            #B[nbrs_x, nbrs_y] -= GiGiT
            #B[neighbors[i], neighbors[i]] += 1
        # Solving for eigenvector of B
        eigvalsB, eigvecsB = eigh(B)
        return eigvecsB[:, 1:self.n_components + 1]


if __name__ == "__main__":
    ltse = LocalTangentSpaceAnalysis(30)
    from sklearn import datasets
    X = datasets.load_digits(n_class=6)
    from plot import *
    X_r = ltse.run(X["data"])
    plot_embedding(X_r, X.target, X.images, X.target, "Self")

