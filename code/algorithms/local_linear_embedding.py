from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.linalg import solve
from scipy.linalg import eigh
class LocalLinearEmbedding:
    """ Local Linear Embedding.
        Locally Linear Embedding tries to reconstruct the data point in lower dimension using the neighborhood in higher dimension.
        The Locally Linear Embedding algorithm can be summarized in following steps:

        1.    Find neighborhood for each point
        2.    For each point compute the weight matrix that minimizes the reconstruction error for the point using neighborhood
        3.    Using this weight vector find the embedding that best reconstructs the point.

        References:
        [1] Roweis, Sam T., and Lawrence K. Saul. "Nonlinear dimensionality reduction by locally linear embedding." Science 290.5500 (2000): 2323-2326.
    """

    def __init__(self, n_neighbors=5, n_components=2, reg=1E-3):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg

    def run(self, X):

        # Get k nearest neighbor
        knn = NearestNeighbors(self.n_neighbors + 1).fit(X)
        neighbors = knn.kneighbors(X, return_distance=False)[:, 1:]
        W = self._learn_weights(X, neighbors)
        return self._find_embedding(W)


    def _learn_weights(self, X,  neighbors):
        # calculate W
        n_samples, n_neighbors = X.shape[0], neighbors.shape[1]

        W_local = np.zeros((n_samples, n_neighbors))
        Ones = np.ones(n_neighbors)
        # for sample i
        for i, A in enumerate(X[neighbors]):
            G = A - X[i]

            # Covariance matrix
            C = G.dot(G.T)

            trace = np.trace(C)

            # regularization
            Reg = (self.reg * trace if trace > 0 \
                        else reg) * np.eye(n_neighbors)

            C += Reg

            # solve Cw = 1
            w = solve(C, Ones, sym_pos = True)

            # W matrix with normalized ws
            W_local[i, :] = w/np.sum(w)

        W = np.zeros((n_samples, n_samples))
        for i, w in enumerate(W_local):
            W[i][neighbors[i]] = w

        return W


    def _find_embedding(self, W):
        Del = np.eye(W.shape[0])
        M = (Del - W - W.T + W.T.dot(W))

        eigvals, eigvects = eigh(M)

        return eigvects[:, 1:self.n_components + 1]


if __name__ == "__main__":
    lle = LocalLinearEmbedding(30)
    from sklearn import datasets
    X = datasets.load_digits(n_class=6)
    from plot import *
    X_r = lle.run(X["data"])
    plot_embedding(X_r, X.target, X.images, X.target, "Self")
