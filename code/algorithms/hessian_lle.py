from scipy.linalg import eigh
from scipy.linalg import qr
import numpy as np
from sklearn.neighbors import NearestNeighbors
class HessianLLE:
    """
        References:
        [1] Ye, Qiang, and Weifeng Zhi. Discrete hessian eigenmaps method for dimensionality reduction. Technical report, Department of Mathematics, University of Kentucky, 2012.

        [2] Donoho, David L., and Carrie Grimes. "Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data." Proceedings of the National Academy of Sciences 100.10 (2003): 5591-5596.

        [3] Nonlinear Manifold Learning II
            http://web.mit.edu/6.454/www/www_fall_2003/esuddert/manifold2_talk.pdf
    """

    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components


    def run(self, X):
        if self.n_neighbors <= self.n_components * (self.n_components + 3)/2:
            raise ValueError("Hessian eigenmap needs n_components(n_components + 3)/2 neighbors")

        # Get k nearest neighbor
        knn = NearestNeighbors(self.n_neighbors + 1).fit(X)
        neighbors = knn.kneighbors(X, return_distance=False)[:, 1:]
        Hf = self._learn_hessian_matrix(X, neighbors)
        return self._find_embedding(Hf)

    def _learn_hessian_matrix(self, X, neighbors):
        dp = self.n_components * (self.n_components + 1) / 2

        HEst = np.zeros((self.n_neighbors, 1 + self.n_components + dp), dtype=np.float)

        HEst[:, 0] = 1

        M = np.zeros((X.shape[0], X.shape[0]))

        # Calculate tanget space
        for i in xrange(X.shape[0]):
            # Get neighbors
            Ni = X[neighbors[i]]

            # Col wise Mean centering
            Ni -= Ni.mean(0)

            # Do PCA on local neighborhood to get the tanget space
            Ci = Ni.dot(Ni.T)

            lambdas, E = eigh(Ci)

            # We need n_component largest eigenvector
            # the eigh returns the ascending order, so reverse
            E = E[:, ::-1]
            # n_component dimensional tangent space
            HEst[:, 1 : 1 + self.n_components] = E[:, :self.n_components]

            # populate a (.) b where
                # a (.) b = [c_1, c_2, ...c_n(n+1)/2]
                # where c_(k(k+1)/2 +l) = a_k * b_l 1<= l <= k <= N

            j = 1 + self.n_components
            for k in xrange(self.n_components):
                HEst[:, j:j + self.n_components - k] = (E[:, k:k + 1] * E[:, k:self.n_components])

                j += self.n_components - k


            # Do QR decmposition to solve least square
            Q, R = qr(HEst)

            W = Q[:, self.n_components + 1:]

            # Normalize W
            for j in xrange(W.shape[1]):
                S = np.sum(W[:, j])
                if S > 0.0001:
                    W[:, j] /= S

            GG_T = W.dot(W.T)

            # Build quadratic form
            for k, n_y in enumerate(neighbors[i]):
                for j, n_x in enumerate(neighbors[i]):
                    M[n_x, n_y] += GG_T[j, k]
            #nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            #M[nbrs_x, nbrs_y] += np.dot(W, W.T)
        return M

    def _find_embedding(self, H):
        eigvals, eigvects = eigh(H)
        return eigvects[:, 1:self.n_components + 1]

if __name__ == "__main__":
    hLLE = HessianLLE(30)
    from sklearn import datasets
    X = datasets.load_digits(n_class=6)
    from plot import *
    X_r = hLLE.run(X["data"])
    plot_embedding(X_r, X.target, X.images, X.target, "Self")
