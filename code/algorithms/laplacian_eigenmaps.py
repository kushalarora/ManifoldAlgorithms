from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph
import numpy as np

class LaplacianEigenmaps:
    """
        Laplacian Eigenmaps also known as spectral embedding
        Laplacian eigenmaps[5] uses the spectral techniques to conserves the neighborhood in lower dimension.
        The data is discretized and the graph generated can be considered as a discrete approximation of the low dimensional manifold in the high dimensional space. Minimization of a cost function based on the graph ensures that points close to each other on the manifold are mapped close to each other in the low dimensional space, preserving local distances.
        The algorithm can be summarized as follows:
        1.  Calculate the pairwise distances and form an undirected graph.
        2.  Calculate the Laplacian Operator on the the graph.
        3.  Solve the following generalized eigenvalue problem
                Lv = \lambda Dv
        4.  Ignoring the first eigenvector corresponding to = 0, next d eigenvectors give embedding in lower d dimensional space.
         This can be viewed as approximate solution to N graph cut problem where attempt is that to cluster together the similar data points in lower dimension.

        References:

        [1] Belkin, Mikhail, and Partha Niyogi. "Laplacian eigenmaps and spectral techniques for embedding and clustering." NIPS. Vol. 14. 2001.

    """

    def __init__(self, n_components=2, affinity='nearest_neighbor', gamma=None, n_neighbors=None):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors


    def run(self, X):

        ### get adj matrix ###
        affinity_matrix = None

        if self.affinity == 'rbf':
            affinity_matrix = rbf_kernel(X, self.gamma)
        elif self.affinity == 'nearest_neighbor':
            _n_neighbors = self.n_neighbors if self.n_neighbors is not None \
                                else max(int(X.shape[0]/10), 1)
            kneighbors = kneighbors_graph(X, _n_neighbors).toarray()
            affinity_matrix = (kneighbors + kneighbors.T) * 0.5
        else:
            raise ValueError("affinity(str) can only be rbf or nearest_neighbor")

        assert(affinity_matrix is not None, "Affinity matrix needed for computation")

        return self._spectral_embedding(affinity_matrix)

    def _spectral_embedding(self, affinity_matrix):
        """ Computes spectral embedding.
            First calculates normalized laplacian
            Then does the eigenvalue decomposition
        """
        numComponents, labels = connected_components(affinity_matrix)

        if numComponents > 1:
            # for each component figure out embedding, return the complete embedding
            embedding = []
            connected_component = np.zeros(affinity_matrix.shape)
            for i in xrange(numComponents):
                for j in affinity_matrix.shape[0]:
                    if labels[j] == i:
                        connected_component[:, i]
                embedding.append(self._spectral_embedding(connected_component))
            return embedding


        self.n_components += 1;
        L, diag_vector = laplacian(affinity_matrix, normed=True, return_diag=True)

        D = np.diag(diag_vector)
        #eigvals, eigvects = eigsh(-L, k=self.n_components, sigma=1.0, which='LM')

        eigvals, eigvects = eigh(L)
        embedding = eigvects.T[:self.n_components] * diag_vector

        return embedding[1:self.n_components].T

def rbf_kernel(X, gamma=None):

    if gamma == None:
        gamma = 1.0/X.shape[1]

    # Compute (||X_i - X_j||_2)^2
    Dist = squareform(pdist(X, 'sqeuclidean'), checks=True)

    Dist *= -gamma

    # exponentiate in place
    np.exp(Dist, Dist)

    return Dist

if __name__ == "__main__":
    lapEigMaps = LaplacianEigenmaps(n_components=3, affinity='nearest_neighbor')
    from sklearn import datasets
    X = datasets.fetch_olivetti_faces()
    from plot import *
    X_r = lapEigMaps.run(X["data"])
    plot_embedding(X_r, X.target, X.images, X.target, "Self")
