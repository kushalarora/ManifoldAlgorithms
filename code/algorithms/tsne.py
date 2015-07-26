from sklearn import manifold
from sklearn import datasets
from plot import *
class StochasticNeighborEmbedding():

    def __init__(self, n_components=2, n_neighbors=30, init='pca'):
        self.tsne = manifold.TSNE(n_components, init=init, random_state=0)

    def run(self, X):
        return self.tsne.fit_transform(X)


if __name__ == "__main__":

    tsne = manifold.TSNE(n_components=2, init='pca')
    X = datasets.make_swiss_roll(n_samples=2000)
    X[0].dtype='float64'
    import pdb;pdb.set_trace()
    X_tsne = tsne.fit_transform(X[0])
    plot_artificial_dataset(X[0], X_tsne, color=X[1], title='title')
