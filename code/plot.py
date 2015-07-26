import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox, animation
from sklearn import random_projection, manifold, lda
import os
import logging

def plot_embedding(X, Y, YImg=None, YLbls=None, title=None, save=False, saveDir=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y_uniq = list(set(Y))
    y_col_map = dict([(y_uniq[idx], idx) for idx in xrange(len(y_uniq))])

    fig = plt.figure()
    ax = fig.add_subplot(211)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(Y[i]),
            color=plt.cm.Set1(float(y_col_map[Y[i]]) / len(y_uniq)),
            fontdict={'weight': 'bold', 'size': 9})

    ax2 = fig.add_subplot(212)
    if hasattr(offsetbox, 'AnnotationBbox') and \
                    (YImg != None or YLbls != None):

        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i][:2] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i][:2]]]

            children = []
            if YLbls is not None:
                children.append(offsetbox.TextArea(YLbls[i]))

            if YImg is not None:
                children.append(offsetbox.OffsetImage(YImg[i], cmap=plt.cm.gray_r, zoom=0.5))

            vpacker = offsetbox.VPacker(children=children, pad=0, sep=5, align='center')

            bbox = offsetbox.AnnotationBbox(
                vpacker,
                X[i][0:2])
            ax2.add_artist(bbox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if save:
        assert(saveDir is not None, "No directory passed as argument")
        filename = os.path.join(saveDir, title + ".png")
        logging.info("Saving file: " + filename)
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def plot_artificial_dataset(X, X_r, color=None, title=None, save=False, saveDir=None):
    fig = plt.figure()
    try:
        # compatibility matplotlib < 1.0
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    except:
        ax = fig.add_subplot(211)
        ax.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

    ax.set_title(title + " (Original data)")
    ax = fig.add_subplot(212)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title(title + ' (Projected data)')
    if save:
        assert(saveDir, "No output directory passed as argument")
        filename = os.path.join(saveDir, title)
        logging.info("Saving file: " + filename)
        #anim.save(filename + ".gif", fps=4, writer='imagemagick')
 #       anim.save(filename + ".mp4", fps=30, writer='mencoder')
        plt.savefig(filename, dpi=200)

    else:
        plt.show()

def plot_embedding_3D(X, Y, title=None, save=True, saveDir=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y_uniq = list(set(Y))
    y_col_map = dict([(y_uniq[idx], idx) for idx in xrange(len(y_uniq))])

    fig = plt.figure(figsize=(20, 20))
    ax3D = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax3D.text(X[i, 0], X[i, 1], X[i, 2], s=str(Y[i]),
            color=plt.cm.Set1(float(y_col_map[Y[i]])/len(y_uniq)))

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if save:
        assert(saveDir is not None, "No output directory passed as argument")
        def animate(i):
            ax3D.view_init(elev=10, azim=12 * i)

        anim = animation.FuncAnimation(fig, animate, frames=30, interval=30, blit=True)
        filename = os.path.join(saveDir, title)
        logging.info("Saving file: " + filename)
        anim.save(filename + ".gif", fps=8, writer='imagemagick')
#        anim.save(filename + ".mp4", fps=30, writer='mencoder')
    else:
        plt.show()
if __name__ == "__main__":
    #from laplacian_eigenmaps import LaplacianEigenmaps
    X_lda = lda.LDA(n_components=2)
    from sklearn import datasets

    from load_datasets import load_racespace
    #type, X = load_racespace(gender_agnostic=True, races2incl=['caucasian', 'negroid', 'mongoloid'])
    #digits = datasets.load_digits(n_class=6)
    ##X_projected = X_lda.fit_transform(racespace["data"], racespace["target"])
    #embedding = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver='arpack')
    #X_projected = embedding.fit_transform(racespace["data"])
    #plot_embedding(X_projected, racespace['target'], racespace["images"], racespace["names"], "Digits Random Projection")


    #embedding = manifold.SpectralEmbedding(n_components=3, random_state=0, eigen_solver='arpack')
    #X_projected = embedding.fit_transform(racespace["data"])
    #plot_embedding_3D(X_projected, racespace['target'], "Digits Random Projection 3D")

    X = datasets.fetch_olivetti_faces()

    embedding = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='ltsa')
    X_r = embedding.fit_transform(X["data"])
    plot_embedding(X_r, X["target"], X["images"], X["target"], "Original")
    plt.show()
