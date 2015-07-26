from plot import plot_embedding, plot_embedding_3D, plot_artificial_dataset
from isomap import Isomap
from sklearn import datasets
import load_datasets
from load_datasets import NATURAL, ARTIFICIAL
import algorithms as algos
import os
import shutil
import argparse
import logging


OUTPUT_DIR="../output"
ALGORITHMS = {
        "Isomap": {
            "n_neighbors": 40
        },
        "LocalLinearEmbedding": {
            "n_neighbors": 70
        },
        "HessianLLE": {
            "n_neighbors": 40
            },
        "LaplacianEigenmaps": {},
        "LocalTangentSpaceAnalysis": {
            "n_neighbors": 40
        },
        "StochasticNeighborEmbedding": {
            "init": "pca"
        }
        }

DATASETS = {
        "faces": {
            },
        "digits": {
            "n_samples": 6
            },
        "swiss_roll": {
            },
        "racespace": {
            "gender_agnostic" : True,
            "races2incl": ["negroid", "caucasian", "mongolian"]
            }
        }


parser = argparse.ArgumentParser()
FORMAT = '%(levelname)s %(asctime)s %(name)s: %(message)s'

parser.add_argument("-s", "--save", type=bool, default=False)


parser.add_argument("-a", "--algorithms", type=str, nargs='+',
        help="""Algorithm to run. Options are:
                \t1. Isomap
                \t2. LaplacianEigenmaps
                \t3. LocalTangentSpaceAnalysis
                \t4. StochasticNeighborEmbedding
                \t5. HessianLLE
                \t6. LocalLinearEmbedding
                \t7. all""",
            default=['all'])

parser.add_argument("-d", "--datasets", type=str, nargs='+',
        help="""Datasets to run. Options are:\n
                \t1. faces
                \t2. digits
                \t3. swissroll
                \t4. racespace
                \t5. all""",
            default=['all'])

parser.add_argument("-o", "--output", type=str,
        help="""Output Directory""", default="../output")

parser.add_argument("-3d", "--threeD",  type=bool,
        help="""Plot 3D output""", default=False)
logging.basicConfig(format=FORMAT, level=logging.INFO)
args = parser.parse_args()


output_dir = args.output if args.output else OUTPUT_DIR

datasets = DATASETS.keys() if len(args.datasets) == 0 or args.datasets.count('all') > 0 else args.datasets

algorithms = ALGORITHMS.keys() if len(args.algorithms) == 0 or args.algorithms.count('all') > 0 else args.algorithms

if (args.save):
    logging.info("Output being saved in %s" % output_dir)
    if os.path.exists(output_dir):
        logging.info("Older output directory exists. Removing and creating a fresh one.")
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)

for dataset in datasets:
    assert (dataset in DATASETS, "Unknown dataset: %s" % dataset)

    dataset_class = getattr(load_datasets, "load_%s" % dataset)
    type, X = dataset_class(**DATASETS.get(dataset, {}))

    for algo_name in algorithms:
        assert(algo_name in ALGORITHMS, "Unknown algorithm: %s" % algo_name)

        title = algo_name.title() + "-" +  dataset.title()
        algo_class = getattr(algos, algo_name)
        algo = algo_class(n_components=2, **ALGORITHMS.get(algo_name, {}))
        try:
            logging.info("Running " + " ".join(title.split("-")))
            if type == ARTIFICIAL:
                X_data = X[0]
                X_data.dtype='float64'

                X_reduced = algo.run(X_data)
                plot_artificial_dataset(X_data, X_reduced, color=X[1], title=title, save=args.save, saveDir=output_dir)
            else:
                X_data = X["data"]

                X_reduced = algo.run(X_data)


                # For 2 dimensional
                plot_embedding(X_reduced, X["target"], X["images"], X['target'], title, save=args.save, saveDir=output_dir)
                if args.threeD:
                    # For 3 dimensional
                    algo = algo_class(n_components=3, n_neighbors=30)
                    X_reduced = algo.run(X_data)
                    title = title + "-" + "3D"
                    plot_embedding_3D(X_reduced, X["target"], title, save=args.save, saveDir=output_dir)
        except Exception, e:
            logging.error("Dataset: %s, Algorithm: %s crashed. Error::%s" % (dataset, algo_name, e))
