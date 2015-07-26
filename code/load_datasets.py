import os
import numpy as np
from scipy.misc import imread, imresize
from sklearn import datasets

RACESPACE_PATH = "../RaceSpaceReg"

ARTIFICIAL = 1
NATURAL = 2

RACE_LABELS = {
        "caucasian_female" : "cf",
        "caucasian_male" : "cm",
        "indianer_male" : "im",
        "indianer_female" : "if",
        "indo-arier_female" : "iaf",
        "indo-arier_male" : "iam",
        "mongolian_female" : "mf",
        "mongolian_male" : "mm",
        "negroid_female" : "nf",
        "negroid_male" : "nm",
        "caucasian" : "c",
        "indianer" : "i",
        "indo-arier" : "ia",
        "mongolian" : "m",
        "negroid" : "n",
        "male": "_m",
        "female": "_f"
        }

def load_racespace(races2incl=['caucasian', 'caucasian', 'indianer', 'indo-arier', 'mongolian', 'negroid'],
                            genders2incl=['male', 'female'], gender_agnostic=False, race_agnostic=False):
    races = []
    genders = []
    names = []
    images = []
    data = []
    targets = []
    for race_gender in os.listdir(RACESPACE_PATH):
        (race, gender) = race_gender.split("_")

        target_string = None
        if not gender_agnostic and not race_agnostic:
            target_string = race_gender
        elif gender_agnostic and race_agnostic:
            raise ValueError("Data set cannot be both gender and race agnostic")
        elif gender_agnostic:
            target_string = race
        else:                           # race agnostic
            target_string = gender

        if race not in races2incl or gender not in genders2incl:
            continue

        for people_dir in os.listdir(os.path.join(RACESPACE_PATH, race_gender)):
            for image in os.listdir(os.path.join(RACESPACE_PATH, race_gender, people_dir)):
                image = imresize(imread(os.path.join(RACESPACE_PATH, race_gender, people_dir, image)), (64, 64))
                images.append(image)
                data.append(image.reshape(image.shape[0]*image.shape[1]))
                names.append(people_dir)
                races.append(race)
                genders.append(gender)
                targets.append(RACE_LABELS[target_string])
    return (NATURAL, {
            "races": np.array(races),
            "names": np.array(names),
            "images": np.array(images),
            "data": np.array(data),
            "target": np.array(targets)
            })

def load_faces():
    X = datasets.fetch_olivetti_faces()
    X.data.dtype='float64'
    return (NATURAL, X)

def load_digits(n_samples=10):
    return (NATURAL, datasets.load_digits(n_samples))

def load_swiss_roll():
    return (ARTIFICIAL, datasets.make_swiss_roll(n_samples=1500))

