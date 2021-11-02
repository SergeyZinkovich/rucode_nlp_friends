import numpy as np
import pandas as pd
import pickle
import configparser
from gensim.corpora import Dictionary
import gensim.models
import zipfile

config = configparser.ConfigParser()
config.read("config.ini")

TRAIN_DATASET_FILENAME = config["DEFAULT"]["TRAIN_DATASET_FILENAME"]
VAL_DATASET_FILENAME = config["DEFAULT"]["VAL_DATASET_FILENAME"]
TEST_DATASET_FILENAME = config["DEFAULT"]["TEST_DATASET_FILENAME"]

DICTIONARY_FILENAME = config["DEFAULT"]["DICTIONARY_FILENAME"]
TFIDF_MODEL_FILENAME = config["DEFAULT"]["TFIDF_MODEL_FILENAME"]
THEME_DICT_FILENAME = config["DEFAULT"]["THEME_DICT_FILENAME"]

W2V_MODEL_FILENAME = config["DEFAULT"]["W2V_MODEL_FILENAME"]


def save_w2v_theme_dict(obj):
    pickle.dump(obj, open(THEME_DICT_FILENAME, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_w2v_theme_dict():
    return pickle.load(open(THEME_DICT_FILENAME, 'rb'))


def save_tfidf(dct, tfidf):
    dct.save(DICTIONARY_FILENAME)
    tfidf.save(TFIDF_MODEL_FILENAME)


def load_tfidf():
    dct = Dictionary.load(DICTIONARY_FILENAME)
    tfidf = gensim.models.TfidfModel.load(TFIDF_MODEL_FILENAME)

    return dct, tfidf


def load_w2v_model():
    stream = zipfile.ZipFile(W2V_MODEL_FILENAME, 'r').open('model.bin')
    return gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)


def get_data():
    train = pd.read_csv(TRAIN_DATASET_FILENAME).sample(frac=1).reset_index(drop=True)
    val = pd.read_csv(VAL_DATASET_FILENAME).sample(frac=1).reset_index(drop=True)
    test = pd.read_csv(TEST_DATASET_FILENAME, index_col=0)
    return train, val, test


def get_theme_dict():
    return {'ЧЕНДЛЕР': 0, 'ДЖОУИ': 1, 'РОСС': 2, 'МОНИКА': 3, 'ФИБИ': 4, 'РЕЙЧЕЛ': 5}
