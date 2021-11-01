import numpy as np
import pandas as pd
import pickle
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

TRAIN_DATASET_FILENAME = config["DEFAULT"]["TRAIN_DATASET_FILENAME"]
VAL_DATASET_FILENAME = config["DEFAULT"]["VAL_DATASET_FILENAME"]
TEST_DATASET_FILENAME = config["DEFAULT"]["TEST_DATASET_FILENAME"]

META_DICT_FILENAME = config["DEFAULT"]["META_DICT_FILENAME"]
TOKENIZER_FILENAME = config["DEFAULT"]["TOKENIZER_FILENAME"]
X_TRAIN_FILENAME = config["DEFAULT"]["X_TRAIN_FILENAME"]
Y_TRAIN_FILENAME = config["DEFAULT"]["Y_TRAIN_FILENAME"]
X_VAL_FILENAME = config["DEFAULT"]["X_VAL_FILENAME"]
Y_VAL_FILENAME = config["DEFAULT"]["Y_VAL_FILENAME"]
X_TEST_FILENAME = config["DEFAULT"]["X_TEST_FILENAME"]


def get_data():
    train = pd.read_csv(TRAIN_DATASET_FILENAME).sample(frac=1).reset_index(drop=True)
    val = pd.read_csv(TRAIN_DATASET_FILENAME).sample(frac=1).reset_index(drop=True)
    test = pd.read_csv(TRAIN_DATASET_FILENAME, index_col=0)
    return train, val, test


def load_tokenizer():
    tokenizer = pickle.load(open(TOKENIZER_FILENAME, 'rb'))

    return tokenizer


def load_metadata():
    meta_dict = pickle.load(open(META_DICT_FILENAME, 'rb'))

    return meta_dict['max_words'], meta_dict['unic_words_count']


def get_theme_dict():
    return {'ЧЕНДЛЕР': 0, 'ДЖОУИ': 1, 'РОСС': 2, 'МОНИКА': 3, 'ФИБИ': 4, 'РЕЙЧЕЛ': 5}


def save_meta_dict(max_words, unic_words_count):
    meta_dict = {
        'max_words': max_words,
        'unic_words_count': unic_words_count
    }

    pickle.dump(meta_dict, open(META_DICT_FILENAME, 'wb'), pickle.HIGHEST_PROTOCOL)


def save_tokenizer(tokenizer):
    with open(TOKENIZER_FILENAME, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_prepared_dataset(x, y=None, dataset_type='train'):
    if dataset_type == 'train':
        np.save(X_TRAIN_FILENAME, x)
        np.save(Y_TRAIN_FILENAME, y)
    elif dataset_type == 'val':
        np.save(X_VAL_FILENAME, x)
        np.save(Y_VAL_FILENAME, y)
    elif dataset_type == 'test':
        np.save(X_TEST_FILENAME, x)


def load_prepared_dataset(dataset_type='train'):
    if dataset_type == 'train':
        x = np.load(X_TRAIN_FILENAME)
        y = np.load(Y_TRAIN_FILENAME)

        return x, y
    elif dataset_type == 'val':
        x = np.load(X_VAL_FILENAME)
        y = np.load(Y_VAL_FILENAME)

        return x, y
    elif dataset_type == 'test':
        x = np.load(X_TEST_FILENAME)

        return x


def save_train_files(x_train, y_train, tokenizer, max_words):
    save_meta_dict(max_words, len(tokenizer.word_counts))
    save_tokenizer(tokenizer)
    save_prepared_dataset(x_train, y_train)
