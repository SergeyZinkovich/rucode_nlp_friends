import numpy as np
import pandas as pd
import keras
import configparser
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from preprocess_udpipe import done_text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import data_processing
import gensim
import zipfile

config = configparser.ConfigParser()
config.read("config.ini")

LSTM_MODEL_FILENAME = config["DEFAULT"]["LSTM_MODEL_FILENAME"]
META_DICT_FILENAME = config["DEFAULT"]["META_DICT_FILENAME"]
TOKENIZER_FILENAME = config["DEFAULT"]["TOKENIZER_FILENAME"]
PREDICTION_FILENAME = config["DEFAULT"]["PREDICTION_FILENAME"]

THEMES_COUNT = 6


def prepare_data(data):
    data.dropna(inplace=True)

    x = list(map(done_text, data.friend_response.tolist()))

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(x)
    # x = tokenizer.texts_to_sequences(x)

    with zipfile.ZipFile('185.zip', 'r') as archive:
        stream = archive.open('model.bin')
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    x_vec = []
    for words in x:
        x_vec.append([])
        for w in words:
            if w not in w2v_model:
                continue
            x_vec[-1].append(w2v_model.get_vector(w))

    max_words = 0
    for i in range(len(x_vec)):
        # if len(x_vec[i]) > 1000:
        #     x_vec[i] = x_vec[i][:1000]
        if len(x_vec[i]) > max_words:
            max_words = len(x_vec[i])

    x_vec = np.array(x_vec)

    x_vec = sequence.pad_sequences(x_vec, maxlen=max_words, value=np.zeros(300, dtype='float32'))

    y = []
    theme_dict = data_processing.get_theme_dict()
    for category in data.Category:
        y.append(theme_dict[category])
    y = np.array(y)

    data_processing.save_train_files(x_vec, y, max_words)

    return x_vec, y, theme_dict, max_words


def prepare_val_data(data):
    data.dropna(inplace=True)

    x = prepare_dataset_x(data)

    theme_dict = data_processing.get_theme_dict()

    y = [theme_dict[i] for i in data.label]
    y = np.array(y)

    data_processing.save_prepared_dataset(x, y, 'val')

    return x, y


def prepare_test(data):
    x = prepare_dataset_x(data)
    data_processing.save_prepared_dataset(x, dataset_type='test')

    return x


def prepare_dataset_x(data):
    data.fillna('', inplace=True)
    max_words = data_processing.load_metadata()

    x = list(map(done_text, data.friend_response.tolist()))

    with zipfile.ZipFile('185.zip', 'r') as archive:
        stream = archive.open('model.bin')
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    x_vec = []
    for words in x:
        x_vec.append([])
        for w in words:
            if w not in w2v_model:
                continue
            x_vec[-1].append(w2v_model.get_vector(w))

    # if len(x[0]) > 1000:
    #     x[0] = x[0][:1000]

    x_vec = sequence.pad_sequences(x_vec, maxlen=max_words, value=np.zeros(300, dtype='float32'))

    return x_vec


def train(x_train, y_train, x_val, y_val, max_words):
    model = Sequential()
    model.add(keras.layers.LSTM(150, input_shape=(max_words, 300), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(THEMES_COUNT))
    model.add(Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

    model.save(LSTM_MODEL_FILENAME)


def test(x_val, y_val):
    model = keras.models.load_model(LSTM_MODEL_FILENAME)
    model.evaluate(x_val, y_val, batch_size=128)


def predict(x):
    model = keras.models.load_model(LSTM_MODEL_FILENAME)

    y_predicted = model.predict(x, batch_size=128)
    theme_dict = {v: k for k, v in data_processing.get_theme_dict().items()}
    y_predicted = [theme_dict[i] for i in y_predicted.argmax(axis=1)]
    pd.DataFrame(y_predicted).to_csv(PREDICTION_FILENAME, header=['Category'], index_label='Id')


if __name__ == "__main__":
    train_data, val_data, test_data = data_processing.get_data()

    x_train, y_train, theme_dict, max_words = prepare_data(train_data.iloc[:100])
    # x_train, y_train = data_processing.load_prepared_dataset()
    # max_words = data_processing.load_metadata()
    # theme_dict = data_processing.get_theme_dict()

    x_val, y_val = prepare_val_data(val_data.iloc[:100])
    # x_val, y_val = data_processing.load_prepared_dataset('val')

    train(x_train, y_train, x_val, y_val, max_words)
    # test(x_val, y_val)

    # x_test = prepare_test(test_data)
    # x_test = data_processing.load_prepared_dataset('test')

    # predict(x_test)
