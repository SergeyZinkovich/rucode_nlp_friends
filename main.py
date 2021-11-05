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

config = configparser.ConfigParser()
config.read("config.ini")

LSTM_MODEL_FILENAME = config["DEFAULT"]["LSTM_MODEL_FILENAME"]
META_DICT_FILENAME = config["DEFAULT"]["META_DICT_FILENAME"]
TOKENIZER_FILENAME = config["DEFAULT"]["TOKENIZER_FILENAME"]
PREDICTION_FILENAME = config["DEFAULT"]["PREDICTION_FILENAME"]

THEMES_COUNT = 6


def prepare_data(data):
    data.fillna('', inplace=True)

    x = list(map(done_text, data.friend_response.tolist()))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)

    max_words = 0
    for i in range(len(x)):
        # if len(x[i]) > 1000:
        #     x[i] = x[i][:1000]
        if len(x[i]) > max_words:
            max_words = len(x[i])

    x = sequence.pad_sequences(x, maxlen=max_words)

    y = []
    theme_dict = data_processing.get_theme_dict()
    for category in data.Category:
        y.append(theme_dict[category])
    y = np.array(y)

    data_processing.save_train_files(x, y, tokenizer, max_words)

    return x, y, theme_dict, max_words, len(tokenizer.word_counts)


def prepare_val_data(data):
    data.fillna('', inplace=True)

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
    tokenizer = data_processing.load_tokenizer()
    max_words, unic_words_count = data_processing.load_metadata()

    x = list(map(done_text, data.friend_response.tolist()))

    x = tokenizer.texts_to_sequences(x)

    # if len(x[0]) > 1000:
    #     x[0] = x[0][:1000]

    x = sequence.pad_sequences(x, maxlen=max_words)

    return x


def train(x_train, y_train, x_val, y_val, max_words, unic_words_count):
    model = Sequential()
    model.add(keras.layers.Embedding(unic_words_count+1, max_words))
    model.add(keras.layers.LSTM(150, dropout=0.2, recurrent_dropout=0.2))
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

    x_train, y_train, theme_dict, max_words, unic_words_count = prepare_data(train_data)
    # x_train, y_train = data_processing.load_prepared_dataset()
    # max_words, unic_words_count = data_processing.load_metadata()
    # theme_dict = data_processing.get_theme_dict()

    x_val, y_val = prepare_val_data(val_data)
    # x_val, y_val = data_processing.load_prepared_dataset('val')

    train(x_train, y_train, x_val, y_val, max_words, unic_words_count)
    # test(x_val, y_val)

    x_test = prepare_test(test_data)
    # x_test = data_processing.load_prepared_dataset('test')

    predict(x_test)
