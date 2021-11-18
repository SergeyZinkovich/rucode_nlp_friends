import numpy as np
import pandas as pd
import keras
import keras.layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle


def text_classifier(vectorizer, transformer, classifier):
    return Pipeline(
        [("vectorizer", vectorizer),
         ("transformer", transformer),
         ("classifier", classifier)]
    )


def train_logreg():
    train = pd.read_csv('input/train_data.csv')
    val = pd.read_csv('input/val_data.csv')

    train_val = train.append(val, ignore_index=True)
    train_val['Category'][train_val.Category.isna()] = train_val['label'][train_val.Category.isna()]
    train_val.fillna('', inplace=True)
    train_val.to_csv('input/train_val.csv')
    unique_labels = np.unique(train.Category)
    np.save('logreg_model/unique_labels.npy', unique_labels)

    texts_friend = train_val.friend_response
    texts_other = train_val.other_speaker
    labels = train_val.Category.apply(lambda x: np.argwhere(x == unique_labels)[0][0])

    clf = text_classifier(CountVectorizer(), TfidfTransformer(), LogisticRegression())
    clf.fit(texts_friend, labels)
    pickle.dump(clf, open('logreg_model/logreg_friend.pkl', 'wb'))

    clf2 = text_classifier(CountVectorizer(), TfidfTransformer(), LogisticRegression())
    clf2.fit(texts_other, labels)
    pickle.dump(clf2, open('logreg_model/logreg_other.pkl', 'wb'))


def train_nn():
    train_val = pd.read_csv('input/train_val.csv', index_col=0)
    train_val.fillna('', inplace=True)
    unique_labels = np.load('logreg_model/unique_labels.npy', allow_pickle=True)

    labels = train_val.Category.apply(lambda x: np.argwhere(x == unique_labels)[0][0])

    clf = pickle.load(open('logreg_model/logreg_friend.pkl', 'rb'))
    clf2 = pickle.load(open('logreg_model/logreg_other.pkl', 'rb'))
    train_confidence_friend = clf.decision_function(train_val.friend_response)
    train_confidence_other = clf2.decision_function(train_val.other_speaker)

    x = list(zip(train_confidence_friend, train_confidence_other))
    x = np.array(x)
    x = x.reshape((-1, 12))

    labels = labels.values

    x_train, x_val, y_train, y_val = train_test_split(x, labels)

    model = keras.Sequential()
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(12))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(6))
    model.add(keras.layers.Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, labels, batch_size=128, epochs=40)
    model.save('logreg_model/nn.h5')


def predict_output():
    test = pd.read_csv('input/test.csv')
    unique_labels = np.load('logreg_model/unique_labels.npy', allow_pickle=True)

    clf = pickle.load(open('logreg_model/logreg_friend.pkl', 'rb'))
    clf2 = pickle.load(open('logreg_model/logreg_other.pkl', 'rb'))

    ans_confidence_friend = clf.decision_function(test.friend_response)
    ans_confidence_other = clf2.decision_function(test.other_speaker)

    x_test = list(zip(ans_confidence_friend, ans_confidence_other))
    x_test = np.array(x_test)
    x_test = x_test.reshape((-1, 12))

    model = keras.models.load_model('logreg_model/nn.h5')

    ans = model.predict(x_test).argmax(axis=1)

    answer = pd.DataFrame()
    answer.index.name = 'Id'
    answer['Category'] = list(map(lambda x: unique_labels[x], ans))
    answer.to_csv('output/ans.csv')


if __name__ == "__main__":
    train_logreg()
    train_nn()
    predict_output()
