from gensim.corpora import Dictionary
import gensim.models
import pickle
import zipfile
import scipy.spatial.distance as ds
from preprocess_udpipe import done_text
import data_processing


def eval_tfidf(data):
    data1 = [done_text(str(d)) for d in data.friend_response]
    dct = Dictionary(data1)
    corpus = [dct.doc2bow(line) for line in data1]
    tfidf = gensim.models.TfidfModel(corpus)
    data_processing.save_tfidf(dct, tfidf)

    return dct, tfidf


def count_theme_dict(data, model):
    dct, tfidf = eval_tfidf(data)
    theme_dict = {}
    for i in data.values:
        words = done_text(str(i[2]))
        for w in words:
            if w not in model:
                continue
            vec = model.get_vector(w)
            if not i[3] in theme_dict:
                theme_dict[i[3]] = [vec * tfidf.idfs[dct.token2id[w]], 1]
            else:
                theme_dict[i[3]] = [theme_dict[i[3]][0] + vec * tfidf.idfs[dct.token2id[w]], theme_dict[i[3]][1] + 1]

    for key, val in theme_dict.items():
        theme_dict[key] = val[0] / val[1]
    data_processing.save_w2v_theme_dict(theme_dict)


def test(data, model):
    theme_dict = data_processing.load_w2v_theme_dict()

    correct, wrong = 0, 0
    j = 0

    dct, tfidf = data_processing.load_tfidf()
    for i in data.values:
        words = done_text(str(i[2]))
        vec = 0
        for w in words:
            if w not in model:
                continue
            v = model.get_vector(w)
            if w in dct.token2id:
                coef = tfidf.idfs[dct.token2id[w]]
            else:
                coef = 1
            if vec == 0:
                vec = [v * coef, 1]
            else:
                vec = [vec[0] + v * coef, vec[1] + 1]
        vec = vec[0]/vec[1]

        min = -1
        id = 0
        for key, val in theme_dict.items():
            norm = ds.cosine(val, vec)
            if min == -1 or norm < min:
                min = norm
                id = key
        j += 1

        if id == i[3]:
            correct += 1
        else:
            wrong += 1

    return correct / (correct + wrong)


if __name__ == "__main__":
    train_data, val_data, test_data = data_processing.get_data()
    w2v_model = data_processing.load_w2v_model()
    count_theme_dict(train_data, w2v_model)
    print(test(val_data, w2v_model))