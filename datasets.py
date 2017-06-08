import logging

import numpy as np
import pandas as pd

from utils import utils
from word2vec_gensim import word2vec_gensim

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class Datasets():
    def __init__(self):
        self.word2vec = word2vec_gensim()
        self.matches = None

    def load_dblp1(self):
        logging.info('loading data')

        data = pd.read_csv('data/DBLP1.csv', encoding='iso-8859-1')

        data = data.fillna('')
        data = data.applymap(lambda x: utils.unicodeToAscii(x) if isinstance(x, str) else x)

        data['doc'] = data['authors'] + ' ' + data['title'] + ' ' + data['venue']

        vecs = []

        for row in data.iterrows():
            # print(row[1]['doc'])
            vecs.append(self.word2vec.sentence2vec(row[1]['doc']))

        return vecs

    def load_scholar(self):
        logging.info('loading data')

        data = pd.read_csv('data/Scholar.csv', encoding='iso-8859-1')

        data = data.fillna('')
        data = data.applymap(lambda x: utils.unicodeToAscii(x) if isinstance(x, str) else x)
        data['doc'] = data['authors'] + ' ' + data['title'] + ' ' + data['venue']

        vecs = []
        for row in data.iterrows():
            # print(row[1]['doc'])
            vecs.append(self.word2vec.sentence2vec(row[1]['doc']))

        return vecs

    def load_mapping(self):
        logging.info('loading data')

        dblp = pd.read_csv('data/DBLP1.csv', encoding='iso-8859-1')
        scholar = pd.read_csv('data/Scholar.csv', encoding='iso-8859-1')

        mapping = pd.read_csv('data/DBLP-Scholar_perfectMapping.csv', encoding='iso-8859-1')

        id_dblp = dblp['id'].tolist()
        id_dblp = list(map(lambda x: x.lower(), id_dblp))

        id_scholar = scholar['id'].tolist()
        id_scholar = list(map(lambda x: x.lower(), id_scholar))

        dblp_map = dict([(v, i) for (i, v) in enumerate(id_dblp)])
        scholar_map = dict([(v, i) for (i, v) in enumerate(id_scholar)])

        matrix = np.zeros((len(id_dblp), len(id_scholar)))

        list_dblp = mapping['idDBLP'].tolist()
        list_scholar = mapping['idScholar'].tolist()

        for (d, s) in zip(list_dblp, list_scholar):
            print(d, s)
            id_d = dblp_map[d.lower()]
            id_s = scholar_map[s.lower()]
            matrix[id_d][id_s] = 1

        print(matrix.sum())
        return matrix

    def gen_pos(self):

        if self.matches is None:
            self.matches = self.load_mapping().nonzero()

        pass

    def gen_neg(self):
        if self.matches is None:
            self.matches = self.load_mapping().nonzero()

        pass

    def augment(self, words):

        rnd = np.random.choice(range(len(words)))
        rand_word = words[rnd]

        replace = self.word2vec.w2v.most_similar([rand_word])[0][0]
        print(rand_word, replace)

        new_words = words.copy()
        new_words[rnd] = replace

        return new_words


if __name__ == "__main__":
    model = Datasets()
    # print(np.array(model.load_dblp1()).shape)
    # print(np.array(model.load_scholar()).shape)
    model.load_mapping()
    # load_scholar()
    # load_mapping()
    pass
