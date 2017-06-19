import logging
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from utils.utils import dataset_analyser
from word2vec_gensim import word2vec_gensim

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class DatasetGenerator2(dataset_analyser):
    def __init__(self):
        super(DatasetGenerator2, self).__init__()

        self.data_loading()
        self.word2vec_gensim = word2vec_gensim()
        self.tovec()

        # self.matches = self.load_matching_matrix()
        # self.n_dblp, self.n_scholar = self.matches.shape
        # self.idx_matches = set(zip(self.matches.nonzero()[0], self.matches.nonzero()[1]))

    def data_loading(self):
        self.pd_dblp = pd.read_csv('data/DBLP1.csv', encoding='iso-8859-1')
        self.pd_dblp.fillna('')
        self.pd_dblp.pop('year')
        # self.pd_dblp['sent'] = self.dblp_sent(self.pd_dblp)

        self.pd_schol = pd.read_csv('data/Scholar.csv', encoding='iso-8859-1')
        self.pd_schol.fillna('')
        self.pd_schol.pop('year')
        # self.pd_schol['sent'] = self.scholar_sent(self.pd_schol)

        self.pd_mapping = pd.read_csv('data/DBLP-Scholar_perfectMapping.csv', encoding='iso-8859-1')

        id_pd_dblp = self.pd_dblp['id'].tolist()
        id_pd_schol = self.pd_schol['id'].tolist()
        id_dblp = self.pd_mapping['idDBLP'].tolist()
        id_schol = self.pd_mapping['idScholar'].tolist()

        mapping_dblp_idx = [id_pd_dblp.index(i) for i in id_dblp]
        mapping_schol_idx = [id_pd_schol.index(i) for i in id_schol]

        matrix = np.zeros((len(id_pd_dblp), len(id_pd_schol)))
        for (d, s) in zip(mapping_dblp_idx, mapping_schol_idx):
            matrix[d][s] = 1

        self.matches = csc_matrix(matrix)

        # print(self.matches.sum(axis=1))

    def tovec(self):
        self.pd_dblp['sent'] = self.dblp_sent(self.pd_dblp)
        self.pd_schol['sent'] = self.scholar_sent(self.pd_schol)

        dblp_vec = [self.word2vec_gensim.sentence2vec(sent) for sent in self.pd_dblp['sent'].tolist()]
        schol_vec = [self.word2vec_gensim.sentence2vec(sent) for sent in self.pd_schol['sent'].tolist()]

        self.pd_dblp['vec'] = dblp_vec
        self.pd_schol['vec'] = schol_vec

    def compose(self, v1, v2):
        return np.concatenate([v1 * v2, np.abs(v1 - v2)])

    # def load_dblp(self):
    #     logging.info('load_dblp')
    #     dblp_sentences = self.dblp_sentences('data/DBLP1.csv')
    #     return dblp_sentences
    #
    # def load_scholar(self):
    #     logging.info('load_scholar')
    #     scholar_sentences = self.dblp_sentences('data/Scholar.csv')
    #     return scholar_sentences
    #
    # def load_matching_matrix(self):
    #
    #     id_dblp = self.pd_dblp['id'].tolist()
    #     id_dblp = list(map(lambda x: x.lower(), id_dblp))
    #
    #     id_scholar = self.pd_schol['id'].tolist()
    #     id_scholar = list(map(lambda x: x.lower(), id_scholar))
    #
    #     dblp_map = dict([(v, i) for (i, v) in enumerate(id_dblp)])
    #     scholar_map = dict([(v, i) for (i, v) in enumerate(id_scholar)])
    #
    #     matrix = np.zeros((len(id_dblp), len(id_scholar)))
    #
    #     list_dblp = self.pd_mapping['idDBLP'].tolist()
    #     list_scholar = self.pd_mapping['idScholar'].tolist()
    #
    #     for (d, s) in zip(list_dblp, list_scholar):
    #         # print(d, s)
    #         id_d = dblp_map[d.lower()]
    #         id_s = scholar_map[s.lower()]
    #         matrix[id_d][id_s] = 1
    #
    #     return matrix
    #
    def dump(self, file, n_pos=500, n_neg=500, dup=False):
        idx_pos = self.pos_samples(n_pos)
        idx_neg = self.neg_samples(n_neg)

        train_pos = [self.compose(self.pd_dblp['vec'].iloc[d_i], self.pd_schol['vec'].iloc[s_i])
                     for (d_i, s_i) in idx_pos]
        pos = [(p, 1) for p in train_pos]

        train_neg = [self.compose(self.pd_dblp['vec'].iloc[d_i], self.pd_schol['vec'].iloc[s_i])
                     for (d_i, s_i) in idx_neg]

        neg = [(p, 0) for p in train_neg]

        with open(file, mode='wb') as f:
            pickle.dump([pos, neg], f)

        with open(file, mode='rb') as f:
            pos, neg = pickle.load(f)
        print(len(pos))
        print(len(neg))

    def onerow(self, n):
        vd = self.pd_dblp['vec'].iloc[n]
        vs_list = self.pd_schol['vec'].tolist()

        features = [self.compose(vd, vs) for vs in vs_list]
        labels = self.matches[n].toarray()[0].tolist()

        return features, labels

    def neg_samples(self, n_sample=1):
        idx_matches = np.array(self.matches.nonzero()).T
        idx_matches = [(pair[0], pair[1]) for pair in idx_matches]

        n_dblp, n_schol = self.matches.shape

        x_n = np.random.choice(range(n_dblp), n_sample * 2)
        y_n = np.random.choice(range(n_schol), n_sample * 2)
        xy_n = set(zip(x_n, y_n))

        xy_n.difference_update(list(idx_matches))
        xy_n = np.random.permutation(list(xy_n))

        return xy_n[:n_sample]

    def pos_samples(self, n_sample=1):
        idx_matches = np.array(self.matches.nonzero()).T
        perm_idx_matches = np.random.permutation(idx_matches)

        return [(pair[0], pair[1]) for pair in (perm_idx_matches[:n_sample])]


if __name__ == "__main__":
    model = DatasetGenerator2()

    # model.pos_samples(4)
    # model
    # print(np.array(model.load_dblp1()).shape)
    # print(np.array(model.load_scholar()).shape)
    # print(model.matches)
    # pos = model.pos_samples(10)[0]
    # neg = model.neg_samples(10)[0]
    # print(pos)
    # print(model.idx_matches)
    # print(tuple(pos) in model.idx_matches)
    # print(neg)
    # print(tuple(neg) in model.idx_matches)

    model.dump('train', n_pos=5000, n_neg=5000)
    # print(model.onerow(3))

    # batch = model.next_batch(64)
    # p_batch = [doc for doc in batch if doc in model.idx_matches]
    # print(len(p_batch))
    # load_scholar()
    # load_mapping()
    pass
