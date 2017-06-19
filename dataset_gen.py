import logging
import pickle

import numpy as np
import pandas as pd

from utils.utils import dataset_analyser
from word2vec_gensim import word2vec_gensim

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class DatasetGenerator(dataset_analyser):
    def __init__(self):
        super(DatasetGenerator, self).__init__()
        # dataset_analyser.__init__(self)
        self.word2vec_gensim = word2vec_gensim()
        self.matches = self.load_matching_matrix()
        self.n_dblp, self.n_scholar = self.matches.shape
        self.idx_matches = set(zip(self.matches.nonzero()[0], self.matches.nonzero()[1]))

    def load_dblp(self):
        logging.info('load_dblp')
        dblp_sentences = self.dblp_sentences('data/DBLP1.csv')
        return dblp_sentences

    def load_scholar(self):
        logging.info('load_scholar')
        scholar_sentences = self.dblp_sentences('data/Scholar.csv')
        return scholar_sentences

    def load_matching_matrix(self):
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
            # print(d, s)
            id_d = dblp_map[d.lower()]
            id_s = scholar_map[s.lower()]
            matrix[id_d][id_s] = 1

        return matrix

    def dump(self, file, n_pos=500, n_neg=500, dup=False):
        dblp = self.load_dblp()
        scholar = self.load_scholar()

        # d_wv = self.load_dblp_wv()
        # s_wv = self.load_scholar_wv()

        idx_pos = self.pos_samples(n_pos)

        train_pos = []

        train_tmp_neg = [(dblp[d_i], scholar[s_i]) for (d_i, s_i) in idx_pos]
        train_pos.extend(train_tmp_neg)

        if dup:
            for _ in range(dup):
                new_train = [self.permute(pair) for pair in train_tmp_neg]
                train_pos.extend(new_train)

        pos = []

        for pair in train_pos:
            w0 = self.word2vec_gensim.sentence2vec(pair[0])
            w1 = self.word2vec_gensim.sentence2vec(pair[1])

            # print(np.concatenate([w0, w1]))
            w_prod = w0 * w1
            w_diff = np.abs(w0 - w1)
            pos.append(np.concatenate([w_prod, w_diff]))

        pos = [(p, 1) for p in pos]

        idx_neg = self.neg_samples(n_neg)

        train_tmp_neg = [(dblp[d_i], scholar[s_i]) for (d_i, s_i) in idx_neg]
        neg = []
        for pair in train_tmp_neg:
            w0 = self.word2vec_gensim.sentence2vec(pair[0])
            w1 = self.word2vec_gensim.sentence2vec(pair[1])

            # print(np.concatenate([w0, w1]))
            w_prod = w0 * w1
            w_diff = np.abs(w0 - w1)
            neg.append(np.concatenate([w_prod, w_diff]))

        neg = [(p, 0) for p in neg]

        with open(file, mode='wb') as f:
            pickle.dump([pos, neg], f)

        with open(file, mode='rb') as f:
            pos, neg = pickle.load(f)
            print(len(pos))
            print(len(neg))
        pass

    def permute(self, pair):
        d_i, s_i = pair

        s_v2 = []
        s_v2.extend(s_i)
        rnd = np.random.choice(range(len(s_i)))
        s_v2.remove(s_i[rnd])

        return (d_i, s_v2)

    # def next_batch(self, batch_size, threshold=0.5):
    #     n = batch_size
    #     batch_samples = []
    #     for _ in range(n):
    #         rnd = np.random.random_sample()
    #         if rnd <= threshold:
    #             batch_samples.append(self.pos_samples()[0])
    #         else:
    #             batch_samples.append(self.gen_neg())
    #     return batch_samples





    def neg_samples(self, n_sample=1):
        x_n = np.random.choice(range(self.n_dblp), n_sample * 2)
        y_n = np.random.choice(range(self.n_scholar), n_sample * 2)
        xy_n = set(zip(x_n, y_n))

        xy_n.difference_update(self.idx_matches)
        xy_n = np.random.permutation(list(xy_n))

        return xy_n[:n_sample]

    def pos_samples(self, n_sample=1):
        idx_matches = list(self.idx_matches)
        perm_idx_matches = np.random.permutation(idx_matches)

        return perm_idx_matches[:n_sample]


if __name__ == "__main__":
    model = DatasetGenerator()
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

    model.dump('train', n_pos=5000, n_neg=45000)

    # batch = model.next_batch(64)
    # p_batch = [doc for doc in batch if doc in model.idx_matches]
    # print(len(p_batch))
    # load_scholar()
    # load_mapping()
    pass
