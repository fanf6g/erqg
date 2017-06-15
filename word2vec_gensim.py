import logging
import os

import numpy as np
from gensim.models import Word2Vec

from utils.utils import dataset_analyser

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class word2vec_gensim():
    def __init__(self):
        self.dataset_utils = dataset_analyser()
        filename = 'word2vec_gensim'

        if not os.path.exists(filename):
            self.train_word2vec(filename)

        logging.info('loading word2vec_gensim')
        self.w2v = Word2Vec.load(filename)  # you can continue training with the loaded model!

    def train_word2vec(self, filename='word2vec_gensim'):
        logging.info('loading data')

        sentences = self.dataset_utils.dblp_traning_sentences()

        logging.info('traning word2vec_gensim')
        model = Word2Vec(sentences, size=100, window=3, min_count=1, workers=4)

        logging.info('saving word2vec_gensim')
        model.save(filename)

    def words2vec(self, word):
        try:
            v = self.w2v.wv[word]
        except:
            v = np.zeros((self.w2v.vector_size))

        return v

    def sentence2vec(self, sentence):

        v = [self.words2vec(word) for word in sentence]
        v = np.array(v)
        return np.average(v, axis=0)


if __name__ == "__main__":
    model = word2vec_gensim()
    word1 = ['database']
    word2 = ['database']
    word3 = ['fanfengfeng']
    # print(model.w2v.wv[word1])  # KeyedVectors
    print(model.words2vec(word1))
    print(model.words2vec(word2))
    print(model.sentence2vec(word1))
    print(model.sentence2vec(word2))

    print(model.words2vec(word3))  # Raise KeyError: "word 'fanfengfeng' not in vocabulary"
    print(model.w2v.most_similar(word1))
