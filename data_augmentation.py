import numpy as np

from utils import utils
from word2vec_gensim import word2vec_gensim


class Data_augment():
    def __init__(self):
        self.word2vec = word2vec_gensim()

    def augment(self, words):
        '''
        使用近义词,随机替换句子中的token, 用于 Data augmentation.
        :param sentence:
        :return:
        '''

        rnd = np.random.choice(range(len(words)))
        rand_word = words[rnd]

        replace = self.word2vec.w2v.most_similar([rand_word])[0][0]
        print(rand_word, replace)

        new_words = words.copy()
        new_words[rnd] = replace

        return new_words


if __name__ == "__main__":
    sents_d = utils.dblp_recs2sent('data/DBLP1.csv')
    sents_s = utils.dblp_recs2sent('data/Scholar.csv')

    non_zero = utils.load_mapping().nonzero()
    print(non_zero)
    print(non_zero[0][0])
    print(non_zero[1][0])

    print(sents_d[non_zero[0][0]])
    print(sents_s[non_zero[1][0]])

    model = Data_augment()
    v = model.augment(sents_d[0])
    print(sents_d[0])
    print(v)

    scores = []

    for non in np.array(non_zero).T:
        v_d = model.word2vec.sentence2vec(sentence=' '.join(sents_d[non[0]]))
        v_s = model.word2vec.sentence2vec(sentence=' '.join(sents_s[non[1]]))
        score = np.inner(v_d, v_s)
        scores.append(score)
        print(score)
        pass

    scores = np.array(scores)
    print(scores.mean())
    print(scores.std())
    print(scores.max())
    print(scores.min())
    #
    v_d = model.word2vec.sentence2vec(sentence=' '.join(sents_d[non_zero[0][2]]))
    v_s = model.word2vec.sentence2vec(sentence=' '.join(sents_s[non_zero[1][2]]))

    print(v_d)
    print(v_s)

    print(v_d.shape)

    print(np.inner(v_d, v_s))

    print(sents_d[0])
    print(model.word2vec.sentence2vec(sentence=' '.join(sents_d[0])))
    pass
