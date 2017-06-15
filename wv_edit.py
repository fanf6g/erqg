import numpy as np
import numpy.linalg as la

from word2vec_gensim import word2vec_gensim


class sem_edit():
    def __init__(self):
        self.wv = word2vec_gensim()

    def edist(self, s, t):

        def cost(s, t):
            c = 0
            if s != t:
                v_s = self.wv.words2vec(s)
                v_t = self.wv.words2vec(t)

                if np.alltrue(v_s == 0) or np.alltrue(v_t == 0):
                    c = 1.0
                else:
                    # c = 1 - cosine_similarity(v_s, v_t)
                    c = 1 - np.dot(v_s, v_t) / (la.norm(v_s) * la.norm(v_t))
            return c

        words_s = s.split()
        words_t = t.split()
        n_s = len(words_s)
        n_t = len(words_t)

        if n_s == 0 or n_t == 0:
            return max(n_s, n_t)

        d = np.zeros((n_s, n_t))

        for i in range(n_s):
            try:
                d[i, 0] = cost(words_s[i], words_t[0])
            except:
                print(i, s, t)

        for j in range(n_t):
            d[0, j] = cost(words_s[0], words_t[j])

        for j in range(1, n_t):
            for i in range(1, n_s):
                if s[i] == t[j]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min([d[i - 1, j] + 1,  # a deletion
                                   d[i, j - 1] + 1,  # an insertion
                                   d[i - 1, j - 1] + cost(words_s[i], words_t[j])  # a substitution
                                   ])

        return d[n_s - 1, n_t - 1]


if __name__ == "__main__":
    se = sem_edit()

    s = 'database system manage'
    t = 'databases system management'
    print(se.edist(s, t))
    pass
