import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset

from word2vec_gensim import word2vec_gensim


class Dataset_dblp(Dataset):
    def __init__(self):
        self.analyze = CountVectorizer(min_df=1, strip_accents='unicode', stop_words='english').build_analyzer()
        self.data_loading()
        self.word2vec_gensim = word2vec_gensim()
        self.tovec()

    def data_loading(self):
        self.pd_dblp = pd.read_csv('data/DBLP1.csv', encoding='iso-8859-1')
        self.pd_dblp.fillna('')
        self.pd_dblp.pop('year')

    def rec2vec(self, df):
        df = df.fillna('')
        df['sent'] = df['authors'] + ' ' + df['title'] + ' ' + df['venue']
        sents = df['sent'].tolist()
        sents = [self.analyze(sent) for sent in sents]
        # df['sent'] = sents
        return sents

    def tovec(self):
        self.pd_dblp['sent'] = self.rec2vec(self.pd_dblp)
        dblp_vec = [self.word2vec_gensim.sentence2vec(sent) for sent in self.pd_dblp['sent'].tolist()]
        self.pd_dblp['vec'] = dblp_vec

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.pd_dblp)
        # raise NotImplementedError
