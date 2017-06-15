import logging

import pandas as pd
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class dataset_analyser:
    def __init__(self):
        self.analyze = CountVectorizer(min_df=1, strip_accents='unicode', stop_words='english').build_analyzer()
        # self.word2vec = word2vec_gensim()

    def dblp_sentences(self, file):
        # file = '../data/DBLP1.csv'
        df = pd.read_csv(file, encoding='iso-8859-1')
        df = df.fillna('')

        df['sent'] = df['authors'] + ' ' + df['title'] + ' ' + df['venue']
        sents = df['sent'].tolist()
        sents = [self.analyze(sent) for sent in sents]
        return sents

    def scholar_sentences(self, file):
        # file = '../data/Scholar.csv'
        data = pd.read_csv(file, encoding='iso-8859-1')
        data = data.fillna('')

        data['sent'] = data['authors'] + ' ' + data['title'] + ' ' + data['venue']
        sents = data['sent'].tolist()
        sents = [self.analyze(sent) for sent in sents]
        return sents

    def dblp_traning_sentences(self):
        client = pymongo.MongoClient('localhost', 27017)
        db = client['dblp']
        tbl = db.get_collection('dblp')

        data = pd.DataFrame(list(tbl.find()))
        data.pop('_id')
        data = data.fillna('')

        data['sent'] = data['author'] + ' ' + data['title'] + ' ' + data['journal']

        sents = data['sent'].tolist()
        sents = [self.analyze(sent) for sent in sents]
        print(sents[0])
        return sents


if __name__ == "__main__":
    file = '../data/DBLP1.csv'

    u = dataset_analyser()

    # print(len(scholar_recs2sent(file)))
    # print(len(dblp_traning_rec2sent()))

    # print(u.().nonzero())

    pass
