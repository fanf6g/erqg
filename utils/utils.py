import logging
import string
import unicodedata

import numpy as np
import pandas as pd
import pymongo

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


def unicodeToAscii(s):
    all_letters = string.ascii_letters + string.digits + " /"

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    ).lower()


def dblp_recs2sent(file):
    # file = '../data/DBLP1.csv'
    data = pd.read_csv(file, encoding='iso-8859-1')

    data = data.fillna('')
    data = data.applymap(lambda x: unicodeToAscii(x) if isinstance(x, str) else '')

    data['sent'] = data['authors'] + ' ' + data['title'] + ' ' + data['venue']
    sents = data['sent'].tolist()
    sents = [sent.split() for sent in sents]
    return sents


def scholar_recs2sent(file):
    # file = '../data/Scholar.csv'
    data = pd.read_csv(file, encoding='iso-8859-1')

    data = data.fillna('')
    data = data.applymap(lambda x: unicodeToAscii(x) if isinstance(x, str) else '')

    data['sent'] = data['authors'] + ' ' + data['title'] + ' ' + data['venue']
    sents = data['sent'].tolist()
    sents = [sent.split() for sent in sents]
    return sents


def dblp_traning_rec2sent():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['dblp']
    tbl = db.get_collection('dblp')

    data = pd.DataFrame(list(tbl.find()))
    data.pop('_id')
    data = data.fillna('')
    data = data.applymap(lambda x: unicodeToAscii(x) if x is not None else '')

    data['sent'] = data['author'] + ' ' + data['title'] + ' ' + data['journal']

    sents = data['sent'].tolist()
    sents = [sent.split() for sent in sents]
    return sents


def load_mapping():
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


if __name__ == "__main__":
    file = '../data/DBLP1.csv'

    # print(len(scholar_recs2sent(file)))
    # print(len(dblp_traning_rec2sent()))

    print(load_mapping().nonzero())

    pass
