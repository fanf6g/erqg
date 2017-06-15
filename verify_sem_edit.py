import logging
import pickle

import numpy as np

import wv_edit
from dataset_gen import DatasetGenerator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

if __name__ == "__main__":
    model = DatasetGenerator()
    m_golden = model.load_matching_matrix()
    mv_metric = wv_edit.sem_edit()

    dblp = model.load_dblp()
    scholar = model.load_scholar()

    dblp = [' '.join(d) for d in dblp]
    scholar = [' '.join(d) for d in scholar]

    m_pred = []
    h_edit = np.vectorize(mv_metric.edist)
    for (i, d) in enumerate(dblp):
        logging.info('{0}, {1}'.format(i, d))
        m_pred.append(h_edit(d, scholar))

    with open('m_pred', mode='wb') as f:
        pickle.dump(m_pred, f)
