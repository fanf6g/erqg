import pymongo

from utils import utils

if __name__ == "__main__":
    client = pymongo.MongoClient('localhost', 27017)
    db = client['dblp']
    tbl0 = db.get_collection('dblp0')

    tbl = db.get_collection('dblp')

    cur = tbl0.find({"type": "article"}, {'_id': 0, 'author': 1, 'title': 1, 'journal': 1})
    cnt = 0

    for rec in cur:
        cnt = cnt + 1
        # print(rec)
        new_rec = dict()

        for (k, v) in rec.items():
            new_rec[k] = utils.unicodeToAscii(v.replace('\n', ' ')).strip()

        tbl.save(new_rec)
        if cnt % 10000 == 0:
            print(cnt, new_rec)

        pass
