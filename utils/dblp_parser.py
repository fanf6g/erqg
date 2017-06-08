import logging
from xml import sax

import pymongo

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class DblpHandler(sax.ContentHandler):
    def __init__(self):
        super(DblpHandler, self).__init__()
        self.cnt = 0
        self.current = dict()
        self.lasttag = None
        self.cur_level = 0

        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client['dblp']
        self.tbl = self.db.get_collection('dblp0')

    def startElement(self, name, attrs):
        self.cur_level = self.cur_level + 1
        self.lasttag = name
        for (k, v) in attrs.items():
            self.current[k] = v

    def characters(self, content):
        self.current[self.lasttag] = self.current.get(self.lasttag, ' ') + content

        # print(self.lasttag, content)

    def endElement(self, name):
        self.cur_level = self.cur_level - 1
        if self.cur_level == 1:
            self.current['type'] = name
            print(self.current)
            self.tbl.save(self.current)
            self.current = dict()


if __name__ == "__main__":
    parser = sax.make_parser()
    parser.setContentHandler(DblpHandler())
    parser.parse(open("data/dblp.xml", "r"))
