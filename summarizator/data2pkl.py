import json
import gzip
import pickle

import config

SIGNALMEDIA_DATA_FILENAME = config.path_data + '/' + 'signalmedia-1m.jsonl.gz'


def data2pkl(acount):
    head_list = []
    desc_list = []

    count = 0
    file = gzip.open(SIGNALMEDIA_DATA_FILENAME)

    for each_line in file:
        record = json.loads(each_line)
        head_list.append(record['title'])
        desc_list.append(record['content'])

        count += 1

        if count >= acount:
            break

    file_open = open(config.path_data + '/' + 'tokens.pkl', 'wb')
    pickle.dump((head_list, desc_list), file_open, -1)
