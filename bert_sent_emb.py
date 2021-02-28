import os

import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, show_statisctic, clean_document, clean_str_simple_version
import collections
from collections import Counter
import random
import numpy as np
import pickle
from nltk import tokenize
from sentence_transformers import SentenceTransformer


def read_file():

    # model = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cuda:0')
    model = SentenceTransformer('/scratch/shamnast/stsb-roberta-large/', device='cuda:0')
    splits = {'train': [('pos', 3754), ('neg', 4724)], 'test': [('u', 3000)]}
    data = {}
    for d in splits:
        document_list = []
        targets = []
        for lb in splits[d]:
            folder = 'data/' + d + '/' + lb[0] + '/'
            for i in range(lb[1]):
                file = folder + str(i + 1) + '.txt'
                if i % 20 == 0:
                    print(file, flush=True)
                f = open(file, 'rb')

                # paragraph_list = []
                doc = []

                for line in f.readlines():
                    sentences = tokenize.sent_tokenize(clean_str_simple_version(line.strip().decode('latin1')))
                    if len(sentences) == 0:
                        continue
                    embeddings = model.encode(sentences, show_progress_bar=False)
                    doc.append(embeddings)
                f.close()

                document_list.append(doc)
                if lb[0] == 'pos':
                    targets.append(0)

                elif lb[0] == 'neg':
                    targets.append(1)

                elif lb[0] == 'u':
                    targets.append(2)
                else:
                    print(lb)

                # max_num_sentence = show_statisctic(paragraph_list)

        data[d] = (document_list, targets)

    return data


def get_data(pickle_file='/scratch/shamnast/dump3.pkl'):
    dev_prop = .1
    if os.path.exists(pickle_file):
        return pickle.load(open(pickle_file, 'rb'))

    data = read_file()

    train_dev_x = data['train'][0]
    train_dev_y = data['train'][1]

    train_x = []
    train_y = []
    dev_x = []
    dev_y = []

    train_dev_x_size = len(train_dev_x)
    train_dev_idx = np.random.permutation(train_dev_x_size)
    dev_size = int(train_dev_x_size * dev_prop)
    train_size = train_dev_x_size - dev_size
    train_idx = train_dev_idx[:train_size]
    dev_idx = train_dev_idx[train_size:]

    for i in train_idx:
        train_x.append(train_dev_x[i])
        train_y.append(train_dev_y[i])

    for i in dev_idx:
        dev_x.append(train_dev_x[i])
        dev_y.append(train_dev_y[i])

    data['train'] = (train_x, train_y)
    data['dev'] = (dev_x, dev_y)

    pickle.dump(data, open(pickle_file, 'wb'))

    return data


def main():
    get_data(pickle_file='/scratch/shamnast/dump3.pkl')


if __name__ == '__main__':
    main()
