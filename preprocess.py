import os

from nltk.corpus import stopwords
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


def read_file():

    splits = {'train': [('pos', 3754), ('neg', 4724)]}

    data = {}
    for d in splits:
        document_list = []
        targets = []
        for lb in splits[d]:
            folder = 'data/' + d + '/' + lb[0] + '/'
            for i in range(lb[1]):
                f = open(folder + str(i + 1) + '.txt', 'rb')

                # paragraph_list = []
                doc = []

                for line in f.readlines():
                    # paragraph_list.append(line.strip().decode('latin1'))
                    doc.append(tokenize.sent_tokenize(clean_str_simple_version(line.strip().decode('latin1'))))
                f.close()

                document_list.append(doc)
                if lb[0] == 'pos':
                    targets.append(0)
                elif lb[0] == 'neg':
                    targets.append(1)
                else:
                    print(lb)

                # max_num_sentence = show_statisctic(paragraph_list)

        data[d] = (document_list, targets)

    data = clean_document(data)

    word_freq = Counter()
    word_set = set()

    for d in data:
        for document in data[d][0]:
            for paragraph in document:
                for sentence in paragraph:
                    for word in sentence:
                        word_set.add(word)
                        word_freq[word] += 1

    vocab = ['<pad>'] + list(word_set)
    vocab_size = len(vocab)

    vocab_dic = {}
    for i, word in enumerate(word_set):
        vocab_dic[word] = i

    print('Total_number_of_words: ' + str(len(vocab)))

    for d in data:
        id_docs = []
        for document in data[d][0]:
            id_doc = []
            for paragraph in document:
                id_paragraph = []
                count_num = 0
                for sentence in paragraph:
                    id_sent = []
                    for word in sentence:
                        id_sent.append(vocab_dic[word])

                    id_paragraph.append(id_sent)
                    count_num += len(id_sent)

                id_doc.append(id_paragraph)
            id_docs.append(id_doc)
        data[d] = (id_docs, data[d][1])

    return data, vocab_dic


def get_embedding(word_to_id):
    filename = '../glove.840B.300d.txt'
    dim = 300

    print('vocab size :', len(word_to_id))
    count = 0
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_id), dim))
    with open(filename, encoding='utf-8') as fp:
        for line in fp:
            elements = line.strip().split()
            word = elements[0]
            if word in word_to_id:
                try:
                    embeddings[word_to_id[word]] = [float(v) for v in elements[1:]]
                    count += 1
                except ValueError:
                    pass
    print('got embeddings of :', count)
    embeddings[0] = np.zeros(dim, dtype='float32')
    return embeddings


def get_data():
    pickle_file = './data/dump.pkl'
    if os.path.exists(pickle_file):
        return pickle.load(open(pickle_file, 'rb'))

    data, vocab_dic = read_file()

    train_dev_x = data['train'][0]
    train_dev_y = data['train'][1]

    train_x = []
    train_y = []
    dev_x = []
    dev_y = []

    train_dev_x_size = len(train_dev_x)
    train_dev_idx = np.random.permutation(train_dev_x_size)
    dev_size = int(train_dev_x_size * .1)
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

    word_vectors = get_embedding(vocab_dic)

    pickle_data = (data, word_vectors)

    pickle.dump(pickle_data, open(pickle_file, 'wb'))

    return pickle_data


def main():
    get_data()


if __name__ == '__main__':
    main()
