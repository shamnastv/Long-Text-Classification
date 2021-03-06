from nltk.corpus import stopwords

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import *


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_simple_version(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def show_statisctic(clean_docs):
    min_len = 10000
    aver_len = 0
    max_len = 0
    num_sentence = sum([len(i) for i in clean_docs])
    ave_num_sentence = num_sentence * 1.0 / len(clean_docs)

    for doc in clean_docs:
        for sentence in doc:
            temp = sentence
            aver_len = aver_len + len(temp)

            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)

    aver_len = 1.0 * aver_len / num_sentence

    print('min_len_of_sentence : ' + str(min_len))
    print('max_len_of_sentence : ' + str(max_len))
    print('min_num_of_sentence : ' + str(min([len(i) for i in clean_docs])))
    print('max_num_of_sentence : ' + str(max([len(i) for i in clean_docs])))
    print('average_len_of_sentence: ' + str(aver_len))
    print('average_num_of_sentence: ' + str(ave_num_sentence))
    print('Total_num_of_sentence : ' + str(num_sentence))

    return max([len(i) for i in clean_docs])


def clean_document(data):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    stemmer = WordNetLemmatizer()

    word_freq = Counter()

    for d in data:
        for paragraphs in data[d][0]:
            for doc_sentences in paragraphs:
                for sentence in doc_sentences:
                    temp = word_tokenize(clean_str(sentence))
                    temp = ' '.join([stemmer.lemmatize(word) for word in temp])

                    words = temp.split()
                    for word in words:
                        word_freq[word] += 1

    for d in data:
        highbar = word_freq.most_common(10)[-1][1]
        clean_docs = []
        for document in data[d][0]:
            clean_doc = []
            for paragraph in document:
                clean_paragraph = []
                count_num = 0
                for sentence in paragraph:
                    temp = word_tokenize(clean_str(sentence))
                    temp = ' '.join([stemmer.lemmatize(word) for word in temp])

                    words = temp.split()
                    clean_sent = []
                    for word in words:
                        if (word not in stop_words) and (word_freq[word] >= 5) and (word_freq[word] < highbar):
                            clean_sent.append(word)

                    clean_paragraph.append(clean_sent)
                    count_num += len(clean_sent)

                clean_doc.append(clean_paragraph)
            clean_docs.append(clean_doc)
        data[d] = (clean_docs, data[d][1])

    return data

