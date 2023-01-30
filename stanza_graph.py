# -*- coding: utf-8 -*-

import numpy as np
import stanza
import pickle

from tqdm import tqdm

nlp = stanza.Pipeline('en', use_gpu=True, tokenize_pretokenized=True, download_method=False)


def dependency_adj_matrix(text):
    doc = nlp(text)
    sent = doc.sentences[0]
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(sent.words)

    for word in sent.words:
        matrix[word.id - 1][word.id - 1] = 1
        if word.head == 0:
            continue
        matrix[word.id - 1][word.head - 1] = 1
        matrix[word.head - 1][word.id - 1] = 1

    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.stanza.graph', 'wb')
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':
    process('datasets/acl-14-short-data/train')
    process('datasets/acl-14-short-data/test')
    process('datasets/semeval14/lap14_train')
    process('datasets/semeval14/lap14_test')
    process('datasets/semeval14/rest14_train')
    process('datasets/semeval14/rest14_test')
    process('datasets/semeval15/rest15_train')
    process('datasets/semeval15/rest15_test')
    process('datasets/semeval16/rest16_train')
    process('datasets/semeval16/rest16_test')
