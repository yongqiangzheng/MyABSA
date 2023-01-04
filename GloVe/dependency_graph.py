# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import argparse

from spacy.tokens import Doc
from tqdm import tqdm


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.spacy.graph', 'wb')
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':
    process('../datasets/acl-14-short-data/train')
    process('../datasets/acl-14-short-data/test')
    process('../datasets/semeval14/lap14_train')
    process('../datasets/semeval14/lap14_test')
    process('../datasets/semeval14/rest14_train')
    process('../datasets/semeval14/rest14_test')
    process('../datasets/semeval15/rest15_train')
    process('../datasets/semeval15/rest15_test')
    process('../datasets/semeval16/rest16_train')
    process('../datasets/semeval16/rest16_test')
