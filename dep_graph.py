# -*- coding: utf-8 -*-

import numpy as np
import spacy
import stanza
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
spacy_nlp = spacy.load('en_core_web_trf')
spacy_nlp.tokenizer = WhitespaceTokenizer(spacy_nlp.vocab)

stanza_nlp = stanza.Pipeline('en', use_gpu=True, tokenize_pretokenized=True, download_method=False)


def spacy_dep_adj(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = spacy_nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def stanza_dep_adj(text):
    doc = stanza_nlp(text)
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


def bert_dep_adj(ori_adj, tokenizer, text_left, aspect, text_right, heter=False):
    left_tokens, term_tokens, right_tokens = [], [], []
    left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []
    if heter:
        heter_tokens = ['POS', 'NEG'] # add your nodes
        heter_tok2ori_map = list(range(0,len(heter_tokens)))
        offset = len(heter_tokens)
    else:
        heter_tokens = []
        heter_tok2ori_map = []
        offset = 0

    for ori_i, w in enumerate(text_left):
        for t in tokenizer.tokenize(w):
            left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
            left_tok2ori_map.append(ori_i + offset)  # * [0, 0, 1, 2, 2]
    offset += len(text_left)
    for ori_i, w in enumerate(aspect):
        for t in tokenizer.tokenize(w):
            term_tokens.append(t)
            term_tok2ori_map.append(ori_i + offset)
    offset += len(aspect)
    for ori_i, w in enumerate(text_right):
        for t in tokenizer.tokenize(w):
            right_tokens.append(t)
            right_tok2ori_map.append(ori_i + offset)

    bert_tokens = heter_tokens + left_tokens + term_tokens + right_tokens
    tok2ori_map = heter_tok2ori_map + left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
    truncate_tok_len = len(bert_tokens)
    tok_adj = np.zeros(
        (truncate_tok_len, truncate_tok_len), dtype='float32')
    for i in range(truncate_tok_len):
        for j in range(truncate_tok_len):
            tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]
    return tok_adj


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph_spacy, idx2graph_stanza = {}, {}
    fout_spacy = open(filename + '.spacy.graph', 'wb')
    fout_stanza = open(filename + '.stanza.graph', 'wb')
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        spacy_adj = spacy_dep_adj(text_left + ' ' + aspect + ' ' + text_right)
        stanza_adj = stanza_dep_adj(text_left + ' ' + aspect + ' ' + text_right)
        idx2graph_spacy[i] = spacy_adj
        idx2graph_stanza[i] = stanza_adj
    pickle.dump(idx2graph_spacy, fout_spacy)
    pickle.dump(idx2graph_stanza, fout_stanza)
    fout_spacy.close()
    fout_stanza.close()


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
