# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPooler, BertSelfAttention


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class BERT_GCN(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_GCN, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.gcn1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gcn2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.dense = nn.Linear(2 * opt.bert_dim, opt.polarities_dim)

        self.bert_SA = SelfAttention(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, adj = inputs[0], inputs[1], inputs[2]
        hidden_out, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        x0 = self.dropout(hidden_out)
        x1 = F.relu(self.gcn1(x0, adj))
        x2 = F.relu(self.gcn2(x1, adj))
        self_attention_out = self.bert_SA(x2)
        gcn_pooled_out = self.bert_pooler(self_attention_out)
        r_cat = torch.cat((pooled_output, gcn_pooled_out), dim=-1)
        output = self.dropout(r_cat)
        logits = self.dense(output)
        return logits
