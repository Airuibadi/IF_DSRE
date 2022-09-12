#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 20/08/03 15:52:16

@author: Ziyin Huang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import BaseEncoder

from nltk import word_tokenize

class BiLstmEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=500, 
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 dropout=0.0,
                 num_layer = 1,
                 bidirectional = True,
                 mask_entity=False) :
        super().__init__(token2id, max_length, hidden_size, word_size,
                position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.one_direc_hidden_size = hidden_size
        self.hidden_size = self.one_direc_hidden_size*(2 if bidirectional else 1)
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.input_size, self.one_direc_hidden_size, num_layer, batch_first=True, bidirectional=bidirectional)
        self.h_0 = nn.Parameter(torch.randn(num_layer*(2 if bidirectional else 1), 1, self.one_direc_hidden_size)) 
        self.c_0 = nn.Parameter(torch.randn(num_layer*(2 if bidirectional else 1), 1, self.one_direc_hidden_size)) 
        self.attW = nn.Linear(self.hidden_size, 1)

    def forward(self, token, pos1, pos2) :
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1), 
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        h_0 = torch.cat([self.h_0 for _ in range(x.shape[0])], 1)
        c_0 = torch.cat([self.c_0 for _ in range(x.shape[0])], 1)
        output, (hn, cn) = self.rnn(x, (h_0, c_0))
        a = self.attW(torch.tanh(output))
        x = torch.bmm(a.transpose(1,2), output).squeeze(1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)
