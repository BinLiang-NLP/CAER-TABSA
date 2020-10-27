# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class CAER(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CAER, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.embed_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.text_dfc = nn.Linear(2*opt.embed_dim, 1)
        self.text_fc = nn.Linear(opt.embed_dim, 1)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.fc_dropout = nn.Dropout(0.3)

    def get_orig_embeddings(self, embeddings, aspect_len):
        batch_size = embeddings.shape[0]
        aspect_len = aspect_len.cpu().numpy()
        orig_embedding = [np.array(self.opt.embed_dim) for i in range(batch_size)]
        for i in range(batch_size):
            t_embed = embeddings[i].cpu().numpy()
            orig_embedding[i] = np.mean(t_embed)
        return orig_embedding

    def step_function(self, x, text_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        text_len = text_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        max_len = max(text_len)
        for i in range(batch_size):
            vector_u = x[i].cpu().detach().numpy()
            context_len = text_len[i]
            mean_u = np.mean(vector_u[:context_len])
            for j in range(context_len):
                if vector_u[j] >= mean_u:
                    weight[i].append(1)
                else:
                    weight[i].append(0)
            for j in range(seq_len-context_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def forward(self, inputs):
        text_indices, aspect_indices, target_indices = inputs
        text_len = torch.sum(text_indices != 0, dim=1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        orig_target_embed = self.embed(target_indices)
        text_out, (_, _) = self.text_lstm(text, text_len)
        text_out = self.text_dfc(text_out) # u (adopt a LSTM layer, performance is better)
        text_out = self.text_fc(text) #u 
        step_out = self.step_function(text_out, text_len) # u'
        #step_out = self.fc_dropout(step_out) # dropout
        output = text * step_out # t~
        return output, orig_target_embed
