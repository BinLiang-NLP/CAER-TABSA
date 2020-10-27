# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_target_indices = []
        batch_polarity = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, target_indices, polarity = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['target_indices'], item['polarity']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            target_padding = [0] * (max_len - len(target_indices))

            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_target_indices.append(target_indices + target_padding)

            batch_polarity.append(polarity)
        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'target_indices': torch.tensor(batch_target_indices), \
                'polarity': torch.tensor(batch_polarity), \
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
