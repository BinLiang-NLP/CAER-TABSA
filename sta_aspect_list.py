#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------ 
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

from collections import defaultdict
import nltk

seg_data_path = './dataset/sentihood/data_total.csv'
aspect_list_path = './dataset/sentihood_aspect.list'
orig_data_path = './dataset/sentihood/data.orig'

data_file = open(seg_data_path, 'r')
aspect_file = open(aspect_list_path, 'w')
orig_data_file = open(orig_data_path, 'w')

aspect_dic = defaultdict(int)

lines = data_file.readlines()
for i in range(1,len(lines)):
    sample = lines[i].strip().split(',')
    idx = sample[0]
    orig_sentence = ','.join(sample[1:-3])
    sentence = nltk.word_tokenize(orig_sentence)
    target = sample[-3]
    aspect = sample[-2]
    polarity = sample[-1]
    aspect_dic[aspect] += 1
    orig_data_file.write(target+'\t'+aspect+'\t'+polarity+'\t'+' '.join(sentence)+'\n')
aspect_dic = sorted(aspect_dic.items(), key = lambda a: a[1], reverse=True)
for aspect, count in aspect_dic:
    aspect_file.write(aspect+'\t'+str(count)+'\n')
data_file.close()
orig_data_file.close()
aspect_file.close()
