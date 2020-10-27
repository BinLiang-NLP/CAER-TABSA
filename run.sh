#!/bin/bash
#########################################################################
# Author: Bin Liang
# mail: bin.liang@stu.hit.edu.cn
#########################################################################

CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name caer --dataset rest15  --learning_rate 1e-3 --batch_size 16  --num_epoch 30
