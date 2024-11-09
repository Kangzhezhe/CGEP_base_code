# -*- coding: utf-8 -*-

'''
此文件是代码运行时必要的参数设置，一般不会再对其进行更改 （可以更改训练的epoch、学习率t_lr）
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # Dataset
    parser.add_argument('--fold', default=1, type=int, help='Fold number used to be test set')
    parser.add_argument('--len_arg', default=300, type=int, help='Sentence length')
    parser.add_argument('--len_temp', default=0, type=int, help='Template length')
    parser.add_argument('--cause_ratio', default=1, type=int, help='cause ratio')
    parser.add_argument('--becausedby_ratio', default=1, type=int, help='be caused by ratio')

    # Model
    # parser.add_argument('--model-name', default='/home/gp3_zhanch/pkg/RoBERTa/RoBERTaForMaskedLM/roberta-base', type=str, help='Model used to be encoder')
    parser.add_argument('--model-name', default='t5',
                        choices=['roberta', 'albert','t5','sedgpl','sedgpl1'], 
                         type=str, help='Model used to be encoder')
    # parser.add_argument('--model-name', default='/home/bbx/NLP/PLM/RoBERTaForMaskedLM/roberta-base', type=str,help='Model used to be encoder')
    # parser.add_argument('--model-name', default='E:/Desktop/yan1/PLM/RoBERTaForMaskedLM/roberta-base', type=str, help='Model used to be encoder')
    parser.add_argument('--vocab_size', default=50265, type=int, help='Size of RoBERTa vocab')

    parser.add_argument('--Sim_ratio', default=0.5, type=float, help='Ratio of similarity in loss function')

    # Prompt and Contrastive Training
    parser.add_argument('--num_epoch', default=2, type=int, help='Number of total epochs to run prompt learning')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for prompt learning')
    parser.add_argument('--t_lr', default=5e-6, type=float, help='Initial lr')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Set Warmup Ratio')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')

    # Others
    parser.add_argument('--seed', default=209, type=int, help='Seed for reproducibility')
    parser.add_argument('--log', default='./out/', type=str, help='Log result file name')
    parser.add_argument('--model', default='./ckpts/', type=str, help='Model parameters result file name')

    parser.add_argument('--train',  action='store_true', help='Train or eval')
    parser.add_argument('--eval',  action='store_true', help='Train or eval')
    parser.add_argument('--test',  action='store_true', help='Train or eval')

    parser.add_argument('--ckpt', default='', type=str, help='Checkpoint file name')
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args
