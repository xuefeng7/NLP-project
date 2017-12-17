import pandas as pd
import pickle as pkl
import string
import numpy as np; np.random.seed(7)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse as sp
from meter import AUCMeter, MAP, MRR, precision

def build_corpus(path):
    title_corpus = []
    body_corpus = []
    idx_corpus = []
    with open('data/' + path, 'r', encoding='utf-8') as src:
        src = src.read().strip().split('\n')
        for line in src:
            context = line.strip().split('\t')
            qid = context.pop(0)
            title_corpus.append(context[0])

            body_corpus.append(context[1] if len(context)!=1 else '')
            idx_corpus.append(int(qid))
    return title_corpus, body_corpus, {k: v for v, k in enumerate(idx_corpus)}

def score_from_idx(idx1, idx2, embedding):
    v1 = embedding[android_idx[idx1]]
    v2 = embedding[android_idx[idx2]]
    return (v1 @ v2.T).toarray()[0][0] / \
            (sp.linalg.norm(v1)*sp.linalg.norm(v2))
        
def read_annotations(pos_path, neg_path, embedding): #, K_neg=20, prune_pos_cnt=20):
    dic = {}
    with open('data/android/' + pos_path) as src:
        src = src.read().strip().split('\n')
        for line in src:
            indices = line.strip().split()
            idx1, idx2 = int(indices[0]), int(indices[1])
            if idx1 not in dic:
                dic[idx1] = {}
                dic[idx1]['score'] = []
                dic[idx1]['target'] = []
            dic[idx1]['score'].append(score_from_idx(idx1, idx2, embedding))
            dic[idx1]['target'].append(1)
    with open('data/android/' + neg_path) as src:
        src = src.read().strip().split('\n')
        for line in src:
            indices = line.strip().split()
            idx1, idx2 = int(indices[0]), int(indices[1])
            if idx1 not in dic:
                dic[idx1] = {}
                dic[idx1]['score'] = []
                dic[idx1]['target'] = []
            dic[idx1]['score'].append(score_from_idx(idx1, idx2, embedding))
            dic[idx1]['target'].append(0)
    return dic

        
def build_eval_android(pos_path, neg_path, embedding):
    pos_scores = []
    with open('data/android/' + pos_path, 'r') as src:
        src = src.read().strip().split('\n')
        for line in src:
            indices = line.strip().split()
            idx1, idx2 = int(indices[0]), int(indices[1])
            pos_scores.append(score_from_idx(idx1, idx2, embedding))
    targets = np.ones(len(pos_scores))
    neg_scores = []
    with open('data/android/' + neg_path, 'r') as src:
        src = src.read().strip().split('\n')
        for line in src:
            indices = line.strip().split()
            idx1, idx2 = int(indices[0]), int(indices[1])
            neg_scores.append(score_from_idx(idx1, idx2, embedding))
    targets = np.concatenate([targets, np.zeros(len(neg_scores))])
    return np.array(pos_scores+neg_scores), targets

def eval_metrics(eval_set):
    labels = [] 
    for qid in eval_set.keys():
        labels.append(np.array(eval_set[qid]['target'])[np.argsort(eval_set[qid]['score'])][::-1])
    print (' Performance MAP', MAP(labels))
    print (' Performance MRR', MRR(labels))
    print (' Performance P@1', precision(1, labels))
    print (' Performance P@5', precision(5, labels))  

android_title, android_body, android_idx = build_corpus('android/corpus.tsv')
ubuntu_title, ubuntu_body, _ = build_corpus('text_tokenized.txt')
vec = TfidfVectorizer().fit(ubuntu_title+ubuntu_body) # l1->.58,.62
title_vec = vec.transform(android_title)
body_vec = vec.transform(android_body)

# tf-idf transform

tf_embedding = (title_vec + body_vec) / 2
dev = read_annotations('dev.pos.txt', 'dev.neg.txt', tf_embedding)
test = read_annotations('test.pos.txt', 'test.neg.txt', tf_embedding)
auc = AUCMeter()
for qid in dev.keys():
    auc.add(np.array(dev[qid]['score']), np.array(dev[qid]['target']))
print('Dev AUC', auc.value(0.05))
eval_metrics(dev)
auc.reset()
for qid in test.keys():
    auc.add(np.array(test[qid]['score']), np.array(test[qid]['target']))
print('Test AUC', auc.value(0.05))
eval_metrics(test)