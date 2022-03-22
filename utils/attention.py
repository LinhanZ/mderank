from transformers import BertForMaskedLM, BertTokenizer
from bertviz.neuron_view import show
import re
import torch
from numpy import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import argparse
import codecs
import json
import os
import string
import nltk
from stanfordcorenlp import StanfordCoreNLP
from accelerate import Accelerator
import time
from nltk.corpus import stopwords
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby

MAX_LEN =512
en_model = StanfordCoreNLP(r'/home/zhanglinhan.zlh/SIFRank/stanford-corenlp-full-2018-02-27',quiet=True)



stopword_dict = set(stopwords.words('english'))

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info'):

        self.logger = logging.getLogger(filename)
        # # format_str = logging.Formatter(fmt)  # 设置日志格式
        # if args.local_rank == 0 :
        #     level = level
        # else:
        #     level = 'warning'
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        # sh.setFormatter(format_str)  # 设置屏幕上显示的格式

        th = logging.FileHandler(filename,'w')
        # formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
        # th.setFormatter(formatter)

        self.logger.addHandler(sh)  # 代表在屏幕上输出，如果注释掉，屏幕将不输出
        self.logger.addHandler(th)  # 代表在log文件中输出，如果注释掉，将不再向文件中写入数据

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate

class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged, en_model)


def clean_text(text="",database="Inspec"):

    #Specially for Duc2001 Database
    if(database=="Duc2001" or database=="Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    # delete blank line
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'

    return text_new


def get_inspec_data(file_path="data/Inspec"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text = text.replace("%", '')
                text=clean_text(text)
                data[left]=text
            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text=text.replace("\n",' ')
                text=clean_text(text,database="Inspec")
                text=text.lower()
                label=text.split("; ")
                labels[left]=label
    return data,labels

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):

    punct = set(string.punctuation)

    stop_words = set(nltk.corpus.stopwords.words('english'))
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)))
    candidate_phrase = []
    candidates = []
    for word, tag in tagged_words:
        if tag in good_tags and word.lower() not in stop_words and not all(char in punct for char in word):
            candidate_phrase.append(word)
            continue
        else:
            if candidate_phrase:
                candidates.append(candidate_phrase)
                candidate_phrase = []
            else:
                continue

    candiates_num = len(candidates)

    return candidates, candiates_num

def dedup(candidates):
    new_can = {}
    for can in candidates:
        can_set = can.split()
        candidate_len = len(can_set)
        # can = ' '.join(can)
        new_can[can] = candidate_len

    return  new_can

def generate_absent_doc(doc, candidates, idx):

    doc_pairs = []
    #每个文章的candidate， 可能有多个
    doc_candidate = dedup(candidates)
    for id, candidate in enumerate(doc_candidate.keys()):
        candidate_len = doc_candidate[candidate]
        mask = ' '.join(['[MASK]']*candidate_len)
        try:
            candidate_re = re.compile(r"\b" + candidate + r"\b")
            masked_doc = re.sub(candidate_re, mask, doc.lower())
        except:
            continue

        doc_pairs.append([doc.lower(), masked_doc, candidate, idx])
        # print("Candidate: ", candidate)
        # print("Masked Doc {} : {}".format(idx, masked_doc))
        # print("Ori_doc {}: {}".format(idx, doc.lower()))

    return doc_pairs

porter = nltk.PorterStemmer()
data, referneces = get_inspec_data("../SIFRank/data/Inspec")
docs_pairs = []
doc_list = []
labels = []
labels_stemed = []
candidates_num = 0


def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e!=0 else 0.0
    R = float(num_c) / float(num_s) if num_s!=0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def print_PRF(P, R, F1, N):

    log.logger.info("\nN=" + str(N))
    log.logger.info("P=" + str(P))
    log.logger.info("R=" + str(R))
    log.logger.info("F1=" + str(F1))
    return 0

P_5 = R_5 = F1_5 = 0.0
P_10 = R_10 = F1_10 = 0.0
P_15 = R_15 = F1_15 = 0.0
num_c_5 = num_c_10 = num_c_15 = 0
num_e_5 = num_e_10 = num_e_15 = 0
num_s = 0
lamda = 0.0

log = Logger('result/Inspec.kpe.cls.attention.5.log')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

for idx, (key, doc) in enumerate(data.items()):
    set_can = set()
    labels_o = []
    for ref in referneces[key]:
        labels_o.append(ref.replace(" \n",""))
    labels.append(labels_o)
    labels_s = []
    for l in labels_o:
        tokens = l.split()
        labels_s.append(' '.join(porter.stem(t) for t in tokens))
    labels_stemed.append(labels_s)
    doc_list.append(doc)
    encoded_dict = tokenizer.encode_plus(
        doc,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True
    )
    outputs = model(**encoded_dict, output_attentions=True)
    attention = outputs.attentions[-1]


    text_obj = InputTextObj(en_model, doc)
    cans = text_obj.keyphrase_candidate
    candidates = []
    for can, pos in cans:
        candidates.append(can)
        set_can.add(can.lower())
    candidates_num += len(set_can)
    candidate_att = []
    for can in set_can:
        doc_pair = [doc, can, idx]
        docs_pairs.append(doc_pair)
        can_tokens = tokenizer.tokenize(can)
        can_token_ids = tokenizer.convert_tokens_to_ids(can_tokens)
        print("can token ids: ", can_token_ids)
        can_token_num = len(can_token_ids)
        head_att = []
        for i in range(attention.size(1)):
            att_matrix = attention[0][i][0].detach().numpy()
            token_ids = encoded_dict["input_ids"].squeeze()
            cans_pos = []
            for idx in can_token_ids:
                cans_pos.append(torch.nonzero(token_ids == idx))
            print("can positions: ", cans_pos)
            can_att_avg = 0
            min_app = 0
            positions = []
            pos_set = set()
            for c in range(len(cans_pos)):
                for p in cans_pos[c]:
                    positions.append(int(p[0].numpy()))
                    pos_set.add(int(p[0].numpy()))

            positions = sorted(positions)
            pos_set = sorted([p for p in pos_set])
            print("pos set: ", pos_set)
            fun = lambda x: x[1] - x[0]
            can_att = []
            for k, g in groupby(enumerate(pos_set), fun):
                l1 = [j for i, j in g]  # 连续数字的列表
                print("l1: ", l1)
                print("can token num: ", can_token_num)
                if len(l1) >= can_token_num:
                    for l in l1:
                        can_att_avg +=att_matrix[l]
                    can_att.append(can_att_avg/len(l1))
                if len(can_att) == 0:
                    print("can tokens: ", can_token_ids)
                    print("tokens: ", token_ids)
            # head_att.append(sum(can_att)/len(can_att)) #不同head的attention
            if len(can_att) == 0:
                break
            else:
                head_att.append(sum(can_att)/len(can_att))
        candidate_att.append(head_att)
    heads_num=np.arange(attention.size(1))
    candidate_att = pd.DataFrame(candidate_att,index=set_can, columns=heads_num)
    candidate_att["max"] = candidate_att.max(axis=1) #取出该最大值
    candidate_att["mean"] = candidate_att.mean(axis=1)

    #
    # for i in heads_num:
    results = candidate_att.sort_values(by=[5],ascending=False) #降序排列
    top_k = results.index.values.tolist()
    doc_labels = [ref.replace(" \n", "") for ref in referneces[key]]
    log.logger.info("Doc {} Head {} \nTop15 Candidates: {} \nReference: {}".format(idx, 5, top_k[:15], doc_labels))

    j = 0
    matched_candidate = []
    for temp in top_k[0:15]:
        tokens = temp.lower().split()
        tt = ' '.join(porter.stem(t) for t in tokens)
        if (tt in labels_s or temp in labels_o):
            if (j < 5):
                num_c_5 += 1
                num_c_10 += 1
                num_c_15 += 1

            elif (j < 10 and j >= 5):
                num_c_10 += 1
                num_c_15 += 1

            elif (j < 15 and j >= 10):
                num_c_15 += 1
            matched_candidate.append(temp)
        j += 1

    if (len(top_k[0:5]) == 5):
        num_e_5 += 5
    else:
        num_e_5 += len(top_k[0:5])

    if (len(top_k[0:10]) == 10):
        num_e_10 += 10
    else:
        num_e_10 += len(top_k[0:10])

    if (len(top_k[0:15]) == 15):
        num_e_15 += 15
    else:
        num_e_15 += len(top_k[0:15])

    num_s += len(labels_s)
    log.logger.info("Matched Candidate: {} \n".format(matched_candidate))


p, r, f = get_PRF(num_c_5, num_e_5, num_s)
print_PRF(p, r, f, 5)
p, r, f = get_PRF(num_c_10, num_e_10, num_s)
print_PRF(p, r, f, 10)
p, r, f = get_PRF(num_c_15, num_e_15, num_s)
print_PRF(p, r, f, 15)


en_model.close()
