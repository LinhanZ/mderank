import torch
import numpy as np
import codecs
from tqdm import tqdm
import json
import re

def load_dataset(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                data_list.append(jsonl)
            except:
                raise ValueError

    return data_list

def generate_doc(dataset_dir, save_txt_path, save_label_path):

    doc_list = []
    keyphrases = []
    doc_tok_num = 0
    dataset = load_dataset(dataset_dir)
    txt_file = open(save_txt_path,"a")
    label_file = open(save_label_path,"a")
    for idx, example in enumerate(dataset):
        keywords = example['keywords']
        if type(keywords) == str:
            label_file.write(keywords + "\n")
        else:
            keywords = ';'.join(keywords)
            label_file.write(keywords + "\n")
        abstract = example['abstract']
        doc = abstract
        doc = re.sub('\. ', ' . ', doc)
        doc = re.sub(', ', ' , ', doc)
        doc_tok_num +=len(doc.split(' '))
        txt_file.write(doc + "\n")

        doc_list.append(doc)
        keyphrases.append(keywords)
    txt_file.close()
    label_file.close()



dataset = "/home/zhanglinhan.zlh/unsupervised_bert_kpe/Dataset/json/kp20k/kp20k_train.json"
save_train_txt = "/home/zhanglinhan.zlh/kpe_test_experiment/data/kp20k_train_text.txt"
save_label_txt = "/home/zhanglinhan.zlh/kpe_test_experiment/data/kp20k_train_label.txt"


generate_doc(dataset, save_train_txt, save_label_txt)


