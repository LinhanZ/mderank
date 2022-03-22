import random
import numpy as np
import re, os
import nltk
from stanfordcorenlp import StanfordCoreNLP
import time
from tqdm import tqdm
from nltk.corpus import stopwords
import codecs
import json
en_model = StanfordCoreNLP(r'/home/zhanglinhan.zlh/SIFRank/stanford-corenlp-full-2018-02-27',quiet=True)
stopword_dict = set(stopwords.words('english'))
MAX_LEN =512

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

def read_file(input_path):
    with open(input_path, 'r', errors='replace_with_space') as input_file:
        return input_file.read()

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


def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data,labels

def get_short_data(file_path="data/kp20k/kp20k_valid2k_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc =abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data,labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text,database="Duc2001")
                data[fname]=text.strip("\n")
                # data[fname] = text
    return data,labels

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

def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text,database="Semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls=text.splitlines()
            labels[left] = ls
            f.close()
    return data,labels

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

if __name__ == '__main__':

    dataset_name = "krapivin"
    dataset_dir = "../SIFRank/data/"
    if dataset_name == "SemEval2017":
        data, referneces = get_semeval2017_data(dataset_dir +dataset_name + "/docsutf8", dataset_dir + dataset_name + "/keys")
    elif dataset_name == "DUC2001":
        data, referneces = get_duc2001_data(dataset_dir + dataset_name)
    elif dataset_name == "nus":
        data, referneces = get_long_data(dataset_dir + dataset_name+ "/nus_test.json")
    elif dataset_name == "krapivin":
        data, referneces = get_long_data(dataset_dir + dataset_name +"/krapivin_test.json")
    elif dataset_name == "kp20k":
        data, referneces = get_short_data(dataset_dir + dataset_name +"/kp20k_valid2k_test.json")
    elif dataset_name == "SemEval2010":
        data, referneces = get_short_data(dataset_dir + dataset_name +"/semeval_test.json")
    else:
        data, referneces = get_inspec_data(dataset_dir + dataset_name)


    docs_pairs = []
    doc_list = []
    key_list = []
    labels = []
    labels_stemed = []
    porter = nltk.PorterStemmer()
    set_cans_num = []
    set_cans = []
    max_can_num = 0
    max_reference_num = 0
    for idx, (key, doc) in enumerate(data.items()):
        set_can = set()
        labels.append([ref.replace(" \n", "") for ref in referneces[key]])
        labels_s = []
        set_total_cans = set()
        for l in referneces[key]:
            tokens = l.split()
            labels_s.append(' '.join(porter.stem(t) for t in tokens))
        if len(labels_s) > max_reference_num:
            max_reference_num = len(labels_s)
        labels_stemed.append(labels_s)
        try:
            text_obj = InputTextObj(en_model, doc)
            doc_list.append(doc)
        except:
            continue
        cans = text_obj.keyphrase_candidate
        candidates = []
        for can, pos in cans:
            set_can.add(can)
        set_cans.append(set_can)
        set_cans_num.append(len(set_can))
        if len(set_can) > max_can_num:
            max_can_num = len(set_can)

    for t in range(100):
        f1_doc =[]
        for idx, doc in enumerate(doc_list):
            print("Calculating doc {} in process {}".format(idx, t))
            doc_p = 0
            doc_r = 0
            doc_f = 0
            doc_cans = list(set_cans[idx])
            doc_ref = labels[idx]
            doc_shuffled_cans = random.sample(doc_cans, len(doc_cans))

            f1_x = []
            for x in range(1,max_can_num):
                candidates = doc_shuffled_cans[:x]
                f1_y = []
                for y in range(1, max_reference_num):
                    selection = random.choices(candidates, k = y)
                    m = 0
                    for s in selection:
                        if s in doc_ref:
                            m +=1
                    p = m/len(candidates)
                    r = m/len(doc_ref)
                    if (p + r == 0.0):
                        f1 = 0
                    else:
                        f1 = 2 * p * r / (p + r)
                    f1_y.append(f1)
                exp_f1_y = sum(f1_y)/len(f1_y)
                f1_x.append(exp_f1_y)
            exp_f1_x = sum(f1_x)/len(f1_x)
            f1_doc.append(exp_f1_x)
        exp_f1_doc = sum(f1_doc)/len(f1_doc)
    print("Expect f1: ", exp_f1_doc)
