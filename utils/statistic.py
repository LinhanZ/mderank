import re
import numpy as np
import os
import json
from tqdm import tqdm
import codecs
from nltk.stem import PorterStemmer
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


porter = PorterStemmer()
en_model = StanfordCoreNLP(r'/home/zhanglinhan.zlh/SIFRank/stanford-corenlp-full-2018-02-27',quiet=True)

stopword_dict = set(stopwords.words('english'))

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


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
                doc = abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data,labels



def matched_label(doc, golds):
    matched_gold = []
    for gold in golds:
        try:
            gold_pattern = re.compile(r"\b" + gold + r"\b")
            matched = gold_pattern.findall(doc)
        except:
            continue
        if len(matched)>=1:
            matched_gold.append(gold)
    matched_count = len(matched_gold)

    return matched_count



dataset_dir = "/home/zhanglinhan.zlh/SIFRank-master/data"
# dataset_name = "nus"

figure_data = ["DUC2001","nus","krapivin"]
x_axis = {}
y_axis = {}
max_length_list = {}
all_max_length = 0
for dataset_name in figure_data:
    if dataset_name == "SemEval2017":
        data, referneces = get_semeval2017_data(dataset_dir + "/SemEval2017/docsutf8", dataset_dir + "/SemEval2017/keys")
    elif dataset_name == "DUC2001":
        data, referneces = get_duc2001_data(dataset_dir+"/DUC2001")
    elif dataset_name == "nus":
        data, referneces = get_long_data(dataset_dir + "/nus/nus_test.json")
    elif dataset_name == "krapivin":
        data, referneces = get_long_data(dataset_dir + "/krapivin/krapivin_test.json")
    elif dataset_name == "kp20k":
        data, referneces = get_short_data(dataset_dir + "/kp20k/kp20k_valid2k_test.json")
    elif dataset_name == "SemEval2010":
        data, referneces = get_short_data(dataset_dir + "/SemEval2010/semeval_test.json")
    else:
        data, referneces = get_inspec_data(dataset_dir+'/Inspec')
    print(len(data))
    print(len(referneces))
    labels = []
    labels_stemed = []
    doc_list = []
    total_words_num = 0
    max_doc_length = 0
    token_num_total = 0
    total_label_num = 0
    total_words_num = 0

    for idx, (key, doc) in tqdm(enumerate(data.items()), desc="Importing document ..."):

        labels_doc = [ref.replace(" \n", "") for ref in referneces[key]]
        labels.append(labels_doc)
        labels_s = []
        set_total_cans = set()
        token_num = 0
        for l in labels_doc:
            tokens = l.split()
            token_num += len(tokens)
            labels_s.append(' '.join(porter.stem(t) for t in tokens))
        total_label_num +=len(labels_doc)
        token_num_total += token_num/len(labels_doc)
        labels_stemed.append(labels_s)
        # if len(doc.split()) > 510:
        #     doc = ' '.join(doc.split()[:510])
        total_words_num += len(doc.split())
        if len(doc.split()) > max_doc_length:
            max_doc_length = len(doc.split())
        doc_list.append(doc)

    max_length_list[dataset_name] = max_doc_length
    if max_doc_length > all_max_length:
        all_max_length = max_doc_length
    print("Avg token num: ", token_num_total/len(doc_list))
    print("Avg labels: ", total_label_num/len(doc_list))
    print("Words num: ", total_words_num/len(doc_list))

    total_matched_count = 0
    total_matched_count = []
    for id, doc in tqdm(enumerate(doc_list[:2]),desc="Making statistic ..."):
        golds = labels[id]
        label_num = len(golds)
        doc_tokens = doc.split()
        doc_length = len(doc.split())
        matched_count = [0]
        for n in range(max_doc_length):
            doc_n = ' '.join(doc_tokens[:n])
            if doc_length < n:
                matched_count.append(matched_count[-1])
            else:
                matched_count_n = matched_label(doc_n, golds)
                matched_count.append(matched_count_n/label_num)

        total_matched_count.append(matched_count)
        total_label_num +=label_num
    total_matched_count = np.array(total_matched_count)
    avg_matched_count = np.nanmean(total_matched_count, axis=0)
    y_axis[dataset_name]= avg_matched_count


    # total_matched_count = total_matched_count/len(doc_list)
    # total_label_num = total_label_num/len(doc_list)
    # print("Avg Matched labels: ", total_matched_count)


colors = {"DUC2001":'goldenrod', "nus":'darkseagreen', "krapivin":'olivedrab'}
x = np.arange(0, all_max_length, 500)
fig,plt = plt.subplots()
for dataset_name, y in y_axis.items():
    x_axis = np.arange(0, max_length_list[dataset_name],500)
    plt.plot(x[0:len(x_axis)], [y[i] for i in x_axis], label= dataset_name, color = colors[dataset_name],  marker='o')
# plt.bar(methods, [19.00,12.33,20.36,17.54,15.02], align = 'center', color=['gold','olivedrab','darkkhaki','darkseagreen','cadetblue'], alpha = .7 )
# plt.xlabel('Doc Words Num', fontsize=18)
# plt.ylabel('Matched Words', fontsize=18)
# plt.xticks(x_index, x)
plt.legend(loc="upper left")
plt.show()
plt.savefig("length_match.png")
# print("{} total_words_num: {}".format(dataset_name, total_words_num/len(doc_list)))

