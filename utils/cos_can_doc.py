import re
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, BertModel
from torch.utils.data import DataLoader
import pandas as pd
from pke.unsupervised import TextRank
import numpy as np
import logging
import argparse
import codecs
import json
import os
import string
import nltk
from accelerate import Accelerator
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
import itertools

MAX_LEN =512

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
        formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
        th.setFormatter(formatter)

        self.logger.addHandler(sh)  # 代表在屏幕上输出，如果注释掉，屏幕将不输出
        self.logger.addHandler(th)  # 代表在log文件中输出，如果注释掉，将不再向文件中写入数据


class PhraseKPE_Dataset(Dataset):

    def __init__(self, docs_pairs):

        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):

        doc_pair = self.docs_pairs[idx]
        ori_doc = doc_pair[0]
        candidate = doc_pair[1]
        doc_id = doc_pair[2]

        tokenized_ori_doc = self.tokenized_doc(ori_doc, self.tokenizer, candidate)
        tokenized_candidate = self.tokenized_doc(candidate, self.tokenizer, candidate)

        return [tokenized_ori_doc, tokenized_candidate, doc_id]

    def tokenized_doc(self, text, tokenizer, candidate):

        max_len = MAX_LEN

        encoded_dict = tokenizer.encode_plus(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        input_ids = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]
        token_type_ids = encoded_dict["token_type_ids"]

        example = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "candidate": candidate
        }

        return example

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

def generate_doc(dataset_dir, dataset_name):

    doc_list = []
    keyphrases = []
    doc_tok_num = 0
    dataset = load_dataset(dataset_dir)
    for idx, example in enumerate(dataset):
        keywords = example['keywords'].lower()
        abstract = example['abstract']
        doc = abstract
        doc = re.sub('\. ', ' . ', doc)
        doc = re.sub(', ', ' , ', doc)
        doc_tok_num +=len(doc.split(' '))
        doc_list.append(doc)
        keyphrases.append(keywords)
    return doc_list, keyphrases, doc_tok_num/len(dataset)


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


def eval_metric(cans, refs):
    precision_scores, recall_scores, f1_scores = {5: [], 10: [], 15:[]},{5: [], 10: [], 15:[]},{5: [], 10: [], 15:[]}

    stemmer = PorterStemmer()
    references = refs.split(";")
    ref_num = len(references)

    for i, reference in enumerate(references):
        reference = stemmer.stem(reference.lower())
        references[i] = reference.lower()
    candidates_clean = set()
    candidates = []
    for i, can in enumerate(cans):
        can = stemmer.stem(can[0].lower())
        if can in candidates_clean:
            continue
        else:
            candidates_clean.add(can)
            candidates.append(can)


    for topk in [5, 10, 15]:
        m_can = 0
        for i,candidate in enumerate(candidates[:topk]):
            if candidate in references:
                m_can += 1
        micropk = m_can / float(topk)
        micrork = m_can / float(ref_num)

        if micropk + micrork > 0:
            microf1 = float(2 * (micropk * micrork)) / (micropk + micrork)
        else:
            microf1 = 0.0

        precision_scores[topk].append(micropk)
        recall_scores[topk].append(micrork)
        f1_scores[topk].append(microf1)

    return f1_scores, precision_scores, recall_scores, candidates, references, ref_num

def mean_pooling(model_output, attention_mask):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[-2] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def keyphrases_selection(doc_list, references, model, dataloader, log, doc_avg_tok_num):

    model.eval()

    cos_similarity_list = {}
    candidate_list = []
    cos_score_list = []
    doc_id_list = []

    for id, [ori_doc, candidate, doc_id] in enumerate(tqdm(dataloader,desc="Evaluating:")):

        ori_input_ids = torch.squeeze(ori_doc["input_ids"].to('cuda'),1)
        ori_token_type_ids = torch.squeeze(ori_doc["token_type_ids"].to('cuda'),1)
        ori_attention_mask = torch.squeeze(ori_doc["attention_mask"].to('cuda'),1)

        can_input_ids = torch.squeeze(candidate["input_ids"].to('cuda'),1)
        can_token_type_ids = torch.squeeze(candidate["token_type_ids"].to('cuda'),1)
        can_attention_mask = torch.squeeze(candidate["attention_mask"].to('cuda'),1)
        can = candidate["candidate"]


        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            ori_outputs = model(input_ids=ori_input_ids, attention_mask=ori_attention_mask, token_type_ids=ori_token_type_ids, output_hidden_states=True)
            masked_outputs = model(input_ids=can_input_ids, attention_mask=can_attention_mask, token_type_ids=can_token_type_ids, output_hidden_states=True)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            ori_doc_embed = mean_pooling(ori_outputs, ori_attention_mask)
            can_embed = mean_pooling(masked_outputs, can_attention_mask)

            cosine_similarity = torch.cosine_similarity(ori_doc_embed, can_embed, dim=1).cpu()

            doc_id_list.extend(doc_id.numpy().tolist())
            candidate_list.extend(can)
            cos_score_list.extend(cosine_similarity.numpy())

    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["cos"] = cos_score_list

    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)
    total_f1_socres, total_precision_scores, total_recall_scores = np.zeros([len(doc_list),3]),\
                                                                   np.zeros([len(doc_list),3]),\
                                                                   np.zeros([len(doc_list),3])

    doc_num = len(doc_list)
    ref_total_len = 0
    for i in range(len(doc_list)):
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        ranked_keyphrases = doc_results.sort_values(by='cos', ascending=False)
        top_k = ranked_keyphrases.reset_index(drop = True)
        print(top_k)
        top_k = top_k.loc[:, ['candidate']].values.tolist()
        doc_references = references[i]

        f1_scores, precision_scores, recall_scores, candidates_clean, references_clean, ref_num = eval_metric(top_k, doc_references)
        ref_total_len +=ref_num
        for idx, key in enumerate([5,10,15]):
            total_f1_socres[i][idx] = f1_scores[key][0]
            total_precision_scores[i][idx] = precision_scores[key][0]
            total_recall_scores[i][idx] = recall_scores[key][0]
        # if args.local_rank == 0:
        log.logger.info("Doc {} results:\n {}".format(i, candidates_clean))
        log.logger.info("Reference:\n {}".format(references_clean))
        log.logger.info("###########################")
        log.logger.info("F1: {} ".format(f1_scores))
        log.logger.info("P: {} ".format(precision_scores))
        log.logger.info("R: {} ".format(recall_scores))
        log.logger.info("###########################\n")


    log.logger.info("############ Total Result ############")
    for i , key in enumerate([5,10,15]):
        log.logger.info("ref_avg_len: {}".format(ref_total_len/doc_num))
        log.logger.info("doc_avg_len: {}".format(doc_avg_tok_num))
        log.logger.info("@{}".format(key))
        log.logger.info("F1:{}".format(np.mean(total_f1_socres[:,i], axis=0)))
        log.logger.info("P:{}".format(np.mean(total_precision_scores[:,i], axis=0)))
        log.logger.info("R:{}".format(np.mean(total_recall_scores[:,i], axis=0)))
    log.logger.info("#########################\n")
            # doc_sentences = list(filter(None, ex.split('.')))
        #
        #


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset.")
    parser.add_argument("--dataset_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset name.")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Batch size for testing.")
    parser.add_argument("--checkpoints",
                        default=None,
                        type=str,
                        required=False,
                        help="Checkpoint for pre-trained Bert model")
    parser.add_argument("--log_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path for Logging file")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to use CUDA when available")
    args = parser.parse_args()


    log = Logger(args.log_dir + args.dataset_name + '.kpe.log')
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

    doc_list, references, doc_avg_tok_num =  generate_doc(args.dataset_dir, args.dataset_name)

    docs_pairs = []
    for idx, doc in tqdm(enumerate(doc_list),desc="generating pairs..."):
        # candidates, candidates_num = extract_candidate_words(doc)
        extractor = TextRank()
        extractor.load_document(input=doc,
                                language="en",
                                normalization="none")
        extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
        candidates = list(extractor.candidates.keys())

        candidates_num = len(candidates)
        for can in candidates:
            doc_pair = [doc, can, idx]
            docs_pairs.append(doc_pair)

    dataset = PhraseKPE_Dataset(docs_pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    if os.path.exists(args.checkpoints):
        if args.local_rank == 0:
            log.logger.info("Loading Checkpoint ...")
        accelerator = Accelerator()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(torch.load(args.checkpoints))
    log.logger.info("Start Testing ...")
    model.to(device)
    keyphrases_selection(doc_list, references, model, dataloader, log, doc_avg_tok_num)




