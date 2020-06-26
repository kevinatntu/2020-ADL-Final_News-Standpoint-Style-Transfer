'''
ADL Final - News Standpoint Style Transfer

Reference:
https://github.com/Nrgeup/controllable-text-attribute-transfer
author    = {Ke Wang and Hang Hua and Xiaojun Wan},
title     = {Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation},
booktitle = {NeurIPS},
year      = {2019}
'''

import numpy as np
import os
import random
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from transformers import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, batch_size, id_bos, id_eos, id_unk, max_sequence_length, vocab_size, file_list, label_list, tokenizer):
        self.sentences_batches = []
        self.labels_batches = []

        self.src_batches = []
        self.src_mask_batches = []
        self.tgt_batches = []
        self.tgt_y_batches = []
        self.tgt_mask_batches = []
        self.ntokens_batches = []

        self.num_batch = 0
        self.batch_size = batch_size
        self.pointer = 0
        self.id_bos = '[CLS]'
        self.id_sep = '[SEP]'
        self.id_pad = '[PAD]'
        self.id_unk = '[UNK]'
        self.id_eos = '[EOS]'
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.corpus_max_len = 0
        self.tokenizer = tokenizer

        # load train file
        self.data_label_pairs = []
        self.data = []
        for _index in range(len(file_list)):
            with open(file_list[_index]) as fin:
                for line in fin:
                    line = line.strip()
                    parse_line = self.tokenizer.tokenize(line)
                    parse_line = parse_line[:self.max_sequence_length]
                    #parse_line += [self.id_sep]
                    self.corpus_max_len = max(self.corpus_max_len, len(parse_line))
                    self.data.append([parse_line, label_list[_index]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):   
        sent, label = self.data[index]
        
        encoder_input = [self.id_bos] + sent + [self.id_pad] * (self.corpus_max_len - len(sent)) + [self.id_sep]
        decoder_input = [self.id_bos] + sent + [self.id_pad] * (self.corpus_max_len - len(sent))
        decoder_output = sent + self.tokenizer.tokenize(self.id_eos) + [self.id_pad] * (self.corpus_max_len - len(sent)) 

        encoder_input = self.tokenizer.convert_tokens_to_ids(encoder_input)
        decoder_input = self.tokenizer.convert_tokens_to_ids(decoder_input)
        decoder_output = self.tokenizer.convert_tokens_to_ids(decoder_output)

        return sent, torch.LongTensor(encoder_input), torch.LongTensor(decoder_input), torch.LongTensor(decoder_output), label

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
    
    def collate_fn(self, datas):
        this_batch_sentences = []
        this_batch_labels = [] 

        this_batch_sentences = [data[0] for data in datas]
        src = torch.stack([data[1] for data in datas])
        tgt = torch.stack([data[2] for data in datas])
        tgt_y = torch.stack([data[3] for data in datas])
        this_batch_labels = torch.FloatTensor([data[4] for data in datas])
        
        src_mask = torch.zeros(src.shape, dtype=torch.long)
        src_mask = src_mask.masked_fill(src != 0, 1)
        tgt_mask = self.make_std_mask(tgt, 0)
        ntokens = (tgt_y != 0).data.sum().float()

        #return this_batch_sentences, torch.FloatTensor(this_batch_labels), src, src_mask, tgt, tgt_y, tgt_mask, ntokens
        return this_batch_sentences, this_batch_labels, src, src_mask, tgt, tgt_y, tgt_mask, ntokens

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def id2text_sentence(sen_id, tokenizer, task='twnews'):
    sen_text = []
    tokens = tokenizer.convert_ids_to_tokens(sen_id)
    
    sen_text = ''.join(tokens)
    iidx = sen_text.find('[eos]')
    
    if iidx > 0 and iidx != 0:
        sen_text = sen_text[:iidx]
    return sen_text

def prepare_data(data_path, task_type):
    print("prepare data ...")
    
    #id_to_word, vocab_size = None, 10
    # define train / test file
    train_file_list = []
    train_label_list = []
    if task_type == 'news_china_taiwan':
        train_file_list = [
            data_path + 'train.0', data_path + 'train.1',
            data_path + 'dev.0', data_path + 'dev.1',
        ]
        train_label_list = [
            [0],
            [1],
            [0],
            [1],
        ]

    #return id_to_word, vocab_size, train_file_list, train_label_list
    return train_file_list, train_label_list


if __name__ == '__main__':
    '''
    Test function
    '''
    model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")

    print(model)

    train_file_list, train_label_list = prepare_data('./data/preprocessed/', 'news_china_taiwan')

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", do_lower_case=True)
    
    tokenizer.add_tokens('[EOS]')

    trainData = TextDataset(batch_size=4, id_bos='[CLS]',
                            id_eos='[EOS]', id_unk='[UNK]',
                            max_sequence_length=60, vocab_size=0,
                            file_list=train_file_list, label_list=train_label_list, tokenizer=tokenizer)

    print(len(trainData))
    print(trainData.corpus_max_len)
    BATCH_SIZE = 4
    trainloader = DataLoader(trainData, 
                            batch_size=BATCH_SIZE, 
                            collate_fn=trainData.collate_fn,
                            num_workers=0)

    # test correctness
    data = next(iter(trainloader))

    #user_tensors, item_tensors, label_tensors = data
    this_batch_sentences, this_batch_labels, src_tensors, src_mask_tensors, \
    tgt_tensors, tgt_y_tensors, tgt_mask_tensors, ntokens_tensors = data

    print(f"""
    this_batch_sentences.shape   = {len(this_batch_sentences)} 
    {this_batch_sentences}
    ------------------------
    this_batch_labels.shape = {this_batch_labels.shape}
    {this_batch_labels}
    ------------------------
    src_tensors.shape = {src_tensors.shape}
    {src_tensors}
    ------------------------
    src_mask_tensors.shape = {src_mask_tensors.shape}
    {src_mask_tensors}
    ------------------------
    tgt_tensors.shape = {tgt_tensors.shape}
    {tgt_tensors}
    ------------------------
    tgt_y_tensors.shape = {tgt_y_tensors.shape}
    {tgt_y_tensors}
    ------------------------
    tgt_mask_tensors.shape = {tgt_mask_tensors.shape}
    {tgt_mask_tensors}
    ------------------------
    ntokens_tensors.shape = {ntokens_tensors.shape}
    {ntokens_tensors}
    """)

    print(id2text_sentence(src_tensors[0].tolist(), tokenizer))
    print(id2text_sentence(tgt_tensors[0].tolist(), tokenizer))
    print(id2text_sentence(tgt_y_tensors[0].tolist(), tokenizer))
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = model.forward(src_tensors.to(device), tgt_tensors.to(device), src_mask_tensors.to(device), tgt_mask_tensors.to(device))
    print(out[2].shape)
    print(out[0].shape, out[2][1].shape)
    '''