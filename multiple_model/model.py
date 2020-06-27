'''
ADL Final - News Standpoint Style Transfer

Reference:
https://github.com/Nrgeup/controllable-text-attribute-transfer
author    = {Ke Wang and Hang Hua and Xiaojun Wan},
title     = {Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation},
booktitle = {NeurIPS},
year      = {2019}
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math, copy, time
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from transformers import *
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from transformers import BertTokenizer, AutoTokenizer, AutoModel, AlbertModel
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, args, device, d_model=256, nhead=4, d_ff=1024, nlayers=2, dropout=0.5):
        super(Autoencoder, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout) # encoder's position
        self.pos_decoder = PositionalEncoding(d_model, dropout) # decoder's position

        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, decoder_norm)

        # self.bert_encoder = BertModel.from_pretrained(args.PRETRAINED_MODEL_NAME, output_hidden_states=args.distill_2)
        if args.use_albert:
            self.bert_encoder =  AlbertModel.from_pretrained("clue/albert_chinese_tiny")
            self.bert_embed = self.bert_encoder.embeddings
        # self.tgt_embed = self.bert_embed
            d_vocab = self.bert_encoder.config.vocab_size + 1
            self.tgt_embed = nn.Sequential(Embeddings(d_model, d_vocab), PositionalEncoding(d_model, dropout))
        elif args.use_tiny_bert:
            self.bert_encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-256_A-4")
            self.bert_embed = self.bert_encoder.embeddings
            self.tgt_embed = self.bert_embed
        elif args.use_distil_bert:
            configuration = DistilBertConfig()
            self.bert_encoder = DistilBertModel(configuration)
            self.bert_embed = self.bert_encoder.embeddings
            self.tgt_embed = self.bert_embed
        # self.tgt_embed = self.bert.embeddings
        else:
            self.bert_encoder = BertModel.from_pretrained(args.PRETRAINED_MODEL_NAME, output_hidden_states=args.distill_2)
            self.bert_embed = self.bert_encoder.embeddings
            self.tgt_embed = self.bert_embed

        self.distill_2 = args.distill_2
        self.gru = nn.GRU(d_model, d_model, 1)
        self.lr = nn.Linear(d_model, self.bert_encoder.config.vocab_size + 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        self.lr.bias.data.zero_()
        self.lr.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        outputs = self.bert_encoder(input_ids=src, attention_mask=src_mask)
        if not self.distill_2:
            latent = outputs[0]
        else:
            latent = outputs[2][2]
        
        latent = self.sigmoid(latent)
        # memory = self.position_layer(memory)
        latent = torch.sum(latent, dim=1)  # (batch_size, d_model) 

        # decoder
        src_mask_2 = torch.ones(latent.size(0), 1, 1).long().to(self.device)
        output = self.tgt_embed(tgt)

        output = output.permute(1, 0, 2)
        latent = latent.unsqueeze(0)

        # transformer src shape: [S, N, E]
        # transformer tgt shape: [T, N, E]
        # out shape: [T, N, E]
        out = self.transformer_decoder(output, latent)

        # generator
        out = out.permute(1, 0, 2)
        prob = F.log_softmax(self.lr(out), dim=-1)

        return latent, prob

    def greedy_decode(self, latent, max_len, start_id):
        '''
        latent: (batch_size, max_src_seq, d_model)
        src_mask: (batch_size, 1, max_src_len)
        '''
        batch_size = latent.size(0)
        ys = torch.ones(batch_size, 1).fill_(start_id).long().to(self.device)  # (batch_size, 1)
        #print(ys)
        for i in range(max_len - 1):
            ys = ys.detach()
            ys = ys.to(self.device)
            output = self.tgt_embed(ys)
            output = output.permute(1, 0, 2)
            out = self.transformer_decoder(output, latent.unsqueeze(0))
            out = out.permute(1, 0, 2)
            prob = F.log_softmax(self.lr(out[:, -1]), dim=-1)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        return ys[:, 1:]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Attr_Classifier(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Attr_Classifier, self).__init__()
        self.lr1 = nn.Linear(latent_size, 100)
        self.lr2 = nn.Linear(100, 50)
        self.lr3 = nn.Linear(50, output_size)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out = self.relu(self.lr1(inputs))
        out = self.relu(self.lr2(out))
        out = self.lr3(out)
        out = self.sigmoid(out)

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size


# Original Github's code
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

# Original Github's code
class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())

# Original Github's code
def fgim_attack(model, origin_data, target, ae_model, max_sequence_length, id_bos,
                id2text_sentence, id_to_word, gold_ans, tokenizer, device, task='twnews', save_latent=-1):
    """Fast Gradient Iterative Methods"""

    #dis_criterion = nn.BCELoss(size_average=True)
    dis_criterion = nn.BCELoss(reduction='mean')

    record = ''

    # w= source, 2.0, 4.0, 6.0
    latent_lst = []
    latent_lst.append(origin_data.cpu().detach().numpy())
    # while True:
    for idx, epsilon in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
        it = 0
        data = origin_data
        while True:
            print("epsilon:", epsilon)

            data = data.detach().clone()
            data = data.to(device)  # (batch_size, seq_length, latent_size)
            data.requires_grad_()
            # Set requires_grad attribute of tensor. Important for Attack
            output = model.forward(data)
            loss = dis_criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data = data - epsilon * data_grad
            it += 1
            # data = perturbed_data
            epsilon = epsilon * 0.9

            generator_id = ae_model.greedy_decode(data,
                                                    max_len=max_sequence_length,
                                                    start_id=id_bos)
            generator_text = id2text_sentence(generator_id[0], tokenizer, task)
            print("| It {:2d} | dis model pred {:5.4f} |".format(it, output[0].item()))
            print(generator_text)

            record += "| It {:2d} | dis model pred {:5.4f} |".format(it, output[0].item())
            record += generator_text + '\n'
            if it >= 5:
                if save_latent != -1 and idx in [0, 2, 4]:
                    print("Save latent")
                    latent_lst.append(data.cpu().detach().numpy())
                break
    return record, latent_lst


if __name__ == '__main__':
    pass


