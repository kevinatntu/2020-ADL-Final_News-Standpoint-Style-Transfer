'''
ADL Final - News Standpoint Style Transfer

Reference:
https://github.com/Nrgeup/controllable-text-attribute-transfer
author    = {Ke Wang and Hang Hua and Xiaojun Wan},
title     = {Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation},
booktitle = {NeurIPS},
year      = {2019}
'''
import time
import argparse
import math
import os
import torch
import torch.nn as nn
from torch import optim
import numpy
import matplotlib
from matplotlib import pyplot as plt
from transformers import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import pickle
import numpy as np
from pathlib import Path

# Import your model files.
from model import NoamOpt, LabelSmoothing, fgim_attack, Autoencoder, Attr_Classifier
from data import prepare_data, id2text_sentence, TextDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--id_pad', type=str, default='[PAD]')
parser.add_argument('--id_unk', type=str, default='[UNK]')
parser.add_argument('--id_bos', type=str, default='[CLS]')
parser.add_argument('--id_eos', type=str, default='[EOS]')
parser.add_argument('--task', type=str, default='news_china_taiwan', help='Specify datasets.')
parser.add_argument('--data_path', type=str, default='', help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_sequence_length', type=int, default=60)
parser.add_argument('--num_layers_AE', type=int, default=2)
parser.add_argument('--transformer_model_size', type=int, default=768) # To match BERT
parser.add_argument('--transformer_ff_size', type=int, default=1024)
parser.add_argument('--latent_size', type=int, default=768) # To match BERT
parser.add_argument('--label_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--current_save_path', type=Path, default=Path("./save"))
parser.add_argument('--training', action="store_true")
parser.add_argument('--load_trainData', action="store_true")
parser.add_argument('--load_prev', action="store_true")
parser.add_argument('--PRETRAINED_MODEL_NAME', type=str, default="hfl/chinese-bert-wwm-ext")
parser.add_argument('--fix_first_6', action="store_true")
parser.add_argument('--fix_last_6', action="store_true")
parser.add_argument('--eval_negative', action="store_true")
parser.add_argument('--eval_positive', action="store_true")
parser.add_argument('--distill_2', action="store_true")
parser.add_argument('--save_latent', type=int, default=-1)
parser.add_argument('--save_latent_num', type=int, default=49)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_log(ss):
    now_time = time.strftime("[%Y-%m-%d %H:%M:%S]: ", time.localtime())
    #print(now_time + ss)
    with open(args.log_file, 'a') as f:
        f.write(now_time + str(ss) + '\n')
    return


def add_output(ss):
    with open(args.output_file, 'a') as f:
        f.write(str(ss) + '\n')
    return


def preparation():
    # set model save path
    if not os.path.exists(args.current_save_path):
        os.mkdir(args.current_save_path)
    
    args.log_file = args.current_save_path / 'train_log.txt'
    args.output_file = args.current_save_path / 'eval_log.txt'

    if not args.load_prev: # delete last record if not loading previous model
        if os.path.exists(args.log_file):
            os.remove(args.log_file)
        if os.path.exists(args.output_file):
            os.remove(args.output_file)            

    # set task type
    if args.task == 'news_china_taiwan':
        args.data_path = './data/'
    else:
        raise TypeError('Unsupported task type!')

    # prepare data
    args.train_file_list, args.train_label_list = prepare_data(data_path=args.data_path, task_type=args.task)


def train_iters(ae_model, dis_model):
    tokenizer = BertTokenizer.from_pretrained(args.PRETRAINED_MODEL_NAME, do_lower_case=True)
    tokenizer.add_tokens('[EOS]')
    bos_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]

    ae_model.bert_encoder.resize_token_embeddings(len(tokenizer))
    #print("[CLS] ID: ", bos_id)

    print("Load trainData...")
    if args.load_trainData and os.path.exists('./{}_trainData.pkl'.format(args.task)):
        with open('./{}_trainData.pkl'.format(args.task), 'rb') as f:
            trainData = pickle.load(f)
    else:
        trainData = TextDataset(batch_size=args.batch_size, id_bos='[CLS]',
                                id_eos='[EOS]', id_unk='[UNK]',
                                max_sequence_length=args.max_sequence_length, vocab_size=0,
                                file_list=args.train_file_list, label_list=args.train_label_list, tokenizer=tokenizer)
        with open('./{}_trainData.pkl'.format(args.task), 'wb') as f:
            pickle.dump(trainData, f)

    add_log("Start train process.")

    ae_model.train()
    dis_model.train()
    ae_model.to(device)
    dis_model.to(device)

    '''
    Fixing or distilling BERT encoder
    '''
    if args.fix_first_6:
        print("Try fixing first 6 bertlayers")
        for layer in range(6):
            for param in ae_model.bert_encoder.encoder.layer[layer].parameters():
                param.requires_grad = False
    elif args.fix_last_6:
        print("Try fixing last 6 bertlayers")
        for layer in range(6, 12):
            for param in ae_model.bert_encoder.encoder.layer[layer].parameters():
                param.requires_grad = False
    
    if args.distill_2:
        print("Get result from layer 2")
        for layer in range(2, 12):
            for param in ae_model.bert_encoder.encoder.layer[layer].parameters():
                param.requires_grad = False

    ae_optimizer = NoamOpt(ae_model.d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=0.0001)

    #ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    ae_criterion = LabelSmoothing(size=ae_model.bert_encoder.config.vocab_size, padding_idx=0, smoothing=0.1).to(device)
    dis_criterion = nn.BCELoss(reduction='mean')

    history = {'train':[]}

    for epoch in range(args.epochs):
        print('-' * 94)
        epoch_start_time = time.time()
        total_rec_loss = 0
        total_dis_loss = 0

        train_data_loader = DataLoader(trainData,
                            batch_size=args.batch_size,
                            shuffle=True, 
                            collate_fn=trainData.collate_fn,
                            num_workers=4)
        num_batch = len(train_data_loader)
        trange = tqdm(enumerate(train_data_loader), total=num_batch, desc='Training', file=sys.stdout, position=0, leave=True)
        
        for it, data in trange:
            batch_sentences, tensor_labels, tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, tensor_tgt_mask, tensor_ntokens = data

            tensor_labels = tensor_labels.to(device)
            tensor_src = tensor_src.to(device)
            tensor_tgt = tensor_tgt.to(device) 
            tensor_tgt_y = tensor_tgt_y.to(device)
            tensor_src_mask = tensor_src_mask.to(device)
            tensor_tgt_mask = tensor_tgt_mask.to(device)

            # Forward pass
            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

            # Loss calculation
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data

            ae_optimizer.optimizer.zero_grad()
            loss_rec.backward()
            ae_optimizer.step()

            latent = latent.detach()
            next_latent = latent.to(device)

            # Classifier
            dis_lop = dis_model.forward(next_latent)
            loss_dis = dis_criterion(dis_lop, tensor_labels)

            dis_optimizer.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()

            total_rec_loss += loss_rec.item()
            total_dis_loss += loss_dis.item()

            trange.set_postfix(total_rec_loss=total_rec_loss / (it+1), total_dis_loss=total_dis_loss / (it+1))

            if it % 100 == 0:
                add_log(
                    '| epoch {:3d} | {:5d}/{:5d} batches | rec loss {:5.4f} | dis loss {:5.4f} |'.format(
                        epoch, it, num_batch, loss_rec, loss_dis))
                
                print(id2text_sentence(tensor_tgt_y[0], tokenizer, args.task))
                generator_text = ae_model.greedy_decode(latent,
                                                        max_len=args.max_sequence_length,
                                                        start_id=bos_id)
                print(id2text_sentence(generator_text[0], tokenizer, args.task))

                # Save model
                #torch.save(ae_model.state_dict(), args.current_save_path / 'ae_model_params.pkl')
                #torch.save(dis_model.state_dict(), args.current_save_path / 'dis_model_params.pkl')
        
        history['train'].append({'epoch': epoch, 'total_rec_loss': total_rec_loss / len(trange), 'total_dis_loss': total_dis_loss / len(trange)})

        add_log(
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))
        # Save model
        torch.save(ae_model.state_dict(), args.current_save_path / 'ae_model_params.pkl')
        torch.save(dis_model.state_dict(), args.current_save_path / 'dis_model_params.pkl')
    
    print("Save in ", args.current_save_path)
    return


def eval_iters(ae_model, dis_model):
    tokenizer = BertTokenizer.from_pretrained(args.PRETRAINED_MODEL_NAME, do_lower_case=True)
    tokenizer.add_tokens('[EOS]')
    bos_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    ae_model.bert_encoder.resize_token_embeddings(len(tokenizer))

    print("[CLS] ID: ", bos_id)
    
    if args.task == 'news_china_taiwan':
        eval_file_list = [
            args.data_path + 'test.0', args.data_path + 'test.1',
        ]
        eval_label_list = [
            [0],
            [1],
        ]
    
    
    if args.eval_positive:
        eval_file_list = eval_file_list[::-1]
        eval_label_list = eval_label_list[::-1]
    
    print("Load testData...")
    
    testData = TextDataset(batch_size=args.batch_size, id_bos='[CLS]',
                        id_eos='[EOS]', id_unk='[UNK]',
                        max_sequence_length=args.max_sequence_length, vocab_size=0,
                        file_list=eval_file_list, label_list=eval_label_list, tokenizer=tokenizer)
    
    dataset = testData
    eval_data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False, 
                            collate_fn=dataset.collate_fn,
                            num_workers=4)
    
    num_batch = len(eval_data_loader)
    trange = tqdm(enumerate(eval_data_loader), total=num_batch, desc='Training', file=sys.stdout, position=0, leave=True)
        
    gold_ans = [''] * num_batch

    add_log("Start eval process.")
    ae_model.to(device)
    dis_model.to(device)
    ae_model.eval()
    dis_model.eval()

    total_latent_lst = []

    for it, data in trange:
        batch_sentences, tensor_labels, tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, tensor_tgt_mask, tensor_ntokens = data

        tensor_labels = tensor_labels.to(device)
        tensor_src = tensor_src.to(device)
        tensor_tgt = tensor_tgt.to(device) 
        tensor_tgt_y = tensor_tgt_y.to(device)
        tensor_src_mask = tensor_src_mask.to(device)
        tensor_tgt_mask = tensor_tgt_mask.to(device)
        
        print("------------%d------------" % it)
        print(id2text_sentence(tensor_tgt_y[0], tokenizer, args.task))
        print("origin_labels", tensor_labels.cpu().detach().numpy()[0])

        latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
        generator_text = ae_model.greedy_decode(latent,
                                                max_len=args.max_sequence_length,
                                                start_id=bos_id)
        print(id2text_sentence(generator_text[0], tokenizer, args.task))

        # Define target label
        target = torch.FloatTensor([[1.0]]).to(device)
        if tensor_labels[0].item() > 0.5:
            target = torch.FloatTensor([[0.0]]).to(device)
        print("target_labels", target)

        modify_text, latent_lst = fgim_attack(dis_model, latent, target, ae_model, args.max_sequence_length, bos_id,
                                        id2text_sentence, None, gold_ans[it], tokenizer, device, task=args.task, save_latent=args.save_latent)
        if args.save_latent != -1:
            total_latent_lst.append(latent_lst)
        
        add_output(modify_text)

        if it >= args.save_latent_num:
            break

    print("Save log in ", args.output_file)

    if args.save_latent == -1:
        return

    folder = './latent_{}/'.format(args.task)
    if not os.path.exists(folder):
        os.mkdir(folder)

    if args.save_latent == 0: # full
        prefix = 'full'
    elif args.save_latent == 1: # first 6 layer
        prefix = 'first_6'
    elif args.save_latent == 2: # last 6 layer
        prefix = 'last_6'
    elif args.save_latent == 3: # get second layer
        prefix = 'distill_2'
    
    total_latent_lst = np.asarray(total_latent_lst)
    if args.eval_negative:
        save_label = 0
    else:
        save_label = 1
    with open(folder + '{}_{}.pkl'.format(prefix, save_label), 'wb') as f:
        pickle.dump(total_latent_lst, f)

    print("Save laten in ", folder + '{}_{}.pkl'.format(prefix, save_label))


if __name__ == '__main__':
    preparation()

    ae_model = Autoencoder(d_model=args.transformer_model_size, 
                           d_ff=args.transformer_ff_size, 
                           nlayers=args.num_layers_AE,
                           args=args,
                           device=device)
    dis_model = Attr_Classifier(latent_size=args.latent_size, output_size=args.label_size)
    
    if args.load_prev:
        try:
            ae_model.load_state_dict(torch.load(args.current_save_path / 'ae_model_params.pkl'))
            dis_model.load_state_dict(torch.load(args.current_save_path / 'dis_model_params.pkl'))
        except Exception:
            print("Cannot find model pkl! You need to train the model first")
            exit(1)
    
    if args.training:
        train_iters(ae_model, dis_model)

    if args.eval_negative or args.eval_positive:
        eval_iters(ae_model, dis_model)

    print("Done!")

