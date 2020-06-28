from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import argparse
import pickle
import os

from dataset import TFDataset
from dataset2 import ThreeLabelDataset
from model import NewTransformer, Classifier
from trainer import Trainer
from data import preprocess
from data2 import preprocess_3label



def parse_args():
    parser = argparse.ArgumentParser()
    
    # training process related 
    parser.add_argument("--bsize", type=int, default=128)
    parser.add_argument("--epoch_max", type=int, default=200)
    parser.add_argument('--max_seq_length', type=int, default=60)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_from", type=str, default='')
    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--predict", action="store_true")
    
    # model structure related
    parser.add_argument("--latent_size", type=int, default=256)
    parser.add_argument("--num_ae_layers", type=int, default=2)
    parser.add_argument("--label_size", type=int, default=1)

    # specify each task and dataset
    parser.add_argument("--task", type=str, default='yelp')
    parser.add_argument("--lang", type=str, default='en')

    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # preprocess and get word dict
    if args.task == 'task3':
        data, label, vocab = preprocess_3label(task=args.task, lang=args.lang)  # data = [train_pos, train_neg, dev_pos, dev_neg, test_pos, test_neg]
        train_pos, train_neg, train_neutral, dev_pos, dev_neg, dev_neutral, _, _, _ = data

        # build datasets
        trainset = ThreeLabelDataset(train_pos, train_neg, train_neutral, vocab, args.max_seq_length)
        valset = ThreeLabelDataset(dev_pos, dev_neg, dev_neutral, vocab, args.max_seq_length)

    else:
        data, label, vocab = preprocess(task=args.task, lang=args.lang)  # data = [train_pos, train_neg, dev_pos, dev_neg, test_pos, test_neg]
        train_pos, train_neg, dev_pos, dev_neg, _, _ = data

        # build datasets
        trainset = TFDataset(train_pos, train_neg, vocab, args.max_seq_length)
        valset = TFDataset(dev_pos, dev_neg, vocab, args.max_seq_length)


    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ae_model = NewTransformer(vocab.size, device=args.device).to(args.device)
    cls_model = Classifier(latent_size=args.latent_size, output_size=args.label_size).to(args.device)

    args.vocab = vocab
    trainer = Trainer(trainset, valset, ae_model, cls_model, args)
    

    print('start training...')
    for epoch in range(args.epoch_max):
        print('epoch', epoch)
        if args.valid:
            trainer.run_epoch(epoch, training=False)
        else:
            trainer.run_epoch(epoch, training=True)
            trainer.run_epoch(epoch, training=False)