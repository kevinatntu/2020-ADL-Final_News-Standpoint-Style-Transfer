import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import random
from tqdm import tqdm
import os
import json

from model import LabelSmoothedKLDivLoss, run_fast_grad_iter


class CustomAdam(torch.optim.Adam):
    def __init__(self, model_size, warmup, param, lr, betas, eps):
        super(CustomAdam, self).__init__(param, lr=lr, betas=betas, eps=eps)
        self._step = 0
        self._rate = 0
        self.warmup = warmup
        self.model_size = model_size

    def step(self):
        "Update parameters and rate"
        self._step += 1

        self._rate = self.model_size ** (-0.5) * min(self._step ** (-0.5), self._step * self.warmup ** (-1.5))
        for p in self.param_groups:
            p['lr'] = self._rate
        
        super(CustomAdam, self).step()



class Trainer():
    def __init__(self, trainset, valset, ae_model, cls_model, args):
        self.trainset = trainset
        self.valset = valset
        self.device = args.device
        self.args = args
        self.vocab = args.vocab

        self.ae_model = ae_model
        self.cls_model = cls_model
        
        self.ae_criteria = LabelSmoothedKLDivLoss(self.vocab.size, self.vocab.word2id['<pad>'], smoothing_level=0.1)
        self.cls_criteria = nn.BCELoss(size_average=True)
       
        self.ae_optimizer =  CustomAdam(512, 2000, ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.cls_optimizer = torch.optim.Adam(self.cls_model.parameters(), lr=1e-4)



    def run_epoch(self, epoch, training=True):
        """
            Train or validate with the corresponding dataset.
        """
        self.ae_model.train(training)
        self.cls_model.train(training)
        
        if training:
            description = f'train {epoch}'
            dataset = self.trainset
            shuffle = True
        else:
            description = f'valid {epoch}'
            dataset = self.valset
            shuffle = False

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.args.bsize,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=self.args.num_workers)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description, ascii=True)

        for batch_id, (src, tgt, tgt_truth, src_key_padding_mask, tgt_key_padding_mask, label, num_tokens) in trange:
            src = src.to(self.args.device)
            tgt = tgt.to(self.args.device)
            tgt_truth = tgt_truth.to(self.args.device)
            src_key_padding_mask = src_key_padding_mask.to(self.args.device)
            tgt_key_padding_mask = tgt_key_padding_mask.to(self.args.device)
            label = label.to(self.args.device)
            
            # +--------------------+
            # | update autoencoder |
            # +--------------------+
            # latent: (bsize, d_model)
            # prob  : (bsize, seq_len, vocab)
            latent, prob = self.ae_model(src, tgt, src_mask=None, tgt_mask=None, 
                                            src_key_padding_mask=src_key_padding_mask, 
                                            tgt_key_padding_mask=tgt_key_padding_mask)
            ae_loss = self.ae_criteria(prob.contiguous().view(-1, self.vocab.size),
                                        tgt_truth.contiguous().view(-1)) / num_tokens.item()

            if training:
                self.ae_optimizer.zero_grad()
                ae_loss.backward()
                self.ae_optimizer.step()

        
            # +-------------------+
            # | update classifier |
            # +-------------------+
            label_predict = self.cls_model(latent.clone().detach())
            cls_loss = self.cls_criteria(label_predict, label)

            if training:
                self.cls_optimizer.zero_grad()
                cls_loss.backward()
                self.cls_optimizer.step()


            trange.set_postfix(ae_loss=ae_loss.item(), cls_loss=cls_loss.item())
            

            if batch_id % 200 == 0:
                print('\n')
                print(f'label: {round(label[0].item(), 2)} | cls_pred: {round(label_predict[0].item(), 2)}')
                print('original: \n' + self.vocab.convert_ids_to_sentence(tgt_truth[0, :]))
                # latent: (bsize, d_model) -> (1, bsize=1, d_model)
                predicted_ids = self.ae_model.decode(latent.unsqueeze(0)[:, 0, :].unsqueeze(0),
                                                            max_len=self.args.max_seq_length,
                                                            bos_id=dataset.bos_id, 
                                                            eos_id=dataset.eos_id, 
                                                            pad_id=dataset.pad_id)
                print('ae-decoded: \n' + self.vocab.convert_ids_to_sentence(predicted_ids[0, :], convert_all=False))

                if not training:
                    origin_latent = latent
                    label = (label < 0.5).type(torch.float32)
                    for epsilon in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
                        print(f'----initial epsilon = {epsilon}----')
                        latent = origin_latent
                        
                        for fg_iter in range(5):
                            latent, label_pred = run_fast_grad_iter(self.ae_model, self.cls_model, latent, label, epsilon)
                            epsilon = epsilon * 0.9
                            predicted_ids = self.ae_model.decode(latent.unsqueeze(0)[:, 0, :].unsqueeze(0),
                                                                        max_len=self.args.max_seq_length,
                                                                        bos_id=dataset.bos_id, 
                                                                        eos_id=dataset.eos_id, 
                                                                        pad_id=dataset.pad_id)
                            print(f'iter: {fg_iter} | eps: {epsilon} | cls_pred: {round(label_pred[0].item(), 2)}\n' + self.vocab.convert_ids_to_sentence(predicted_ids[0, :], convert_all=False))

                    print('---------------------')

                print('\n')

        self.save_model(epoch)



    def save_model(self, epoch, path='./model/0'):
        if not os.path.exists(path):
            os.makedirs(path)
        print('Saving models...')
        torch.save({
            'ae_model': self.ae_model.state_dict(),
            'cls_model': self.cls_model.state_dict(),
        }, path + 'model.ckpt')



    def load_model(self, path):
        print('Loading models...')
        model = torch.load(path)
        self.ae_model.load_state_dict(model['ae_model'])
        self.cls_model.load_state_dict(model['cls_model'])