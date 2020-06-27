import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F


class LabelSmoothedKLDivLoss(nn.KLDivLoss):
    """
        KLDivLoss with label smoothing implemented. 
    """
    def __init__(self, vocab_size, ignore_id, smoothing_level=0.1, reduction='mean', log_target=False):
        super(LabelSmoothedKLDivLoss, self).__init__(size_average=False)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.confidence = 1.0 - smoothing_level
        self.smoothing_level = smoothing_level

    def forward(self, x, target):
        """
        shape: 
            x      = (bsize*seq_len, vocab)
            target = (bsize*seq_len)
        """
        assert x.shape[1] == self.vocab_size
        
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing_level / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        true_dist[:, self.ignore_id] = 0
        mask = torch.nonzero(target.data == self.ignore_id)
        if mask.shape[0] > 0:
            true_dist[mask, :] = 0.0
            # true_dist[mask, self.ignore_id] = 1.0
        
        loss = super(LabelSmoothedKLDivLoss, self).forward(x, true_dist.detach())
        return loss



def run_fast_grad_iter(ae_model, cls_model, latent, label, epsilon):
    data = latent.detach()  # (bsize, seq_length, d_model)
    data.requires_grad = True
    
    # get the classification loss
    output = cls_model(data)
    cls_criterion = nn.BCELoss(size_average=True)
    loss = cls_criterion(output, label)
    
    # calculate the gradient
    cls_model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    
    # update the latent by the gradient
    data = data - epsilon * data_grad

    with torch.no_grad():
        label_pred = cls_model(data)

    return data, label_pred



class CustomEmbeddings(nn.Module):
    """
    nn.Embeddings that will times a constant while output
    """
    def __init__(self, d_model, vocab):
        super(CustomEmbeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.sqrt_d = math.sqrt(d_model)

    def forward(self, x):
        return self.emb(x) * self.sqrt_d



class PositionalEncoding(nn.Module):
    """
    the positional encoding function.
    reference: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # w_k = frac{1}{ 10000 ^ {2k/d} }  , for k in 0:(d_model/2)
        k = torch.arange(0, d_model, 2)
        w_k = torch.exp( k / d_model * -1 * math.log(10000.0) )
        
        pe[:, 0::2] = torch.sin(position * w_k)
        pe[:, 1::2] = torch.cos(position * w_k)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.unsqueeze(0)[:, :x.size(1), :]
        return self.dropout(x)

        

class NewTransformer(nn.Transformer):
    def __init__(self, vocab, d_model=256, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=1024, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None, device=torch.device('cpu')):

        super(NewTransformer, self).__init__(d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout,
                 activation, custom_encoder, custom_decoder)

        self.device = device
        
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(d_model, vocab)
        self.shared_emb = CustomEmbeddings(d_model, vocab)

        self.src_emb = nn.Sequential(self.shared_emb, PositionalEncoding(d_model, dropout))
        self.tgt_emb = nn.Sequential(self.shared_emb, PositionalEncoding(d_model, dropout))


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        src = self.src_emb(src)  # (bsize, seq_len, d_modle)
        tgt = self.tgt_emb(tgt)  # (bsize, seq_len, d_modle)

        src = src.permute(1, 0, 2)  # (seq_len, bsize, d_modle)
        tgt = tgt.permute(1, 0, 2)  # (seq_len, bsize, d_modle)

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0]).to(self.device)

        latent = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # print(latent.shape)
        latent = self.sigmoid(latent)                   # (seq_len, batch_size, d_model)
        latent = torch.sum(latent, dim=0).unsqueeze(0)  # (1, batch_size, d_model)
        
        output = self.decoder(tgt, latent, 
                                tgt_mask=tgt_mask, memory_mask=memory_mask,
                                # tgt_key_padding_mask=tgt_key_padding_mask,
                                tgt_key_padding_mask=None,
                                memory_key_padding_mask=memory_key_padding_mask)
    
        prob = F.log_softmax(self.proj(output), dim=-1)

        latent = latent.transpose(0, 1).squeeze(1)  # (bsize, d_model)
        prob = prob.transpose(0, 1)  # (bsize, seq_len, vocab)

        return latent, prob


    def decode(self, latent, max_len, bos_id, eos_id, pad_id):
        # latent (1, bsize=1, d_model)

        with torch.no_grad():
            batch_size = latent.size(0)
            decoded_ids = torch.full((1, batch_size), bos_id, dtype=torch.int64).to(self.device)  # '<bos>' * (seq_len=1, bsize=1)
            
            for _ in range(max_len - 1):
                tgt = self.tgt_emb(decoded_ids)
                
                output = self.decoder(tgt, latent, tgt_mask=None, tgt_key_padding_mask=(decoded_ids.transpose(0, 1)==pad_id)) # (seq_len, bsize=1)
                last_word_prob = self.proj(output)[-1, :].unsqueeze(0)
                last_word_idx = torch.argmax(last_word_prob, dim=-1)
                
                decoded_ids = torch.cat((decoded_ids, last_word_idx), dim=0)  # (seq_len, bsize=1)
        
        return decoded_ids[1:, :].transpose(0, 1)



class Classifier(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(0.2, )
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out