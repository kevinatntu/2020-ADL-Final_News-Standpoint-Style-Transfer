import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
random.seed(24)


class TFDataset(Dataset):
    def __init__(self, pos_seq_ids, neg_seq_ids, vocab, max_seq_length):
        self.seq_label_pairs = [(ids, 1) for ids in pos_seq_ids] + [(ids, 0) for ids in neg_seq_ids]
        random.shuffle(self.seq_label_pairs)
        self.max_seq_length = max_seq_length
        
        self.voacb = vocab
        self.bos_id = vocab.word2id['<bos>']
        self.eos_id = vocab.word2id['<eos>']
        self.pad_id = vocab.word2id['<pad>']
        self.unk_id = vocab.word2id['<unk>']
    

    def __len__(self):
        return len(self.seq_label_pairs)

    def __getitem__(self, index):
        return self.seq_label_pairs[index]
        

    def collate_fn(self, data):
        batch_size = len(data)
        seq_ids_list, labels = zip(*data)  # unzip the (seq_ids, label) pairs

        batch_seq_length = min(self.max_seq_length, max([len(seq_ids) for seq_ids in seq_ids_list])+1)

        src = torch.zeros((batch_size, batch_seq_length-1), dtype=torch.int64)
        tgt = torch.zeros((batch_size, batch_seq_length), dtype=torch.int64)
        tgt_truth = torch.zeros((batch_size, batch_seq_length), dtype=torch.int64)  # for criteria

        for i in range(batch_size):
            tgt[i][0] = self.bos_id
            for j, id in enumerate(seq_ids_list[i]):
                if j >= batch_seq_length-1:
                    break
                if id >= self.voacb.size:
                    id = self.unk_id
                src[i][j] = id
                tgt[i][j+1] = id
                tgt_truth[i][j] = id
                tgt_truth[i][j+1] = self.eos_id

        src_key_padding_mask = (src == self.pad_id)  # (bsize, seq_len)
        tgt_key_padding_mask = (tgt == self.pad_id)  # (bsize, seq_len)
        # print(src_key_padding_mask[0, :], src_key_padding_mask.dtype)
        # print(tgt_key_padding_mask.shape, tgt_key_padding_mask.dtype)
        # exit()
        # tgt_mask_pad = (tgt != 0).unsqueeze(-2).repeat((1, batch_seq_length, 1))  # (bsize, seq_len, seq_len)
        # tgt_mask_seq = torch.tensor(
        #     [[[1,] * n + [0,] * (batch_seq_length-n) for n in range(1, batch_seq_length+1)], ] * batch_size
        # )  # (bsize, seq_len, seq_len)
        # tgt_mask = (tgt_mask_pad==1) & (tgt_mask_seq==1)  # (bsize, seq_len, seq_len)
        # src_mask, tgt_mask = None, None

        label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (bsize)

        num_tokens = (tgt_truth != 0).data.sum().float()

        # src       = [*seq]
        # tgt       = ['bos'] + [*seq]
        # tgt_truth = [*seq] + ['eos']
        return (src, tgt, tgt_truth, src_key_padding_mask, tgt_key_padding_mask, label_tensor, num_tokens)


if __name__ == "__main__":
    pass