import nltk
import json
import os
from transformers import BertTokenizer


class Vocab():
    def __init__(self, tokenizer=None, tokenize_func=nltk.word_tokenize):
        self.word2id = {
            '<pad>': 0, 
            '<bos>': 1, 
            '<eos>': 2, 
            '<unk>': 3, 
        }
        self.id2word = {
            0: '<pad>', 
            1: '<bos>', 
            2: '<eos>', 
            3: '<unk>', 
        }
        self.builtin_words = set(self.word2id.keys())
        self.word_count = {}
        self.size = len(self.word2id)
        self.tokenizer = tokenizer
        self.tokenize_func = tokenize_func


    def load(self, path='./vocab.json'):
        with open(path, 'r', encoding='utf-8') as f:
            self.word2id, self.id2word = json.load(f)
            self.size = len(self.word2id)


    def dump(self, path='./vocab.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([self.word2id, self.id2word], f)


    def add_word(self, word):
        """add a word to the word pool. will NOT build the dict"""
        word = word.strip().lower()
        if word in self.builtin_words:
            return
        if word not in self.word_count:
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1
        

    def _add_word_to_dict(self, word):
        word = word.strip().lower()
        if word not in self.word2id:
            self.word2id[word] = self.size
            self.id2word[self.size] = word
            self.size += 1


    def build_dict(self, min_word_cnt_filter=5):
        word_count_pairs = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        print('total read:', len(word_count_pairs))
        for word, cnt in word_count_pairs:
            if cnt < min_word_cnt_filter:
                break
            self._add_word_to_dict(word)
        


    def add_word_from_sentence(self, sentence):
        for word in self.tokenize_func(sentence):
            self.add_word(word)


    def convert_word_to_id(self, word):
        if word not in self.word2id:
            word = '<unk>'
        return self.word2id[word]

    def convert_id_to_word(self, id):
        if type(id) != str:
            id = str(id)
        return self.id2word[id]

    def convert_sentence_to_ids(self, sentence):
        return [self.convert_word_to_id(word) for word in self.tokenize_func(sentence)]

    def convert_ids_to_sentence(self, ids, convert_all=False):
        if type(ids).__name__ == 'Tensor':
            assert len(ids.shape) == 1
            ids = ids.tolist()
        tokens = [self.convert_id_to_word(id) for id in ids]

        if '<eos>' in tokens and not convert_all:
            return ' '.join(tokens[:tokens.index('<eos>')])
        return ' '.join(tokens)

    


def preprocess(data_dir='./data/yelp/', task='yelp', lang='en'):
    
    
    if task == 'task1':
        lang = 'zh'
        data_dir='./data/task1/'
    elif task == 'task2':
        lang = 'zh'
        data_dir='./data/task2/'
    # elif task == 'task3':
    #     lang = 'zh'
    #     data_dir='./data/task3/'
    
    
    if lang == 'en':
        vocab = Vocab()
    elif lang == 'zh':
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        tokenize_func = tokenizer.tokenize
        vocab = Vocab(tokenizer=tokenizer, tokenize_func=tokenize_func) 
    
    
    data = []  # store converted sentence ids
    label = [0, 1, 0, 1, 0, 1]

    # load preprocessed data if available
    if os.path.exists(f'{data_dir}/vocab.json') and os.path.exists(f'{data_dir}/data.json'):
        print('loading preprocessed data...', end='')
        vocab.load(path=f'{data_dir}/vocab.json')

        with open(f'{data_dir}/data.json', 'r', encoding='utf-8') as f:
            data, label = json.load(f)
    
        if data != [] and vocab.size > 4:
            print('done.')
            print('vocab size:', vocab.size)
            return data, label, vocab
        print('failed.')
    
    
    print('preprocessing data...')
    # read data
    filename_list = [
        'train.0', 'train.1',
        'dev.0', 'dev.1',
        'test.0', 'test.1'
        ]

    # build word pool
    print('collecting words...')
    for i in range(3):
        with open(data_dir + filename_list[i*2], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                if line == '':
                    continue
                vocab.add_word_from_sentence(line)

        with open(data_dir + filename_list[i*2+1], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                if line == '':
                    continue
                vocab.add_word_from_sentence(line)

    # build word dict
    print('building word dict...')
    vocab.build_dict()
    print('vocab size:', vocab.size)
    vocab.dump(path=f'{data_dir}/vocab.json')

    # convert word to ids
    print('converting sentences...')
    vocab.load(path=f'{data_dir}/vocab.json')
    for i in range(3):
        with open(data_dir + filename_list[i*2], 'r', encoding='utf-8') as f:
            ids_list = []
            for line in f:
                line = line.strip().lower()
                if line == '':
                    continue
                ids_list.append(vocab.convert_sentence_to_ids(line))
                
            data.append(ids_list)

        with open(data_dir + filename_list[i*2+1], 'r', encoding='utf-8') as f:
            ids_list = []
            for line in f:
                line = line.strip().lower()
                if line == '':
                    continue
                ids_list.append(vocab.convert_sentence_to_ids(line))
                
            data.append(ids_list)

    with open(f'{data_dir}/data.json', 'w') as f:
        json.dump([data, label], f)

    return data, label, vocab


if __name__ == "__main__":
    preprocess()