import nltk
import json
import os


class Vocab():
    def __init__(self, tokenize_func=nltk.word_tokenize):
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
        assert len(self.word2id) == len(self.id2word)
        self.size = len(self.word2id)
        self.tokenize_func = tokenize_func


    def load(self, path='./vocab.json'):
        with open(path, 'r') as f:
            self.word2id, self.id2word = json.load(f)
            self.size = len(self.word2id)


    def dump(self, path='./vocab.json'):
        with open(path, 'w') as f:
            json.dump([self.word2id, self.id2word], f)


    def add_word(self, word):
        word = word.strip().lower()
        if word not in self.word2id:
            self.word2id[word] = self.size
            self.id2word[self.size] = word
            self.size += 1


    def add_word_from_sentence(self, sentence):
        for word in self.tokenize_func(sentence):
            self.add_word(word)


    def convert_word_to_id(self, word):
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

    


def preprocess(low_occur_word_filter=0):
    vocab = Vocab()
    data = []
    label = [0, 1, 0, 1, 0, 1]

    # load preprocessed data if available
    if os.path.exists('./vocab.json') and os.path.exists('./data.json'):
        print('loading preprocessed data...', end='')
        vocab.load()

        with open('./data.json', 'r') as f:
            data, label = json.load(f)
    
        if data != [] and vocab.size > 4:
            print('done.')
            return data, label, vocab
        print('failed.')
    
    
    print('preprocessing data...')
    # read data
    data_dir = './data/'
    filename_list = [
        'sentiment.train.0', 'sentiment.train.1',
        'sentiment.dev.0', 'sentiment.dev.1',
        'sentiment.test.0', 'sentiment.test.1'
        ]

    for i in range(3):
        with open(data_dir + filename_list[i*2], 'r') as f:
            ids_list = []
            for line in f:
                line = line.strip().lower()
                vocab.add_word_from_sentence(line)
                ids_list.append(vocab.convert_sentence_to_ids(line))
                
            data.append(ids_list)

        with open(data_dir + filename_list[i*2+1], 'r') as f:
            ids_list = []
            for line in f:
                line = line.strip().lower()
                vocab.add_word_from_sentence(line)
                ids_list.append(vocab.convert_sentence_to_ids(line))
                
            data.append(ids_list)

    # discard words with low occurrence or not. 
    if low_occur_word_filter:
        pass

    print('vocab size:', vocab.size)
    vocab.dump()
    with open('./data.json', 'w') as f:
        json.dump([data, label], f)

    return data, label, vocab


if __name__ == "__main__":
    preprocess()