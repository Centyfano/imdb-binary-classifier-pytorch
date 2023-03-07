import torch
from torch.utils.data import Dataset
from torchaudio import datasets
import os
import gzip
import tarfile
import pandas as pd
from collections import Counter

class ImdbDataset(Dataset):
    def __init__(self, dataset_zip_path, val_split=None):
        super().__init__()
        self.dataset_zip_path = os.path.abspath(dataset_zip_path)
        self.base = os.path.dirname(self.dataset_zip_path)
        self.dataset_dir = os.path.join(self.base, 'aclImdb')
        self.val_split = val_split
        self.train_sentences, self.train_labels = self.create_dataset('train')
        self.test_sentences, self.test_labels = self.create_dataset('test')

    def extract(self):
        if not os.path.exists(self.dataset_dir):
            with gzip.open(self.dataset_zip_path, 'rb') as f_in:
                with tarfile.open(fileobj=f_in, mode='r:*') as tar:
                    tar.extractall(path=self.base)

    def append(self, path, review, label):
        sentences = []
        labels = []
        dir_path = os.path.join(path, review)
        for _f in os.listdir(dir_path):
            with open(os.path.join(dir_path, _f), 'r', encoding='utf-8') as f:
                sentences.append(f.read())
                labels.append(label)
        return sentences, labels

    def create_dataset(self, corpus):
        self.extract()
        path = os.path.join(self.dataset_dir, corpus)
        pos_sentences, pos_labels = self.append(path, 'pos', 1)
        neg_sentences, neg_labels = self.append(path, 'neg', 0)
        sentences = pos_sentences + neg_sentences
        labels = pos_labels + neg_labels
        return sentences, labels
        
class Tokenizer():
    def __init__(self, num_words, oov_token=None, pad_token=None):
        self.num_words = num_words
        self.oov_token = oov_token
        self.pad_token = pad_token

    def fit_on_texts(self, texts):
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        most_common = word_counts.most_common(self.num_words)
        first_tokens = [self.oov_token,self.pad_token]
        first_tokens_nn = list(filter(lambda s:s is not None, first_tokens))
        lens = len(first_tokens_nn)
        self.word_index = {word: (idx+1) + lens for idx, (word, count) in enumerate(most_common)}

        if lens > 0:
            list_len = [i+1 for i in range(len(first_tokens_nn))]

            if self.pad_token is not None:
                self.word_index[self.pad_token] = list_len[0]
            if self.oov_token is not None:
                self.word_index[self.oov_token] = list_len[-1]
            
        self.index_word = {index: word for word, index in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = text.split()
            sequence = [self.word_index.get(word, 1) for word in words]
            sequences.append(sequence)
        return sequences
    
    def pad_sequences(self, sequences, max_length):
        padded_seqs = []
        for seq in sequences:
            padded_seq = seq[:max_length] + 0 * (max_length - len(seq))
            padded_seqs.append(padded_seq)
        return padded_seqs



if __name__ == "__main__":
    imdb = ImdbDataset('data/aclImdb_v1.tar.gz')

    d = {'train sentence': imdb.train_sentences, 'train_label': imdb.train_labels}
    df = pd.DataFrame(data=d)
    print(df)
