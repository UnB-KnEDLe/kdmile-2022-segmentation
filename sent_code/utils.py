import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import unicodedata
import numpy as np
from bisect import bisect
from math import sqrt

def create_word2idx_dict(emb, train_path):
    # dic = {}
    # for word in emb.index2word:
    #   dic[word] = emb.vocab[word].index
    # return dic
    return emb.key_to_index

def create_char2idx_dict(train_path):
    f = open(train_path, 'r').readlines()
    dic = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    for line in f:
        if line == '\n':
            continue
        word = line.split()[0]
        for char in word:
            if char not in dic:
                dic[char] = len(dic)
    return dic

def create_tag2idx_dict(train_path):
    f = open(train_path, 'r').readlines()
    iob2_dic = {'<PAD>': 0, 'O': 1}
    for line in f:
        if line != '\n':
            tag = line.split()[3]
            #if tag != 'O' and 'I' + tag[1:] not in dic:
            #    dic[tag] = len(dic)
            if tag != 'O' and tag not in iob2_dic:
                if 'B'+tag[1:] not in iob2_dic:
                    iob2_dic['B'+tag[1:]] = len(iob2_dic)
                if 'I'+tag[1:] not in iob2_dic:
                    iob2_dic['I'+tag[1:]] = len(iob2_dic)
                if 'S'+tag[1:] not in iob2_dic:
                    iob2_dic['S'+tag[1:]] = len(iob2_dic)
                if 'E'+tag[1:] not in iob2_dic:
                    iob2_dic['E'+tag[1:]] = len(iob2_dic)
                if tag not in iob2_dic:
                    iob2_dic[tag] = len(iob2_dic)

    return iob2_dic

class new_custom_collate_fn():
    def __init__(self, pad_idx, unk_idx):
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
    def __call__(self, batch):
        words = [torch.LongTensor(batch[i][0]) for i in range(len(batch))]
        tags  = [torch.LongTensor(batch[i][1]) for i in range(len(batch))]
        chars = [batch[i][2].copy() for i in range(len(batch))]

        # Pad word/tag level
        words = pad_sequence(words, batch_first = True, padding_value=self.pad_idx)
        tags  = pad_sequence(tags, batch_first = True, padding_value = 0)

        # Pad character level
        max_word_len = -1
        for sentence in chars:
            for word in sentence:
                max_word_len = max(max_word_len, len(word))
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = [0 if k >= len(chars[i][j]) else chars[i][j][k] for k in range(max_word_len)]
        for i in range(len(chars)):
            chars[i] = [[0 for _ in range(max_word_len)] if j>= len(chars[i]) else chars[i][j] for j in range(words.shape[1])]
        chars = torch.LongTensor(chars)

        mask = words != self.pad_idx

        return words, tags, chars, mask

def budget_limit(list_idx, budget, dataloader):
    budget_list = []
    for i in range(len(list_idx)):
        sent_len = len(dataloader.dataset.sentences[dataloader.dataset.unlabeled_sentences[list_idx[i]]]) - 2
        if sent_len <= budget:
            budget_list.append(list_idx[i])
            budget -= sent_len
    return budget_list, budget

def budget_limit2(list_idx, budget, dataloader):
    """
    Changes done to adapt to active_self_dataset
    """
    budget_list = []
    for i in range(len(list_idx)):
        sent_len = len(dataloader.dataset.sentences[dataloader.dataset.unlabeled_set[list_idx[i]]]) - 2
        if sent_len <= budget:
            budget_list.append(list_idx[i])
            budget -= sent_len
    return budget_list, budget

def augment_pretrained_embedding(embedding, train_path):
    """
    Augment pretrained embeddings with tokens from the training set
    """
    vocab = {}
    f = open(train_path)
    for line in f:
        try:
            word = line.split()[0]
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
        except:
            pass
    found = {}
    not_found = {}
    for word in vocab:
        if word not in embedding and word.lower() not in embedding:
            not_found[word] = vocab[word]
        else:
            found[word] = vocab[word]

    bias = sqrt(3/embedding.vector_size)
    for word in not_found:
        embedding.add(word, np.random.uniform(-bias, bias, embedding.vector_size))

class CustomDropout(torch.nn.Module):
    """
    Custom dropout layer based on inverted dropout to allow for frozen dropout masks
    """
    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        assert p > 0 and p < 1, 'Dropout probability out of range (0 < p < 1)'
        self.p = p
        self.drop_mask = None
        self.repeat_mask_flag = False

    def forward(self, x):
        if self.training:
            if not self.repeat_mask_flag:
                self.drop_mask = torch.distributions.binomial.Binomial(probs=1-self.p).sample(x.size()).to(x.device)
                self.drop_mask *= (1.0/(1-self.p))
            return x * self.drop_mask
        return x