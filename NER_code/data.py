import torch
# Libs to create Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import re

class dataset(Dataset):
    def __init__(self, word2idx_dic, char2idx_dic, tag2idx_dic, path, data_format='iob1'):
        super(dataset, self).__init__()
        self.data_format = data_format
        self.word2idx_dic = word2idx_dic
        self.char2idx_dic = char2idx_dic
        self.tag2idx_dic  = tag2idx_dic
        # Initialize full dataset
        self.sentences = []
        self.tags = []
        self.words = []

        self.sentences, self.tags, self.words = self.load_data(path)

        self.char2idx()
        self.word2idx()
        self.tag2idx()

    def __getitem__(self, index):
        return self.sentences[index], self.tags[index], self.words[index]

    def __len__(self):
        return len(self.sentences)
        # if self.flag_labeled:
        #     return len(self.labeled_sentences)
        # else:
        #     return len(self.unlabeled_sentences)

    def load_data(self, path):
        f = open(path, 'r').readlines()

        temp_sentences = []
        temp_tags = []
        temp_words = []

        sentences = []
        tags = []
        words = []

        for line in f:
            if line == '\n' or not line:
                if temp_sentences:
                    temp_sentences = [word for word in itertools.chain(['<START>'], temp_sentences, ['<END>'])]
                    temp_tags = [tag for tag in itertools.chain(['O'], temp_tags, ['O'])]
                    temp_words = [[item for item in itertools.chain(['<START>'], [char for char in word], ['<END>'])] for word in temp_sentences]
                    sentences.append(temp_sentences)
                    words.append(temp_words)
                    if self.data_format == 'iob1':
                        tags.append(self.convert_IOB2_2_IOBES(self.convert_IOB1_2_IOB2(temp_tags)))
                    elif self.data_format == 'iob2':
                        tags.append(self.convert_IOB2_2_IOBES(temp_tags))
                    else:
                        tags.append(temp_tags)

                    temp_sentences = []
                    temp_tags = []
                    temp_words = []

            else:
                temp_sentences.append(line.split()[0])
                temp_tags.append(line.split()[3])
                
        if temp_sentences:
            temp_sentences = [word for word in itertools.chain(['<START>'], temp_sentences, ['<END>'])]
            temp_tags = [tag for tag in itertools.chain(['O'], temp_tags, ['O'])]
            temp_words = [[item for item in itertools.chain(['<START>'], [char for char in word], ['<END>'])] for word in temp_sentences]
            sentences.append(temp_sentences)
            words.append(temp_words)

            if self.data_format == 'iob1':
                tags.append(self.convert_IOB2_2_IOBES(self.convert_IOB1_2_IOB2(temp_tags)))
            elif self.data_format == 'iob2':
                tags.append(self.convert_IOB2_2_IOBES(temp_tags))
            else:
                tags.append(temp_tags)

        return sentences, tags, words
    
    def convert_IOB1_2_IOB2(self, sentence):
        prev_tag = 'O'
        for i in range(len(sentence)):
            if sentence[i][0] == 'I' and sentence[i] != prev_tag and prev_tag != 'B'+sentence[i][1:]:
                sentence[i] = 'B' + sentence[i][1:]
            prev_tag = sentence[i]
        return sentence

    def convert_IOB2_2_IOBES(self, sentence):
        for i in range(len(sentence)):
            if sentence[i][0] == 'I' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
                sentence[i] = 'E' + sentence[i][1:]
            elif sentence[i][0] == 'B' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
                sentence[i] = 'S' + sentence[i][1:]
        return sentence

    def clean_numbers(self, x):

        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def word2idx(self):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                word = self.clean_numbers(self.sentences[i][j])
                if word in self.word2idx_dic:
                    self.sentences[i][j] = self.word2idx_dic[word]
                elif word.lower() in self.word2idx_dic:
                    self.sentences[i][j] = self.word2idx_dic[word.lower()]
                else:
                    self.sentences[i][j] = self.word2idx_dic['<UNK>']

    def tag2idx(self):
        for i in range(len(self.tags)):
            for j in range(len(self.tags[i])):
                try:
                    self.tags[i][j] = self.tag2idx_dic[self.tags[i][j]]
                except KeyError:
                    print(f'Label no conjunto de teste não está presente no conjunto de treinamento: {self.tags[i][j][2:]}')
                    self.tags[i][j] = self.tag2idx_dic['O']
    def char2idx(self):
        self.words = [[[self.char2idx_dic['<UNK>'] if char not in self.char2idx_dic else self.char2idx_dic[char] for char in word] for word in sentence] for sentence in self.words]

    def sort_set(self, unordered_sentences, unordered_words, unordered_tags):
        # Change here
        ordered_idx = np.argsort([len(self.sentences[unordered_sentences[i]]) for i in range(len(unordered_sentences))])
        ordered_sentences = [unordered_sentences[i] for i in ordered_idx]
        ordered_words = [unordered_words[i] for i in ordered_idx]
        ordered_tags = [unordered_tags[i] for i in ordered_idx]
        return ordered_sentences, ordered_words, ordered_tags

    def get_word_count(self):
        # Change here
        if self.flag_labeled:
            sentences_idx = self.labeled_sentences
        else:
            sentences_idx = self.unlabeled_sentences
        word_count = 0
        for idx in sentences_idx:
            sentence = self.sentences[idx]
            # word_count += len(sentence) - 2
            for word in sentence:
                if word != self.word2idx_dic['<PAD>'] and word != self.word2idx_dic['<START>'] and word != self.word2idx_dic['<END>']:
                    word_count += 1
        return word_count
