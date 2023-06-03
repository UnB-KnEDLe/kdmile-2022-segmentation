import torch
from torch import nn
from math import sqrt
from crf import CRF
from torch.nn.utils.rnn import pad_sequence
from utils import CustomDropout

from time import process_time

from pynvml import *
def get_memory_available(gpu = None):
    nvmlInit()
    h2 = nvmlDeviceGetHandleByIndex(1)
    info2 = nvmlDeviceGetMemoryInfo(h2)
    factor = 1024*1024*1024
    return info2.free/factor

class char_cnn(nn.Module):
    """
    Character-level word embedding neural network as implemented in Ma and Hovy (https://arxiv.org/abs/1603.01354)
    """
    def __init__(self, embedding_size, embedding_dim, char_out_channels):
        super(char_cnn, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=char_out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)
        self.dropout = CustomDropout(p=0.5)
        self.init_weight()

    def init_weight(self):
        bias = sqrt(3/self.embedding.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bias, bias)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        shape = x.shape
        x = self.conv(x.reshape([shape[0]*shape[1], shape[2], shape[3]]).permute(0, 2, 1))
        # x = self.relu(x)
        # x = self.dropout(self.relu(x))
        x = torch.nn.functional.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
        
        return x.reshape([shape[0], shape[1], -1])

class bilstm_crf(nn.Module):
    def __init__(self, feature_size, num_classes, device, lstm_hidden_size=256):
        super(bilstm_crf, self).__init__()
        self.bilstm = torch.nn.LSTM(input_size=feature_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.word_to_sent = torch.nn.LSTM(input_size = lstm_hidden_size*2, hidden_size = lstm_hidden_size*2, num_layers = 1, batch_first = True)
        self.linear = torch.nn.Linear(in_features=lstm_hidden_size*2, out_features=num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.dropout = CustomDropout(p=0.5)
        self.weight_init()

    def weight_init(self):
        # Initialize linear layer
        bias = sqrt(6/(self.linear.weight.shape[0]+self.linear.weight.shape[1]))
        nn.init.uniform_(self.linear.weight, -bias, bias)
        nn.init.constant_(self.linear.bias, 0.0)
        # Initialize LSTM layer
        for name, params in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(params, 0.0)
                nn.init.constant_(params[self.bilstm.hidden_size:2*self.bilstm.hidden_size], 1.0)
            else:
                bias = sqrt(6/(params.shape[0]+params.shape[1]))
                nn.init.uniform_(params, -bias, bias)
        

    def forward(self, x, y, mask):
        x = self.dropout(x)
        x, (_, _) = self.bilstm(x)
        x = self.dropout(x)
        
        _, (x, _) = self.word_to_sent(x)
        
        x = self.linear(x)
        y = y.view(1, -1)
        
        #x = self.crf(x, y, mask=mask.T[0].view(1, -1))
        x = self.crf(x, y)
        return x

    def decode(self, x, mask):
        x, (_, _) = self.bilstm(x)
        _, (x, _) = self.word_to_sent(x)
        x = self.linear(x)
        pred, prob = self.crf.decode(x)
        return pred, prob
    
class cnn_bilstm_crf(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim, char_out_channels, pretrained_word_emb, num_classes, device, lstm_hidden_size):
        print('CNN-BiLSTM-CRF normal')
        super(cnn_bilstm_crf, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.char_encoder = char_cnn(embedding_size=char_vocab_size, embedding_dim=char_embedding_dim, char_out_channels=char_out_channels)
        self.word_embedder= nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_word_emb.vectors))
        self.decoder      = bilstm_crf(feature_size=char_out_channels+pretrained_word_emb.vector_size, num_classes=num_classes, device=device, lstm_hidden_size=lstm_hidden_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sent, word, tag, mask):
        char_emb = self.char_encoder(word)
        word_emb = self.word_embedder(sent)
        x = torch.cat((word_emb, char_emb), dim=2)
        x = self.decoder(x, tag, mask)
        return -x

    def decode(self, sent, word, mask, return_token_log_probabilities = False):
        """
        return_token_log_probabilities not implemented
        """
        char_emb = self.char_encoder(word)
        word_emb = self.word_embedder(sent)
        x = torch.cat((word_emb, char_emb), dim=2)
        x, prob = self.decoder.decode(x, mask=mask)
        x = [torch.LongTensor(aux) for aux in x]
        predictions = pad_sequence(x, batch_first = True, padding_value = 0)
        return predictions, prob
