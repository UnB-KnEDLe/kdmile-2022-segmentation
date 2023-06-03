# Basic packages
import argparse
import numpy as np
from gensim.models import KeyedVectors
import torch
from torch.utils.data import DataLoader
import itertools
import joblib
import re
import matplotlib.pyplot as plt
import operator
import math
from math import sqrt
from random import sample
# NER open packages
from seqeval.scheme import IOBES
from seqeval.metrics import f1_score
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
# my NER packages
from data import dataset
from utils import create_char2idx_dict, create_tag2idx_dict, create_word2idx_dict, new_custom_collate_fn, budget_limit, augment_pretrained_embedding
from metrics import preprocess_pred_targ, IOBES_tags
from CNN_biLSTM_CRF import cnn_bilstm_crf
from CNN_biLSTM_CRF_Attention import cnn_bilstm_crf_attention
from CNN_CNN_LSTM import CNN_CNN_LSTM

parser = argparse.ArgumentParser(description='Supervised training procedure for NER models!')
parser.add_argument('--save_training_path', dest='save_training', type=str, default=None, help='Path to save training history and hyperaparms used')
parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=50, help='Number of supervised training epochs')
# dataset parameters
parser.add_argument('--train_path', dest='train_path', action='store', type=str, default=None, help='Path to load training set from')
parser.add_argument('--val_path', dest='val_path', action='store', type=str, default=None, help='Path to load validation set from')
parser.add_argument('--test_path', dest='test_path', action='store', type=str, default=None, help='Path to load testing set from')
parser.add_argument('--dataset_format', dest='dataset_format', action='store', type=str, default='iob1', help='Format of the dataset (e.g. iob1, iob2, iobes)')
# Embedding parameters
parser.add_argument('--embedding_path', dest='embedding_path', type=str, default=None, help='Path to load pretrained embeddings from')
parser.add_argument('--augment_pretrained_embedding', dest='augment_pretrained_embedding', type=bool, default=False, help='Indicates whether to augment pretrained embeddings with vocab from training set')
# General model parameters
parser.add_argument('--model', dest='model', action='store', type=str, default='CNN-biLSTM-CRF', help='Neural NER model architecture')
parser.add_argument('--char_embedding_dim', dest='char_embedding_dim', action='store', type=int, default=30, help='Embedding dimension for each character')
parser.add_argument('--char_out_channels', dest='char_out_channels', action='store', type=int, default=50, help='# of channels to be used in 1-d convolutions to form character level word embeddings')
# CNN-CNN-LSTM specific parameters
parser.add_argument('--word_out_channels', dest='word_out_channels', action='store', type=int, default=800, help='# of channels to be used in 1-d convolutions to encode word-level features')
parser.add_argument('--word_conv_layers', dest='word_conv_layers', action='store', type=int, default=2, help='# of convolution blocks to be used to encode word-level features')
parser.add_argument('--decoder_layers', dest='decoder_layers', action='store', type=int, default=1, help='# of layers of the LSTM greedy decoder')
parser.add_argument('--decoder_hidden_size', dest='decoder_hidden_size', action='store', type=int, default=256, help='Size of the LSTM greedy decoder layer')
# CNN-biLSTM-CRF specific parameters
parser.add_argument('--lstm_hidden_size', dest='lstm_hidden_size', action='store', type=int, default=200, help='Size of the lstm for word-level feature encoder')
# Trainign hyperparameters
parser.add_argument('--lr', dest='lr', action='store', type=float, default=0.0015, help='Learning rate for NER mdoel training')
parser.add_argument('--grad_clip', dest='grad_clip', action='store', type=float, default=5.0, help='Value at which to clip the model gradient throughout training')
parser.add_argument('--momentum', dest='momentum', action='store', type=float, default=0.9, help='Momentum for the SGD optimization process')
parser.add_argument('--save_path', dest='save_path', action='store', type=str, default='', help='Path to save everything.')
parser.add_argument('--device', dest='device', action='store', type=str, default='cuda', help='Device (CPU or GPU) for inducing the model.')

# Training parameters
parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=8, help='Batch size for training')
parser_opt = parser.parse_args()

GRAPH_SAVE_PATH = parser_opt.save_path + 'loss_history'
MODEL_SAVE_PATH = parser_opt.save_path + 'model'
REPORT_SAVE_PATH = parser_opt.save_path + 'classification_report'
PREDICTED_SAVE_PATH = parser_opt.save_path + 'predicted_data'

print(f'\n****************************************************************************************************************')
print(f'****************************************************************************************************************')
print(f'****************************************************************************************************************')
print(f'Experiment: Supervised training')
print(f'Model: {parser_opt.model}')
print(f'batch size: {parser_opt.batch_size}')
print(f'Hardware available: {"cuda" if torch.cuda.is_available() else "cpu"}')
# ==============================================================================================
# ==============================================================================================
# =============================     Load embeddings     ========================================
# ==============================================================================================
# ==============================================================================================

#emb = KeyedVectors.load(parser_opt.embedding_path)
emb = KeyedVectors.load_word2vec_format(parser_opt.embedding_path)

if parser_opt.augment_pretrained_embedding:
    augment_pretrained_embedding(emb, parser_opt.train_path)

bias = sqrt(3/emb.vector_size)
if '<START>' not in emb:
    emb.add_vector('<START>', np.random.uniform(-bias, bias, emb.vector_size))
if '<END>' not in emb:
    emb.add_vector('<END>', np.random.uniform(-bias, bias, emb.vector_size))
if '<UNK>' not in emb:
    emb.add_vector('<UNK>', np.random.uniform(-bias, bias, emb.vector_size))
if '<PAD>' not in emb:
    emb.add_vector('<PAD>', np.zeros(emb.vector_size))

# ==============================================================================================
# ==============================================================================================q'
# ============================ Create train and test sets ======================================
# ==============================================================================================
# ==============================================================================================

collate_object = new_custom_collate_fn(pad_idx=emb.key_to_index['<PAD>'], unk_idx=emb.key_to_index['<UNK>'])

print('\nGenerating text2idx dictionaries (word, char, tag)')
word2idx = create_word2idx_dict(emb, parser_opt.train_path)
char2idx = create_char2idx_dict(train_path=parser_opt.train_path)
tag2idx  = create_tag2idx_dict(train_path=parser_opt.train_path)
print(len(char2idx), len(tag2idx))

print('\nCreating training dataset')
train_set = dataset(path=parser_opt.train_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)

print('\nCreating validation dataset')
val_set = dataset(path=parser_opt.val_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)

print('\nCreating test dataset')
test_set  = dataset(path=parser_opt.test_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)

size_val = len(val_set)
size_test = len(test_set)
size_train = len(train_set)

print(f'size_train: {size_train}, size_val: {size_val}, size_test: {size_test}')


test_dataloader = DataLoader(test_set, batch_size=parser_opt.batch_size, shuffle=False, collate_fn=collate_object)
validation_dataloader = DataLoader(val_set, batch_size=parser_opt.batch_size, shuffle=False, collate_fn=collate_object)

# ==============================================================================================
# ==============================================================================================
# ============================= Instantiate neural model =======================================
# ==============================================================================================
# ==============================================================================================

# Instantiating the model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(parser_opt.device)

if parser_opt.model == 'CNN-CNN-LSTM':
    model = CNN_CNN_LSTM(
        char_vocab_size=len(char2idx),
        char_embedding_dim=parser_opt.char_embedding_dim,
        char_out_channels=parser_opt.char_out_channels,
        pretrained_word_emb=emb,
        word2idx = word2idx,
        word_out_channels=parser_opt.word_out_channels,
        word_conv_layers = parser_opt.word_conv_layers,
        num_classes=len(tag2idx),
        decoder_layers = parser_opt.decoder_layers,
        decoder_hidden_size = parser_opt.decoder_hidden_size,
        device=device
    )
elif parser_opt.model == 'CNN-biLSTM-CRF':
    model = cnn_bilstm_crf(
        char_vocab_size=len(char2idx),
        char_embedding_dim=parser_opt.char_embedding_dim,
        char_out_channels=parser_opt.char_out_channels,
        pretrained_word_emb=emb,
        lstm_hidden_size=parser_opt.lstm_hidden_size,
        num_classes=len(tag2idx),
        device=device,
    )
elif parser_opt.model == 'CNN-biLSTM-CRF-Attention':
    model = cnn_bilstm_crf_attention(
        char_vocab_size=len(char2idx),
        char_embedding_dim=parser_opt.char_embedding_dim,
        char_out_channels=parser_opt.char_out_channels,
        pretrained_word_emb=emb,
        lstm_hidden_size=parser_opt.lstm_hidden_size,
        num_classes=len(tag2idx),
        device=device,
    )

lrate = parser_opt.lr
clipping_value = parser_opt.grad_clip
momentum = parser_opt.momentum

model = model.to(device)

# ==============================================================================================
# ==============================================================================================
# =============================== Define training hyperparams ==================================
# ==============================================================================================
# ==============================================================================================

# Defining supervised training hyperparameters
supervised_epochs = parser_opt.epochs
#optim = torch.optim.SGD(model.parameters(), lr=lrate, momentum=momentum)
optim = torch.optim.Adam(model.parameters(), lr=lrate)

# ==============================================================================================
# ==============================================================================================
# ============================= Supervised learning algorithm ==================================
# ==============================================================================================
# ==============================================================================================
print(f'\nInitiating supervised training\n\n')

dataloader = DataLoader(train_set, batch_size=parser_opt.batch_size, pin_memory=True, collate_fn = collate_object, shuffle=False)

labels = list(tag2idx.keys())
labels.remove('<PAD>')
labels.remove('O')

loss_history = []
f1_history = []

print(f'Tamanho do conjunto de treinamento: {len(train_set)}')

# Supervised training (traditional)    

baseline_epochs = int(400//sqrt(len(train_set)))

predicted_data = []

largest_sent = 0

val_loss_history = []
test_loss_history = []

INITIAL_PATIENCE = 30
patience = INITIAL_PATIENCE
smallest_loss = 1e1000

#torch.backends.cudnn.enabled = False

for epoch in range(supervised_epochs):
    print(f'Epoch: {epoch}')

    # Supervised training for one epoch
    model.train()
    total_loss = 0
    
    for sent, tag, word, mask in dataloader:
        tag = torch.t(tag)[1]
        
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        mask = mask.to(device)

        optim.zero_grad()

        loss = model(sent, word, tag, mask)

        
        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optim.step()
        
    print(f'Perda (treinamento): {total_loss}')
    loss_history.append(total_loss.item())
    
    # Verify performance on validation set
    model.eval()
    val_loss = 0
    for sent, tag, word, mask in validation_dataloader:
        tag = torch.t(tag)[1]
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        mask = mask.to(device)
        optim.zero_grad()
        loss = model(sent, word, tag, mask)
        val_loss += loss.item()
        #loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        #optim.step()
        
    val_loss *= size_train/size_val
    print(f'Perda (validação): {val_loss}')
    val_loss_history.append(val_loss)
    
    # Verify performance on test set after supervised training epoch
    model.eval()
    with torch.no_grad():
        predictions, targets = preprocess_pred_targ(model, test_dataloader, device)
        
        predictions = IOBES_tags(predictions, tag2idx)
        targets = IOBES_tags(targets, tag2idx)

        if len(predicted_data) == 0:
            predicted_data.append(targets)
        predicted_data.append(predictions)
        
        weighted_f1 = flat_f1_score(targets, predictions, average='weighted', labels = labels)
        f1_history.append(weighted_f1)
        print(f'weighted f1-score: {weighted_f1}')
        
    if val_loss > smallest_loss:
        patience -= 1
        print(f'Patience: {patience}')
        if patience == 0 or val_loss > 2*smallest_loss:
            break
    else:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        smallest_loss = val_loss
        patience = INITIAL_PATIENCE
    print()

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

with torch.no_grad():
    predictions, targets = preprocess_pred_targ(model, test_dataloader, device)
    predictions = IOBES_tags(predictions, tag2idx)
    targets = IOBES_tags(targets, tag2idx)
    
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            if targets[i][j][0] == 'S':
                targets[i][j] = 'B' + targets[i][j][1:]
            elif targets[i][j][0] == 'E':
                targets[i][j] = 'I' + targets[i][j][1:]
                                              
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[i][j][0] == 'S':
                predictions[i][j] = 'B' + predictions[i][j][1:]
            elif predictions[i][j][0] == 'E':
                predictions[i][j] = 'I' + predictions[i][j][1:]
    
    remove_list = []
    for i in range(len(labels)):
        if labels[i][0] in ['S', 'E']:
            remove_list.append(labels[i])
    for i in remove_list:
        labels.remove(i)
    
    classification_report = flat_classification_report(targets, predictions, labels = labels)
    
from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot([i for i in range(1, len(loss_history) + 1)], loss_history, color = color, label = 'Perda (treinamento)')
ax1.plot([i for i in range(1, len(val_loss_history) + 1)], val_loss_history, color = 'tab:green', label = 'Perda (validação)')
plt.legend(loc = 'lower left')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('f1', color=color)  # we already handled the x-label with ax1
ax2.plot([i for i in range(1, len(loss_history) + 1)], f1_history, color = color, label = 'F1-score (teste)')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.annotate(f'f1-score final: {f1_history[-1]:.3f}', (0,0), (+250, -30), xycoords='axes fraction', textcoords='offset points', va='top')
plt.legend(loc = 'upper left')

image_name = GRAPH_SAVE_PATH + '.png'
plt.savefig(image_name, dpi = 130)

with open(REPORT_SAVE_PATH + '.txt', 'w') as file:
    file.write(classification_report)
    
import pickle
with open(PREDICTED_SAVE_PATH + '.pkl', 'wb') as file:
    pickle.dump(predicted_data, file)
        
# ==============================================================================================
# ==============================================================================================
# ================================= Save training history ======================================
# ==============================================================================================
# ==============================================================================================

hyperparams = {'model': str(model), 'LR': lrate, 'momentum': momentum, 'clipping': clipping_value}
dic = {'f1_hist': f1_history, 'hyperparams': hyperparams}
if parser_opt.save_training:
    joblib.dump(dic, parser_opt.save_training)
#if parser_opt.save_mode:
#    torch.save(model.state_dict(), parser_opt.save_model)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

