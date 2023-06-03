import numpy as np
import torch

def preprocess_pred_targ(model, dataloader, device):
    """
    Transform predictions and targets from torch.Tensors to lists
    """
    full_pred = []
    full_targ = []
    with torch.no_grad():
        for sent, tag, word, mask in dataloader:
            tag = torch.t(tag)[1]
            sent = sent.to(device)
            tag = tag.to(device)
            word = word.to(device)
            mask = mask.to(device)
            pred, _ = model.decode(sent, word, mask)

            
            for i in range(len(tag)):
                full_pred.append([pred[0][i].tolist()])
                full_targ.append([tag[i].tolist()])
    
    return full_pred, full_targ

def IOBES_tags(predictions, tag2idx):
    """
    Transform tags from indices to class name strings
    """
    idx2tag = {}
    for tag in tag2idx:
        idx2tag[tag2idx[tag]] = tag
    
    IOBES_tags = predictions.copy()
    for i in range(len(IOBES_tags)):
        for j in range(len(IOBES_tags[i])):
            IOBES_tags[i][j] = idx2tag[IOBES_tags[i][j]]
    return IOBES_tags

