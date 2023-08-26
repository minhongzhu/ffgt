import os
import pdb

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


class accuracy_SBM:
    def __init__(self, name, num_tasks):
        self.name = name
        self.num_tasks = num_tasks
    
    def _parse_and_check_input(self, input_dict):
        if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
        if not 'y_pred' in input_dict:
            raise RuntimeError('Missing key of y_pred')

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        '''
            y_true: torch tensor of shape (num_graphs, num_tasks) or (num_graphs,)
            y_pred: torch tensor of shape (num_graphs, num_tasks) or (num_graphs,)
        '''

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                y_pred = torch.sigmoid(y_pred)
            else:
                y_pred = F.log_softmax(y_pred, dim=-1)
            y_pred = _get_pred_int(y_pred)
            y_pred = y_pred.detach().cpu()

        ## check type
        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if y_true.dim() == 2 and y_true.shape[1] == 1:
            raise RuntimeError('y_true and y_pred mush to 1-dim arrray, {}-dim array given'.format(y_true.dim()))

        return y_true, y_pred

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)

        '''
            compute acc_SBM
        '''
        acc = acc_SBM(y_true, y_pred)
        return {'acc_SBM': acc}


def acc_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc

def _get_pred_int(pred_score, thresh=0.5):
    '''
        generate classification labels from probability distribution
    '''
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        # binary classification
        return (pred_score > thresh).long()
    else:
        # multi classification
        return pred_score.max(dim=1)[1]