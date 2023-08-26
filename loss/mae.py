import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn


class MAE:
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
            y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
        '''

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        if not y_true.shape[1] == self.num_tasks:
            raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(
                self.name, self.num_tasks, y_true.shape[1]))

        return y_true, y_pred

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)

        '''
            compute Average Precision (AP) averaged across tasks
        '''
        # mae = mean_absolute_error(y_true, y_pred)
        # return {'mae': mae}
        l1_loss = nn.L1Loss()
        loss = l1_loss(torch.tensor(y_true), torch.tensor(y_pred))
        return {'mae': loss.item()}

