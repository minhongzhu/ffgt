# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False

class LinearDecayLR(_LRScheduler):

    def __init__(self, optimizer, lr, end_lr, tot_updates, power, last_epoch = -1, verbose=False):
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        self.tot_updates = tot_updates
        super(LinearDecayLR, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - self._step_count / self.tot_updates
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr
        
        return [lr for group in self.optimizer.param_groups]

    def _get_closed_for_lr(self):
        assert False


def view_model_param(model, MODEL_NAME):
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        if param.data.size()[0] != 9*512 + 1:
            total_param += np.prod(list(param.data.size()))
        else:
            total_param += np.prod([200, param.data.size()[1]])
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param