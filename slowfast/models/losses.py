#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
import torch

class CustomizeCrossEntropy(nn.Module):

    def __init__(self, reduction="mean"):
        super(CustomizeCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, pred, labels):

        pred = -1 * torch.log(pred + 1e-5)
        loss = torch.sum(pred * labels, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            return torch.sum(loss)

class CrossScopeFocalLoss(nn.Module):

    def __init__(self, reduction="mean", gamma=1.0):
        super(CrossScopeFocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, pred1, pred2, labels):

        prob1 = pred1.clone()
        prob2 = pred2.clone()

        pred1 = -1 * torch.log(prob1 + 1e-5)
        pred2 = -1 * torch.log(prob2 + 1e-5)

        loss1 = torch.sum(((1 - prob2) ** self.gamma) * pred1 * labels, dim=-1)
        loss2 = torch.sum(((1 - prob1) ** self.gamma) * pred2 * labels, dim=-1)

        loss = loss1 + loss2

        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            return torch.sum(loss)

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

_LOSSES = {
    "cross_entropy": CustomizeCrossEntropy, # nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "scope_focal_loss": CrossScopeFocalLoss
}

