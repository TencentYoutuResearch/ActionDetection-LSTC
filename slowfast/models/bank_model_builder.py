#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

from slowfast.utils.logging import get_logger
from .build import MODEL_REGISTRY


logger = get_logger(__name__)


@MODEL_REGISTRY.register()
class BankContext(nn.Module):

    act = {
        'sigmoid': nn.Sigmoid,
        'softmax': nn.Softmax
    }

    def __init__(self, cfg):
        super(BankContext, self).__init__()
        self.cfg = cfg
        self.window_size = cfg.AVA.SLIDING_WINDOW_SIZE * 2 + 1
        self.feature_size = sum(cfg.SLOWFAST.OUTPUT_CHANNEL)
        self.classifier = nn.Linear(self.feature_size, cfg.MODEL.NUM_CLASSES)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.act_func = self.act[cfg.MODEL.HEAD_ACT]()

        self._build_aggregators()

    def _build_aggregators(self, ratio = 2):

        inter_size = self.feature_size // ratio
        self.norm1 = nn.LayerNorm(self.feature_size)
        self.ffn1 = nn.Sequential(
            nn.Linear(self.feature_size, inter_size),
            nn.ReLU(),
            nn.Linear(inter_size, self.feature_size)
        )

        self.norm2 = nn.LayerNorm(self.feature_size)
        self.ffn2 = nn.Sequential(
            nn.Linear(self.feature_size, inter_size),
            nn.ReLU(),
            nn.Linear(inter_size, self.feature_size)
        )

    def forward(self, FBs):
        """
        aggregate context information from banks
        Args:
            FBs: list[torch.Tensor]

        Returns:
            torch.Tensor
        """
        num_batch = len(FBs)
        output = []
        for b in range(num_batch):
            feature_bank = FBs[b]
            clip_feat = [torch.mean(val, dim=0)
                         for k, val in enumerate(feature_bank)
                         if val is not None and k != self.window_size // 2]

            clip_feat = torch.stack(clip_feat, dim=0)
            clip_feat = self.ffn1(self.norm1(clip_feat))

            feat = torch.mean(clip_feat, dim=0).unsqueeze(0)
            output.append(self.ffn2(self.norm2(feat)))

        x = torch.cat(output, dim=0)
        x = self.dropout(x)

        return self.act_func(self.classifier(x))
