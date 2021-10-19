#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ROIAlign
from slowfast.models.context_helper import ContextModule, FFN, ReaderUnit
import slowfast.utils.distributed as du

class ResNetPoolHead(nn.Module):
    """
    ResNe(X)t Pool head.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetPoolHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        output_dim = sum(dim_in) + sum(cfg.LSTC.SHORT_CONTEXT_VAL)

        self.num_readers = cfg.LSTC.NUM_READERS
        self.ff_feat = nn.Sequential(
            *[
                FFN(
                    mem_dim=sum(dim_in),
                    key_dim=sum(cfg.LSTC.SHORT_CONTEXT_VAL)
                ) for _ in range(self.num_readers)
            ]
        )
        self.ff_ctx = nn.Sequential(
            *[
                FFN(
                    mem_dim=sum(cfg.LSTC.SHORT_CONTEXT_VAL),
                    key_dim=sum(dim_in)
                ) for _ in range(self.num_readers)
            ]
        )

        # build necessary readers for feature banks
        self.relu = nn.ReLU()
        bank_dim = cfg.AVA.FEATURE_BANK_DIM

        self.bank_classifier = nn.Linear(bank_dim, num_classes)

        self.ff_feat_bank = nn.ModuleList(
            [
                ReaderUnit(
                    query_dim_in = sum(dim_in) if i == 0 else bank_dim,
                    dim_in = bank_dim,
                    window_size = cfg.AVA.SLIDING_WINDOW_SIZE,
                    embed_size = cfg.AVA.TEMPORAL_EMBED,
                    num_pairs = cfg.LSTC.NUM_PAIRS
                ) for i in range(self.num_readers)
            ]
        )

        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            temporal_pool_max = nn.MaxPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)
            self.add_module("s{}_tpool_max".format(pathway), temporal_pool_max)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            spatial_pool_avg = nn.AvgPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)
            self.add_module("s{}_spool_avg".format(pathway), spatial_pool_avg)

            context = ContextModule(
                dim_in = dim_in[pathway],
                dim_ctx_in = dim_in[pathway],
                dim_key = cfg.LSTC.SHORT_CONTEXT_KEY[pathway],
                dim_val = cfg.LSTC.SHORT_CONTEXT_VAL[pathway]
            )
            self.add_module('s{}_context'.format(pathway), context)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(
            output_dim,
            num_classes,
            bias=False
        )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, extract = False, FBs = None, BTs = None):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        ctx_out = []
        box_idx = bboxes[:, 0]

        # attns = []

        for pathway in range(self.num_pathways):

            # local feature
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out_t_avg = t_pool(inputs[pathway])
            t_pool = getattr(self, "s{}_tpool_max".format(pathway))
            out_t_max = t_pool(inputs[pathway])
            assert out_t_avg.shape[2] == 1
            out_t_avg = torch.squeeze(out_t_avg, 2)
            out_t_max = torch.squeeze(out_t_max, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out_t_avg = roi_align(out_t_avg, bboxes)
            out_t_max = roi_align(out_t_max, bboxes)
            s_pool = getattr(self, "s{}_spool".format(pathway))
            out_t_avg = s_pool(out_t_avg)
            s_pool = getattr(self, "s{}_spool_avg".format(pathway))
            out_t_max = s_pool(out_t_max)
            out = (out_t_max + out_t_avg) / 2
            out = out.view(out.shape[0], -1)
            pool_out.append(out)

            # context feature
            ctx = getattr(self, "s{}_context".format(pathway))
            path_context, attn = ctx(out, inputs[pathway], box_idx)
            ctx_out.append(path_context)
            # attns.append(attn)

        # B C H W.
        feat = torch.cat(pool_out, dim=1)
        ctx = torch.cat(ctx_out, dim=1)

        feat_out = feat.clone()
        context_out = ctx.clone()

        feat = self.ff_feat(feat)
        ctx = self.ff_ctx(ctx)

        bank_feat_out = feat
        for i, m in enumerate(self.ff_feat_bank):
            bank_feat_out = m(bank_feat_out, FBs, BTs, box_idx, residual=False if i == 0 else True)

        if hasattr(self, 'dropout'):
            bank_feat_out = self.dropout(bank_feat_out)
        Zl = self.bank_classifier(bank_feat_out)

        x_out = torch.cat([feat, ctx], dim=1)
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x_out)
        else:
            x = x_out

        x = x.view(x.shape[0], -1)
        # x = self.projection(x)
        Zs = self.classifier(x)

        sub_pred = self.act(Zs)
        bank_pred = self.act(Zl)
        x = Zl + Zs

        x = self.act(x)

        if extract:
            return feat_out, context_out
        else:
            return x, sub_pred, bank_pred

class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, extract, FBs = None, BTs = None):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)
        if extract:
            return x

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)

        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x
