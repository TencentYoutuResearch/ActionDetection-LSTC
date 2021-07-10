import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np

import slowfast.utils.logging as logging

from collections import defaultdict
from functools import reduce

logger = logging.get_logger(__name__)

class Pair(nn.Module):

    def __init__(
            self,
            dim_in,
            query_dim_in,
            ratio=2,
            dropout=0.2
    ):
        super(Pair, self).__init__()

        # part for construct context pair

        self.projectc1 = nn.Linear(query_dim_in, dim_in // ratio)
        self.projectc2 = nn.Linear(dim_in, dim_in // ratio)
        self.projectc3 = nn.Linear(dim_in, dim_in // ratio)

        self.projectd1 = nn.Linear(query_dim_in, dim_in // ratio)
        self.projectd2 = nn.Linear(dim_in, dim_in // ratio)
        self.projectd3 = nn.Linear(dim_in, dim_in // ratio)
        self.latent = dim_in // ratio

        self.cffn = nn.Sequential(
            nn.LayerNorm(dim_in // ratio),
            nn.PReLU(),
            nn.Linear(dim_in // ratio, dim_in),
            nn.Dropout(dropout)
        )

    def make_context_pair(self, rois_pair1, key_pair1, val_pair1, rois_pair2, key_pair2, val_pair2):

        ncontext = key_pair1.shape[0]

        p1 = torch.matmul(rois_pair1, key_pair1.permute(1, 0)) / math.sqrt(self.latent)
        p1 = torch.softmax(p1, dim=-1)
        f1 = torch.matmul(p1, val_pair1)
        p2 = torch.matmul(rois_pair2, key_pair2.permute(1, 0)) / math.sqrt(self.latent)
        p2 = torch.softmax(p2, dim=-1)
        f2 = torch.matmul(p2, val_pair2)

        cpair = f1 * f2
        cpair = self.cffn(cpair)

        return cpair

    def forward(self, feat, key_bank):

        # one-to-pair
        pair_kb1 = self.projectc2(key_bank)
        pair_fb1 = self.projectc3(key_bank)
        rois_pair_key1 = self.projectc1(feat)

        pair_kb2 = self.projectd2(key_bank)
        pair_fb2 = self.projectd3(key_bank)
        rois_pair_key2 = self.projectd1(feat)

        pair_ctx = self.make_context_pair(rois_pair_key1, pair_kb1, pair_fb1,
                                          rois_pair_key2, pair_kb2, pair_fb2)

        return pair_ctx


class ReaderUnit(nn.Module):

    def __init__(
            self,
            dim_in,
            query_dim_in,
            ratio = 2,
            dropout = 0.2,
            window_size = 0,
            embed_size = 0,
            num_pairs = 2
    ):
        super(ReaderUnit, self).__init__()
        self.codec = nn.Sequential(
            FFN(mem_dim=dim_in + embed_size, key_dim=dim_in + embed_size),
            FFN(mem_dim=dim_in + embed_size, key_dim=dim_in + embed_size)
        )
        self.project1 = nn.Linear(query_dim_in, dim_in // ratio)
        self.project2 = nn.Linear(dim_in + embed_size, dim_in // ratio)
        self.project3 = nn.Linear(dim_in + embed_size, dim_in // ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_in // ratio),
            nn.PReLU(),
            nn.Linear(dim_in // ratio, dim_in),
            nn.Dropout(dropout)
        )

        self.dim_in = dim_in
        self.latent = dim_in // ratio
        self.window_size = window_size + 1
        self.embed_size = embed_size

        # pair unit
        self.num_pairs = num_pairs
        self.aggregator = nn.Parameter(torch.ones(1, 1, self.num_pairs) / num_pairs)
        self.pairs = nn.ModuleList([
            Pair(dim_in, query_dim_in, ratio)
            for _ in range(num_pairs)
        ])

        if window_size > 0 and embed_size > 0:
            self.embed = nn.Embedding(self.window_size, self.embed_size)

    def forward(self, feat, key_banks, time_stamps, batch_idx, residual=True):

        num_batch = int(torch.max(batch_idx).item()) + 1
        output = feat.new_zeros(feat.shape[0], self.dim_in) if not residual else feat.clone()
        feat_key = self.project1(feat)

        for b in range(num_batch):
            clip_idx = torch.nonzero(batch_idx == b).squeeze(1)
            if clip_idx.shape[0] > 0 and key_banks[b] is not None:

                # one-to-pair
                pairs = torch.stack(
                    [m(feat[clip_idx], key_banks[b]) for m in self.pairs],
                    dim=2
                )
                pair_ctx = torch.sum(self.aggregator * pairs, dim=2)

                # one-to-one
                read_bank = key_banks[b].unsqueeze(0).expand(clip_idx.shape[0], key_banks[b].shape[0], -1)
                rois_keys = feat_key[clip_idx].unsqueeze(1)

                if self.window_size > 0 and self.embed_size > 0:
                    times = torch.Tensor(time_stamps[b]).to(feat.device)
                    nbanks = times.shape[0] - clip_idx.shape[0]
                    nfeat = clip_idx.shape[0]
                    bank_time, feat_time = times[:nbanks], times[-nfeat:]
                    temporal_dist = torch.abs(feat_time.unsqueeze(1) - bank_time.unsqueeze(0)).long()

                    temp_embed = self.embed(temporal_dist)
                    read_bank = torch.cat([read_bank, temp_embed], dim=-1)

                read_bank = self.codec(read_bank)

                kb = self.project2(read_bank)
                key_bank = kb.permute(0, 2, 1).contiguous()
                fb = self.project3(read_bank)

                coeff = torch.matmul(rois_keys, key_bank) / math.sqrt(self.latent)
                coeff = torch.softmax(coeff, dim=-1)

                feat_out = torch.matmul(coeff, fb).squeeze(1)
                feat_out = self.ffn(feat_out)

                feat_out += pair_ctx

                if residual:
                    output[clip_idx] = output[clip_idx] + feat_out
                else:
                    output[clip_idx] = feat_out

        return output

class ContextModule(nn.Module):

    def __init__(
            self,
            dim_in,
            dim_ctx_in,
            dim_key,
            dim_val
    ):
        super(ContextModule, self).__init__()
        self.project = nn.Linear(dim_in, dim_key)
        self.keys = nn.Conv3d(dim_ctx_in, dim_key, 1, 1, 0)
        self.vals = nn.Conv3d(dim_ctx_in, dim_val, 1, 1, 0)
        self.dim_in = dim_in
        self.dim_key = dim_key
        self.dim_val = dim_val

    def forward(self, rois, context, batch_idx):

        """
        Args:
            rois: [N_box x dim_in] torch.Tensor
            context: [N x dim_in x T x H x W] torch.Tensor
            batch_idx: [N_box] torch.Tensor

        Returns:
            [N_box x dim_val] torch.Tensor

        """
        num_batch, _, ctx_time, ctx_height, ctx_width = context.shape
        rois_keys = self.project(rois)
        context_key = self.keys(context).view(num_batch, self.dim_key, -1)
        context_val = self.vals(context).view(num_batch,
                        self.dim_val, -1).permute(0, 2, 1).contiguous()

        ctx = rois.new_zeros((rois.shape[0], self.dim_val))
        attn = []
        for i in range(num_batch):
            box_idx = torch.nonzero(batch_idx == i).squeeze(1)
            nbox = box_idx.shape[0]
            if nbox > 0:
                query = rois_keys[box_idx]
                coeff = torch.mm(query, context_key[i]) / math.sqrt(self.dim_key)
                coeff = torch.softmax(coeff, dim=-1)
                attn.append(coeff.view(nbox, ctx_time, ctx_height, ctx_width) * 255.0)
                batch_ctx = torch.mm(coeff, context_val[i])
                ctx[box_idx] = batch_ctx

        attn = torch.cat(attn, dim=0)

        return ctx, attn

class FFN(nn.Module):

    def __init__(
            self,
            mem_dim,
            key_dim,
            ratio = 4,
            dropout = 0.2,

    ):

        super(FFN, self).__init__()
        self.mem_dim = mem_dim
        self.key_dim = key_dim

        self.output_ffn = nn.Sequential(
            nn.Linear(mem_dim, mem_dim // ratio),
            nn.PReLU(),
            nn.Linear(mem_dim // ratio, mem_dim)
        )

        self.norm2 = nn.LayerNorm(mem_dim)

        if dropout > 0.0:
            self.drop_out = nn.Dropout(dropout)

    def forward(self, query):

        """
        Args:
            query: torch.Tensor [Nd x mem_dim] input query value
            key: torch.Tensor [Nd x key_dim] input key value

        Returns:

        """

        codec = query
        codec = self.output_ffn(codec)
        codec = self.norm2(codec)
        if hasattr(self, 'drop_out'):
            codec = self.drop_out(codec)

        return codec + query


class BasicTransformer(nn.Module):

    def __init__(
            self,
            dim_query_in,
            dim_keyval_in,
            dim_key,
            dim_inner,
            num_head = 1,
    ):
        super(BasicTransformer, self).__init__()
        assert dim_query_in % num_head == 0
        self.dim_key = dim_key
        self.num_head = num_head
        dim_val = dim_query_in // num_head
        self.dim_val = dim_val

        self.query_transform = nn.Linear(dim_query_in, dim_key)

        self.keys = nn.Linear(dim_keyval_in, dim_key)
        self.vals = nn.Linear(dim_keyval_in, dim_val)

        self.ffn = nn.Sequential(
            nn.Linear(dim_query_in, dim_inner),
            nn.ReLU(),
            nn.Linear(dim_inner, dim_query_in)
        )

        self.norm1 = nn.LayerNorm(dim_query_in)
        self.norm2 = nn.LayerNorm(dim_query_in)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, query, keyval):

        """

        Args:
            query: [Nq x dim_query_in] torch.Tensor
            keyval: [Nd x dim_keyval_in] torch.Tensor

        """
        heat = 0.0

        query_key = self.query_transform(query)
        key = self.keys(keyval)
        val = self.vals(keyval)

        attention = torch.softmax(
            torch.mm(query_key, key.permute(1, 0)) / math.sqrt(self.dim_key),
            dim=1)
        heat += torch.sum(attention.detach(), dim=0)

        out = torch.mm(attention, val)

        heat /= self.num_head
        out1 = query + self.dropout1(out)
        out1 = self.norm1(out1)

        out2 = out1 + self.dropout2(self.ffn(out1))
        out2 = self.norm2(out2)

        return out2, heat

class MultiHeadTransformer(nn.Module):

    def __init__(
            self,
            dim_query_in,
            dim_keyval_in,
            dim_key,
            dim_inner,
            num_head,
    ):
        super(MultiHeadTransformer, self).__init__()
        self.transformers = nn.ModuleList()
        for i in range(num_head):
            self.transformers.append(
                BasicTransformer(
                    dim_query_in,
                    dim_keyval_in,
                    dim_key,
                    dim_inner
                )
            )

    def forward(self, query, keyval):
        return torch.cat([m(query, keyval) for m in self.transformers], dim=1)

class MultiLayerTransformer(nn.Module):

    def __init__(
            self,
            transforemer_type,
            num_layers,
            *arg,
            **kwargs
    ):
        super(MultiLayerTransformer, self).__init__()
        self.transformers = nn.ModuleList()
        for i in range(num_layers):
            self.transformers.append(
               transforemer_type(
                   *arg, **kwargs
               )
            )

    def forward(self, query, keyval):

        for m in self.transformers:
            query = m(query, keyval)

        return query

