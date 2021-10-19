#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import functools
import numpy as np
import torch
import random
import math
import functools
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from slowfast.datasets.multigrid_helper import ShortCycleBatchSampler
from slowfast.utils.logging import get_logger

from .build import build_dataset
from collections import defaultdict

logger = get_logger(__name__)

class SequentialDistributedSampler(DistributedSampler):

    """
    a video specific sampler to ensure that each process focus on
    processing a distinct video clip sequence
    """

    def __init__(self, cfg, train, dataset):
        super(SequentialDistributedSampler, self).__init__(dataset)
        # make video samplers divisible

        if train:
            batch_size = cfg.TRAIN.BATCH_SIZE // self.num_replicas
        else:
            batch_size = cfg.TEST.BATCH_SIZE // self.num_replicas
        self.batch_size = batch_size

        # reorganize the clip index, so that each shot can be
        # indexed via a specific clip id
        self.shot_to_clip = dict()
        for vid in range(dataset.num_videos()):
            indexes = dataset.get_idx_sequence_from_video(vid)
            if len(indexes) % self.batch_size > 0:
                left = self.batch_size - len(indexes) % self.batch_size
                indexes += indexes[-left:]

            self.shot_to_clip.update(
                {
                    indexes[idx] : indexes[idx:idx+self.batch_size] \
                    for idx in range(0, len(indexes), self.batch_size)
                }
            )

        self.shot_keys = list(self.shot_to_clip.keys())
        self.shot_keys.sort()
        self.num_shot = len(self.shot_to_clip)
        self.total_shot = int(math.ceil(self.num_shot * 1.0 / self.num_replicas)) \
                          * self.num_replicas
        self.num_clips = self.total_shot * self.batch_size
        self.num_samples = self.num_clips // self.num_replicas

        self.seed = 0
        logger.info('{} shots, {} clips'.format(self.num_shot, self.num_clips))

    def __iter__(self):

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            random_indices = torch.randperm(len(self.shot_keys), generator=g).tolist()
            video_indices = [self.shot_keys[idx] for idx in random_indices]
            video_indices += video_indices[:(self.total_shot - self.num_shot)]
            shot_indices = video_indices[self.rank:self.total_shot:self.num_replicas]

            indices = functools.reduce(
                lambda x, y: x+y,
                [self.shot_to_clip[s] for s in shot_indices]
            )
        else:
            indices = list(range(len(self.dataset)))

            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]

        logger.info('num samples {} {}'.format(len(indices), self.num_samples))
        assert len(indices) == self.num_samples, \
            "{} vs. {}".format(len(indices), self.num_samples)

        return iter(indices)

def detection_collate(batch, cfg):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data, feat_banks, bank_times = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)

    collated_extra_data = {
        "raw_labels": torch.tensor(np.concatenate(labels, axis=0)).float()
    }

    if not cfg.AVA.GATHER_BANK:
        new_labels = []
        for label in labels:
            if label.shape[0] > 0:
                new_labels.append(np.max(label, axis=0, keepdims=True))
            else:
                new_labels.append(label)

        labels = new_labels

    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    feat_banks = list(feat_banks)
    bank_times = list(bank_times)

    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]

        if key == "boxes":
            has_box = [d.shape[0] > 0 for d in data]
            collated_extra_data["has_box"] = torch.Tensor(has_box).bool()

        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data, feat_banks, bank_times

def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
        # Create a sampler for multi-process training
        sampler = (
            DistributedSampler(dataset)
            if cfg.NUM_GPUS > 1
            else RandomSampler(dataset)
        )
        batch_sampler = ShortCycleBatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
        )
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
    else:
        # Create a sampler for multi-process training
        if cfg.NUM_GPUS > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = None
        # Create a loader
        collate_fn = functools.partial(detection_collate, cfg=cfg)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_fn if cfg.DETECTION.ENABLE else None,
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = (
        loader.batch_sampler.sampler
        if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
        else loader.sampler
    )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler, SequentialDistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler) or issubclass(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
