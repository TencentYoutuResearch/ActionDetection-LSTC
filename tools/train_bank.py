#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn as nn
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.parser import parse_args, load_config
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)

train_iter = 0

def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()

    train_meter.iter_tic()
    data_size = len(train_loader)
    logger.info('training for {} iterations'.format(data_size))

    for cur_iter, (inputs, labels, video_idx, meta, FBs, BTs) in enumerate(train_loader):
        # Transfer the data to the current GPU device.

        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            for i in range(len(FBs)):
                if cfg.AVA.GATHER_BANK:
                    FBs[i] = FBs[i].cuda(non_blocking=True) if FBs[i] is not None else None
                else:
                    FBs[i] = [val.cuda(non_blocking=True) for val in FBs[i]] if FBs[i] is not None else None

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        preds = model(FBs)
        preds = preds[meta["has_box"]]

        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]
        loss = loss.item()

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(None, None, None, loss, lr)

        if du.is_master_proc(du.get_local_size()):
            train_meter.log_iter_stats(cur_epoch, cur_iter)
        du.synchronize()
        train_meter.iter_tic()

    # Log epoch stats.
    if du.is_master_proc(du.get_local_size()):
        train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta, FBs, BTs) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            for i in range(len(FBs)):
                if cfg.AVA.GATHER_BANK:
                    FBs[i] = FBs[i].cuda(non_blocking=True) if FBs[i] is not None else None
                else:
                    FBs[i] = [val.cuda(non_blocking=True) for val in FBs[i]] if FBs[i] is not None else None

        # Compute the predictions.
        ori_boxes = meta["ori_boxes"]
        metadata = meta["metadata"]
        raw_label = meta["raw_labels"]

        if cfg.NUM_GPUS > 1:
            ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
            metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

        preds = model(FBs)
        bp = []
        for i in range(cfg.TEST.BATCH_SIZE):
            nbox = torch.sum(ori_boxes[:, 0] == i).item()
            bp.append(preds[i:i+1].expand(nbox, cfg.MODEL.NUM_CLASSES))

        preds = torch.cat(bp, dim=0)

        if cfg.NUM_GPUS > 1:
            preds = torch.cat(du.all_gather_unaligned(preds), dim=0)

        if cfg.NUM_GPUS:
            preds = preds.cpu()
            ori_boxes = ori_boxes.cpu()
            metadata = metadata.cpu()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(preds, ori_boxes, metadata)

        if du.is_master_proc(du.get_local_size()):
            val_meter.log_iter_stats(cur_epoch, cur_iter)
        du.synchronize()

        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)

    val_meter.reset()

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    # precise_bn_loader = loader.construct_loader(
    #     cfg, "train", is_precise_bn=True
    # )

    # Create meters.
    train_meter = AVAMeter(len(train_loader), cfg, mode="train")
    val_meter = AVAMeter(len(val_loader), cfg, mode="val")

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg
        )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(
            cfg, cur_epoch, None
        ):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(
            cfg, cur_epoch, None
        ):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    misc.launch_job(
        cfg=cfg,
        args=args,
        func=train
    )