#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.utils.parser import parse_args, load_config

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    logger.info('online test for {} iters'.format(len(test_loader)))
    for cur_iter, (inputs, labels, video_idx, meta, FBs, BTs) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
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

        preds = model(FBs)

        ori_boxes = meta["ori_boxes"]
        metadata = meta["metadata"]
        raw_label = meta["raw_labels"]

        bp = []
        for i in range(cfg.TEST.BATCH_SIZE):
            nbox = torch.sum(ori_boxes[:, 0] == i).item()
            bp.append(preds[i:i + 1].expand(nbox, cfg.MODEL.NUM_CLASSES))

        preds = torch.cat(bp, dim=0)

        if cfg.NUM_GPUS > 1:
            ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
            metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
            preds = torch.cat(du.all_gather_unaligned(preds), dim=0)

        preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
        ori_boxes = (
            ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
        )
        metadata = (
            metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
        )

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds, ori_boxes, metadata)
        test_meter.log_iter_stats(cur_epoch=0, cur_iter=cur_iter)

        test_meter.iter_tic()

    if du.is_master_proc(du.get_local_size()):
        test_meter.finalize_metrics()
    test_meter.reset()


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    du.synchronize()
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
    test_meter = AVAMeter(len(test_loader), cfg, mode="test")

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg)

if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args)

    misc.launch_job(
        cfg=cfg,
        func=test,
        args=args,
    )