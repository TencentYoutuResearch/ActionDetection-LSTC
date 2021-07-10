#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
import time

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, HieveMeter

logger = logging.get_logger(__name__)

def visualize_tensor(slow, imgs, boxes, metadata, attns, cfg, dir='vis'):

    nbox = metadata.shape[0]
    batchsize, _, _, height, width = slow.shape

    for b in range(batchsize):
        clip_idx = boxes[:, 0] == b
        if clip_idx.shape[0] == 0:
            continue

        meta = metadata[clip_idx]
        box = boxes[clip_idx, 1:]
        data = (slow[b] * cfg.DATA.STD[0] + cfg.DATA.MEAN[0]) * 255
        data = data.numpy().astype(np.uint8).transpose([1, 2, 3, 0])[:, :, :, ::-1]
        attn = attns[clip_idx]
        img = (imgs[b] * cfg.DATA.STD[0] + cfg.DATA.MEAN[0]) * 255
        img = img.numpy().astype(np.uint8).transpose([1, 2, 0])[:, :, ::-1]

        time = attn.shape[1]
        heats = F.interpolate(attn, size=(height, width), mode='bilinear', align_corners=True)
        heats *= 100
        heats[heats>=255] = 255.0
        heats = heats.numpy().astype(np.uint8)
        print(heats.shape, heats.min(), heats.max(), heats.mean(), heats.var())

        if heats.max() < 225:
            continue

        for i in range(clip_idx.shape[0]):
            pimg = img.copy()
            vid, sec = int(meta[i, 0].item()), int(meta[i, 1].item())
            vis_path = os.path.join(dir, str(vid), str(sec))
            if not os.path.exists(vis_path):
                os.makedirs(vis_path, exist_ok=True)

            pt1 = (int(box[i, 0].item() * width), int(box[i, 1].item() * height))
            pt2 = (int(box[i, 2].item() * width), int(box[i, 3].item() * height))
            cv2.rectangle(pimg, pt1, pt2, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(vis_path, 'img_{}.jpg'.format(i+1)), pimg)

            for t in range(time):
                heat = cv2.applyColorMap(heats[i, t], colormap=cv2.COLORMAP_JET)
                color = cv2.addWeighted(data[2 * t], 0.5, heat, 0.5, 0.0)
                cv2.imwrite(os.path.join(vis_path, 'attn_{}_{}.jpg'.format(i+1, t+1)), color)

def visualize_results(imgs, boxes, metadata, short_pred, long_pred, cfg, label_map, dir='results'):
    nbox = metadata.shape[0]
    batchsize, _, height, width = imgs.shape

    for b in range(batchsize):
        clip_idx = boxes[:, 0] == b
        if clip_idx.shape[0] == 0:
            continue

        meta = metadata[clip_idx]
        box = boxes[clip_idx, 1:]
        img = (imgs[b] * cfg.DATA.STD[0] + cfg.DATA.MEAN[0]) * 255
        img = img.numpy().astype(np.uint8).transpose([1, 2, 0])[:, :, ::-1]

        sp = short_pred[clip_idx]
        lp = long_pred[clip_idx]

        for i in range(clip_idx.shape[0]):
            pimg = img.copy()
            canvas = np.zeros(pimg.shape, dtype=np.uint8)
            vid, sec = int(meta[i, 0].item()), int(meta[i, 1].item())
            vis_path = os.path.join(dir, str(vid), str(sec))
            if not os.path.exists(vis_path):
                os.makedirs(vis_path, exist_ok=True)

            sc, si = torch.max(sp[i], dim=0)
            sc = float(sc.item())
            si = int(si.item())

            lc, li = torch.max(lp[i], dim=0)
            lc = float(lc.item())
            li = int(li.item())

            if si == li:
                continue

            ns, nl = label_map[si+1], label_map[li+1]
            print(ns, nl)
            pt1 = (int(box[i, 0].item()), int(box[i, 1].item()))
            pt2 = (int(box[i, 2].item() ), int(box[i, 3].item()))
            print(pt1, pt2, pimg.shape)
            cv2.rectangle(canvas, (pt1[0], pt1[1]+1), (pt2[0], pt1[1]+21), color=(0, 128, 0), thickness=-1)
            cv2.rectangle(canvas, (pt1[0], pt1[1]+25), (pt2[0], pt1[1]+45), color=(160, 128, 0), thickness=-1)
            cv2.rectangle(pimg, pt1, pt2, color=(0, 0, 255), thickness=2)
            cv2.putText(pimg, '{}'.format(ns), (pt1[0], pt1[1]+14), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(255, 255, 255), thickness=1)
            cv2.putText(pimg, '{}'.format(nl), (pt1[0], pt1[1]+38), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(255, 255, 255), thickness=1)
            pimg = cv2.addWeighted(canvas, 0.45, pimg, 1.0, 0.0)

            cv2.imwrite(os.path.join(vis_path, 'img_{}.jpg'.format(i + 1)), pimg)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
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
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
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
                FBs[i] = FBs[i].cuda(non_blocking=True) if FBs[i] is not None else None

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.

            if cfg.LSTC.ENABLE:
                preds, sp, lp = model(inputs, bboxes=meta["boxes"], extract=False, FBs=FBs, BTs=BTs)
            else:
                preds = model(inputs, bboxes=meta["boxes"], extract=False, FBs=FBs, BTs=BTs)

            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            # imgs = inputs[-1][:, :, 16].cpu()
            # slow = inputs[0].cpu()
            # attn = attn.cpu()
            # visualize_tensor(slow, imgs, meta["boxes"].cpu(), metadata, attn, cfg)
            # visualize_results(imgs, meta["boxes"].cpu(), metadata, sp, lp, cfg, labelmap)

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
        else:
            # Perform the forward pass.
            preds = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()
            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
    # Log epoch stats and print the final testing results.
    if writer is not None and not cfg.DETECTION.ENABLE:
        all_preds = [pred.clone().detach() for pred in test_meter.video_preds]
        all_labels = [
            label.clone().detach() for label in test_meter.video_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(preds=all_preds, labels=all_labels)

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

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        if cfg.TEST.DATASET == 'ava':
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            test_meter = HieveMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    tic = time.time()
    perform_test(test_loader, model, test_meter, cfg, writer)
    toc = time.time()
    logger.info(f"total inference is {(toc - tic):.3f}s")
    test_loader.dataset.close()
    if writer is not None:
        writer.close()