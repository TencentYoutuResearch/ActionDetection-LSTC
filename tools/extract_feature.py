#----------------------------------------------------------------------------
#
#      A Basic Script for Feature Bank Extraction Using Trained Models
#
#----------------------------------------------------------------------------

import numpy as np
import torch
import pickle
import os
import os.path as osp
import lmdb

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging

import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, TrainMeter, HieveMeter
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.misc import launch_job

logger = logging.get_logger(__name__)

@torch.no_grad()
def feature_extraction_launch(loader, model, meter, cfg, feature_bank):
    # Enable eval mode.
    model.eval()

    logger.info('extract feature for {} iters'.format(len(loader)))
    for cur_iter, (inputs, labels, video_idx, meta, _, _, _) in enumerate(loader):

        meter.iter_tic()
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

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            feat, ctx = model(inputs, meta["boxes"], meta["metadata"], extract=True)

            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
                feats = torch.cat(du.all_gather_unaligned(feat), dim=0)

            feats = feats.detach().cpu() if cfg.NUM_GPUS else feats.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if du.is_master_proc(du.get_world_size()):
                num = metadata.shape[0]
                for i in range(num):
                    vid, sec = int(metadata[i, 0].item()), int(metadata[i, 1].item())

                    if vid not in feature_bank:
                        feature_bank[vid] = dict()

                    if sec not in feature_bank[vid]:
                        feature_bank[vid][sec] = []

                    feature_bank[vid][sec].append(feats[i].squeeze())

                meter.iter_toc()
                meter.log_iter_stats(cur_epoch=0, cur_iter=cur_iter)
            du.synchronize()
        else:
            raise NotImplementedError()

    meter.reset()

def write(feature_bank, cfg):

    # save as lmdb

    env = lmdb.open(os.path.join(cfg.AVA.FEATURE_BANK_PATH, 'rdb'), map_size=3e10)
    txn = env.begin(write=True)
    count = 0

    for split in feature_bank:
        for vid in feature_bank[split]:
            for sec in feature_bank[split][vid]:
                feat_key = f"{split}/{vid}/{sec}/feature"
                feat_val = pickle.dumps(feature_bank[split][vid][sec])
                txn.put(key=feat_key.encode(), value=feat_val)

                count += 1
                if count % 2000 == 0:
                    logger.info(f"commit for {count} frames")
                    txn.commit()
                    txn = env.begin(write=True)

    txn.commit()
    env.close()

def extract_feature(cfg):
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
    logger.info("Extracting AVA features with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    train_loader = loader.construct_loader(cfg, "train")
    test_loader = loader.construct_loader(cfg, "test")

    # Create video feature bank
    # format {video_idx: {sec: {name: [feature1, feature2, ....]}}} name in ['feature', 'context']
    feature_bank = {
        'train': dict(),
        'test': dict()
    }

    if du.is_master_proc() and not osp.exists(cfg.AVA.FEATURE_BANK_PATH):
        os.makedirs(cfg.AVA.FEATURE_BANK_PATH)

    test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    train_meter = AVAMeter(len(train_loader), cfg, mode="test")

    # main process for feature extraction
    feature_extraction_launch(train_loader, model, train_meter, cfg, feature_bank['train'])
    feature_extraction_launch(test_loader, model, test_meter, cfg, feature_bank['test'])

    if du.is_master_proc(du.get_world_size()):
        write(feature_bank, cfg)

if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args)

    launch_job(
        cfg=cfg,
        args=args,
        func=extract_feature
    )
