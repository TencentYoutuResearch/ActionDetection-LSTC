# LSTC: Boosting Atomic Action Detection with Long-Short-Term Context

This Repository contains the code on AVA of our ACM MM 2021 paper: LSTC: Boosting Atomic Action Detection with Long-Short-Term Context

## Installation

See [INSTALL.md](./INSTALL.md) for details on installing the codebase, including requirement and environment settings

## Data

For data preparation and setup, our LSTC strictly follows the processing of [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md),
See [DATASET.md](./DATASET.md) for details on preparing the data.

## Run the code

We take SlowFast-ResNet50 as an example

* train the model
```shell script
python3 tools/run_net.py --cfg config/AVA/SLOWFAST_32x12_R50_LFB.yaml \
    AVA.FEATURE_BANK_PATH 'path/to/feature/bank/folder' \
    TRAIN.CHECKPOINT_FILE_PATH 'path/to/pretrained/backbone' \
    OUTPUT_DIR 'path/to/output/folder'
```

* test the model
```shell script
python3 tools/run_net.py --cfg config/AVA/SLOWFAST_32x12_R50_LFB.yaml \
    AVA.FEATURE_BANK_PATH 'path/to/feature/bank/folder' \
    OUTPUT_DIR 'path/to/output/folder' \
    TRAIN.ENABLE False \ 
    TEST.ENABLE True
```

*If you want to start the DDP training from command line with `torch.distributed.launch`, please set `start_method='cmd'` in `tools/run_net.py`*

## Resource

The codebase provide following resources for fast training and validation

### Pretrained backbone on Kinetics

| backbone | dataset | model type | link |
|----------|:---------------------:|:------------:|:--------------:|
|ResNet50|Kinetics400|caffe2|[google drive](https://drive.google.com/file/d/1zxS57DAXiLswWG-hI8s76zGdtRFNRgxa/view?usp=sharing)|
|ResNet101|Kinetics600|caffe2|[google drive](https://drive.google.com/file/d/1U6i2lGo8-qdtL_UDPHHCHwmfOERJxfnK/view?usp=sharing)|

### Extracted long term feature bank

| backbone | feature bank | dimension |
|----------|:---------------------:|:------------:|
|ResNet50|[LMDB](https://drive.google.com/file/d/1IqFuq7GMSBFnHopjbNcDJAIES1EtxpQR/view?usp=sharing)|1280|
|ResNet101|[LMDB](https://drive.google.com/file/d/1ND4sSGwAv2SFR42J90Vj9cNn1glz1Ex3/view?usp=sharing)|2304|

### Checkpoint file

| backbone | checkpoint | model type |
|----------|:---------------------:|:-----------:|
|ResNet50|[google drive](https://drive.google.com/file/d/1yimMvcOXaASOFOmp64HKO13LzS5b_YCj/view?usp=sharing)|pytorch|
|ResNet101|[google drive](https://drive.google.com/file/d/1BZ4MzlhUOzuvBPyaS8DAHcyGikh6TAJh/view?usp=sharing)|pytorch|

## Acknowledgement

This codebase is built upon [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md), please follow the official instruction to install the repository.

## Citation

If you find this repository helps your research, please refer following paper
```bibtex
@InProceedings{Yuxi_2021_ACM,
  author = {Li, Yuxi and Zhang, Boshen and Li, Jian and Wang, Yabiao and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Lin, Weiyao},
  title = {Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation},
  booktitle = {ACM Conference on Multimedia},
  month = {October},
  year = {2021}
} 
```