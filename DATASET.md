# Dataset Preparation

The AVA Dataset could be downloaded from the [official site](https://research.google.com/ava/download.html#ava_actions_download)

We followed the same [downloading and preprocessing procedure](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/DATASET.md) as the [Long-Term Feature Banks for Detailed Video Understanding](https://arxiv.org/abs/1812.05038) do.

You could follow these steps to download and preprocess the data:

1. Download videos

```
DATA_DIR="../../data/ava/videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

for line in $(cat ava_file_names_trainval_v2.1.txt)
do
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done
```

2. Cut each video from its 15th to 30th minute

```
IN_DATA_DIR="../../data/ava/videos"
OUT_DATA_DIR="../../data/ava/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done
```

3. Extract frames

```
IN_DATA_DIR="../../data/ava/videos_15min"
OUT_DATA_DIR="../../data/ava/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
```

4. Download annotations

```
DATA_DIR="../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://research.google.com/ava/download/ava_train_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_train_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
```

5. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv)) and put them in
the `frame_lists` folder (see structure above).

6. Download person boxes ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv), [test](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv)) and put them in the `annotations` folder (see structure above).
If you prefer to use your own person detector, please see details
in [here](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/GETTING_STARTED.md#ava-person-detector).


Download the ava dataset with the following structure:

```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv
```

You could also replace the `v2.1` by `v2.2` if you need the AVA v2.2 annotation. You can also download some pre-prepared annotations from [here](https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar).

7. Setup the root folder. In your training and testing phase please ensure your root folder is correctly set in the config file.
You can set `_C.DATA_DIR=/path/to/AVA/folder` in `slowfast/config/defaults.py` before setting up, or config them in the command line

```
DATA_DIR /path/to/AVA/folder ${OTHER COMMAND}
```