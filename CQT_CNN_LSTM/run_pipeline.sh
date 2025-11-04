#!/bin/bash
# -*- coding: utf-8 -*-

# -----------------------------
# 运行实时音频处理脚本的 Shell 文件
# -----------------------------

# 设置 Python 脚本路径
PYTHON_SCRIPT="preprocess.py"

# 默认参数
INPUT_DIR="./audio_wav"
OUTPUT_DIR="./features"
AUDIO_PATH=""  # 如果为空则处理整个文件夹
SR=48000
WIN_DURATION=2.5
STEP_DURATION=0.5
HOP_LENGTH=512
BINS_PER_OCTAVE=12
N_BINS=84
DURATION=10.0
NUM_WORKERS=4


# 可通过命令行参数覆盖默认值
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --audio_path) AUDIO_PATH="$2"; shift ;;
        --sr) SR="$2"; shift ;;
        --win_duration) WIN_DURATION="$2"; shift ;;
        --step_duration) STEP_DURATION="$2"; shift ;;
        --hop_length) HOP_LENGTH="$2"; shift ;;
        --bins_per_octave) BINS_PER_OCTAVE="$2"; shift ;;
        --n_bins) N_BINS="$2"; shift ;;
        --duration) DURATION="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 构建 Python 命令
CMD="python $PYTHON_SCRIPT --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --sr $SR \
--win_duration $WIN_DURATION --step_duration $STEP_DURATION --hop_length $HOP_LENGTH \
--bins_per_octave $BINS_PER_OCTAVE --n_bins $N_BINS --duration $DURATION --num_workers $NUM_WORKERS"

# 如果指定了单个音频文件路径，则添加参数
if [[ ! -z "$AUDIO_PATH" ]]; then
    CMD="$CMD --audio_path $AUDIO_PATH"
fi

# 运行 Python 脚本
echo "运行命令: $CMD"
eval $CMD



