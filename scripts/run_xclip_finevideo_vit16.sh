#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# 配置路径和参数
DATA_PATH=data/yidong
OUTPUT_PATH=ckpts/xclip_finevideo_vit16_${TIMESTAMP}
MODEL_PATH=/sshfs/pretrains/openai/clip-vit-base-patch16
job_name=xclip_finevideo_vit16_${TIMESTAMP}  # 在job_name中加入时间戳

# 创建日志目录（如不存在）
mkdir -p logs

# 记录开始时间
echo "=== 训练开始于 $(date) ===" | tee -a logs/${job_name}

# 启动单卡训练
CUDA_VISIBLE_DEVICES=0 \
python main_xclip.py \
  --do_train \
  --do_eval \
  --cache_dir ${MODEL_PATH} \
  --datatype finevideo \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_PATH} \
  --batch_size 16 \
  --batch_size_val 16 \
  --epochs 1 \
  --max_words 64 \
  --max_frames 64 \
  --feature_framerate 1 \
  --lr 5e-5 \
  --pretrained_clip_name "ViT-B/16" 2>&1 | tee -a logs/${job_name}

# 记录结束时间和总耗时
echo "=== 训练结束于 $(date) ===" | tee -a logs/${job_name}
echo "=== 总耗时: $SECONDS 秒 ===" | tee -a logs/${job_name}
