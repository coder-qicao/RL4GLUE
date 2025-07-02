#!/bin/bash

# 参数
batch_sentence_num=16
from_local="True"
task="RTE"
project="GLUE"
if [ "$from_local" = "True" ]; then
    model_path="./model/single_sft/qwen2-RTE-24-9e-06-20250630-172111"
else
    # model_path="daryl149/llama-2-7b-chat-hf"
    # model_path="mosaicml/mpt-7b-chat"
    # model_path="Qwen/Qwen2.5-7B-Instruct"
    model_path="Qwen/Qwen2.5-1.5B-Instruct"
fi

if [ "$task" = "MNLI" ]; then
    file_name="test_matched"
elif [ "$task" = "AX" ]; then
    file_name="AX"
else
    file_name="test"
fi

if [ "$project" = "SuperGLUE" ]; then
    suffix="$file_name.jsonl"
elif [ "$task" = "MRPC" ]; then
    suffix="$file_name.txt"
else
    suffix="$file_name.tsv"
fi

data_path="../data/$project/$task/$suffix"

# if [ "$task" = "AX" ]; then
#     data_path="../data/$project/$suffix"
# fi

# 打印使用的模型路径
echo "Using model path: $model_path"
echo "Task: $task"

# 运行Python脚本
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
    --train_model_name $model_path \
    --batch_sentence_num $batch_sentence_num \
    --data_path $data_path \
    --task $task \
    --from_local $from_local
