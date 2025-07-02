#!/bin/bash

# 参数
batch_sentence_num=128
from_local="False"
task="BoolQ"
project="SuperGLUE"
if [ "$from_local" = "True" ]; then
    model_path="./model/single_sft/llama3-CoLA-8-9e-06-20240917-201347/checkpoint-1069"
else
    model_path="daryl149/llama-2-7b-chat-hf"
    # model_path="mosaicml/mpt-7b-chat"
    # model_path="Qwen/Qwen2.5-7B-Instruct"
fi

if [ "$task" = "MNLI" ]; then
    file_name="test_matched"
elif [ "$task" = "AX-b" ]; then
    file_name="AX-b"
elif [ "$task" = "AX-g" ]; then
    file_name="AX-g"
else
    file_name="test"
fi

if [ "$project" = "SuperGLUE" ]; then
    suffix="$file_name.jsonl"
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
CUDA_VISIBLE_DEVICES=1 python3 fewshot.py \
    --train_model_name $model_path \
    --batch_sentence_num $batch_sentence_num \
    --data_path $data_path \
    --task $task \
    --from_local $from_local