# 运行训练代码
# 生成模型默认放在cuda:

project="GLUE"
task="MRPC" # {GLUE: MRPC, CoLA, SST, QQP, QNLI, MNLI-m, MNLI-mm, RTE, WNLI, AX, STS-B}; {SuperGLUE: COPA, RTE, WSC, WiC, BoolQ, CB, MultiRC, ReCoRD}
from_local="False"
use_CoT="False"
# 根据from_local的值设置模型路径
if [ "$from_local" = "True" ]; then
    model_path="./model/$task/llama3-16-9e-06-20240716-140737/3"
else
    # model_path="Qwen/Qwen2-7B-Instruct"
    model_path="Qwen/Qwen2.5-1.5B-Instruct"
    # model_path="Qwen/Qwen2.5-7B-Instruct"
    # model_path="mosaicml/mpt-7b-chat"
    # model_path="NousResearch/Llama-2-13b-chat-hf"
fi
if [ "$project" = "SuperGLUE" ]; then
    suffix="jsonl"
else
    suffix="txt"
fi
echo "TASK: $task"
echo "Model Used: $model_path"
CUDA_VISIBLE_DEVICES=1 python3 ppo.py \
--learning_rate 9e-6 \
--start_epoch 0 \
--end_epoch 3 \
--model_temperature 0.8 \
--train_model_name $model_path \
--repeat_time 1 \
--batch_sentence_num 16 \
--checkpoint_rate 1 \
--data_path "../data/$project/$task/train.$suffix" \
--task $task \
--load_in_8bit "False" \
--save_methods "step" \
--save_freq 150 \
--from_local $from_local \
--use_CoT $use_CoT


# learning_rate 学习率
# start_epoch 启动epoch，用来连接之前训练的断点，新开始训练默认为0
# end_epoch 最大epoch
# model_temperature 生成模型训练时的temperature
# train_model_name 生成模型远程或本地路径
# repeat_time 同种语聊重复的次数
# batch_sentence_num 每次处理不同语料的条数
# checkpoint_rate 每多少epoch保存一个checkpoint
