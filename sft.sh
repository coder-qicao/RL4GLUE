# 运行训练代码
# 生成模型默认放在cuda:

project="GLUE"
from_local="False"
# 根据from_local的值设置模型路径
if [ "$from_local" = "True" ]; then
    model_path="./model/$task/llama3-16-9e-06-20240716-140737/3"
else
    # model_path="UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3"
    model_path="daryl149/llama-2-7b-chat-hf"
fi
datafolder_path="../data/$project"
lines_per_file=30000
echo "TASK: $project SFT_Multitask"
echo "Model Used: $model_path"
CUDA_VISIBLE_DEVICES=0 python3 sft.py \
--learning_rate 9e-6 \
--start_epoch 0 \
--end_epoch 1 \
--model_temperature 0.8 \
--train_model_name $model_path \
--repeat_time 1 \
--batch_sentence_num 24 \
--checkpoint_rate 1 \
--datafolder_path $datafolder_path \
--load_in_8bit "False" \
--save_methods "step" \
--save_freq 1500 \
--from_local $from_local \
--lines_per_file $lines_per_file
