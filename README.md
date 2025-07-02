# Enhancing NLU in Small LLMs via Reinforcement Learning

Instruction-fine-tuned large language models (LLMs) under 14B parameters often underperform smaller models (like BERT-base) on natural language understanding (NLU) benchmarks such as GLUE and SuperGLUE. Inspired by reinforcement learning's (RL) success in reasoning tasks (e.g., DeepSeek), we investigate Proximal Policy Optimization (PPO) as a framework for boosting the NLU capabilities of these LLMs.

We reframe NLU as an RL problem: token generation is treated as a sequence of actions, optimized using reward signals derived from alignment with ground-truth labels. Our results demonstrate that PPO consistently outperforms supervised fine-tuning (SFT), achieving an average improvement of **6.3 points on GLUE**. PPO also significantly surpasses zero-shot and few-shot prompting by **38.7 and 26.1 points**, respectively. Notably, PPO-tuned models outperform GPT-4o by **over 4% on average** across sentiment and natural language inference tasks, with gains of **7.3% on the Mental Health dataset** and **10.9% on SIGA-nli**.

This work highlights a promising approach for adapting LLMs to new tasks: by formulating them as RL problems, models can learn effectively using simple end-task rewards, reducing reliance on extensive data curation.

## Quick Start
### Requirements
```bash
pip install trl==1.14.0 wandb peft transformers torch
```
### PPO Training
1. Create a script `ppo.sh` with the following content (adjust parameters as needed):
```bash
#!/bin/bash

project="GLUE"
task="MRPC"  # Options: GLUE: {MRPC, CoLA, SST, QQP, QNLI, MNLI-m, MNLI-mm, RTE, WNLI, AX, STS-B}; SuperGLUE: {COPA, RTE, WSC, WiC, BoolQ, CB, MultiRC, ReCoRD}
from_local="False"
use_CoT="False"  # Set to "True" to enable Chain-of-Thought prompting

# Determine model path
if [ "$from_local" = "True" ]; then
    model_path="./model/$task/llama3-16-9e-06-20240716-140737/3"
else
    model_path="Qwen/Qwen2.5-1.5B-Instruct"  # Options: "Qwen/Qwen2.5-7B-Instruct", "mosaicml/mpt-7b-chat", "NousResearch/Llama-2-13b-chat-hf"
fi

# Determine data file suffix
if [ "$project" = "SuperGLUE" ]; then
    suffix="jsonl"
else
    suffix="txt"
fi

echo "TASK: $task"
echo "MODEL: $model_path"

CUDA_VISIBLE_DEVICES=1 python3 ppo.py \
    --learning_rate 9e-6 \
    --start_epoch 0 \
    --end_epoch 3 \
    --model_temperature 0.8 \
    --train_model_name "$model_path" \
    --repeat_time 1 \
    --batch_sentence_num 16 \
    --checkpoint_rate 1 \
    --data_path "../data/$project/$task/train.$suffix" \
    --task "$task" \
    --load_in_8bit "False" \
    --save_methods "step" \
    --save_freq 150 \
    --from_local "$from_local" \
    --use_CoT "$use_CoT"
```
2. Make the script executable and run it:
```bash
chmod +x ppo.sh
./ppo.sh
```

### SFT
1. Create a script `sft.sh` with the following content:
```bash
#!/bin/bash

project="GLUE"
from_local="False"

# Determine model path
if [ "$from_local" = "True" ]; then
    model_path="./model/$task/llama3-16-9e-06-20240716-140737/3"  # Example path, adjust as needed
else
    model_path="daryl149/llama-2-7b-chat-hf"  # Options: "UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3"
fi

datafolder_path="../data/$project"
lines_per_file=30000

echo "TASK: $project SFT_Multitask"
echo "MODEL: $model_path"

CUDA_VISIBLE_DEVICES=0 python3 sft.py \
    --learning_rate 9e-6 \
    --start_epoch 0 \
    --end_epoch 1 \
    --model_temperature 0.8 \
    --train_model_name "$model_path" \
    --repeat_time 1 \
    --batch_sentence_num 24 \
    --checkpoint_rate 1 \
    --datafolder_path "$datafolder_path" \
    --load_in_8bit "False" \
    --save_methods "step" \
    --save_freq 1500 \
    --from_local "$from_local" \
    --lines_per_file "$lines_per_file"
```
2. Make the script executable and run it:
```bash
chmod +x sft.sh
./sft.sh
```

### Few-shots and zero-shot prompting
```bash
# Run Few-Shot Prompting
bash fewshot.sh

# Run Zero-Shot Prompting
bash inference.sh
```