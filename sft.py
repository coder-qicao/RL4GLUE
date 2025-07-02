""" Supervised finetuning """

import torch
from transformers import AutoTokenizer, AutoModel, Adafactor, BitsAndBytesConfig, TrainingArguments
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.core import LengthSampler
from datasets import Dataset, DatasetDict
from trl.trainer import ConstantLengthDataset
import os
import re
from loadmodel import *
from prompt import *
from utils import *
import json
import time
import random
import argparse
import random

# 设置参数
parser = argparse.ArgumentParser(description='加载参数')
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=2)
parser.add_argument('--model_temperature', type=float, default=0.7)
parser.add_argument('--train_model_name', type=str, default="UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3")
parser.add_argument('--repeat_time', type=int, default=1)
parser.add_argument('--checkpoint_rate', type=int, default=1)
parser.add_argument('--batch_sentence_num', type=int, default=16)
parser.add_argument('--datafolder_path', type=str, default="../data/GLUE")
parser.add_argument('--load_in_8bit', type=str, choices=["True", "False"], default="False")
parser.add_argument('--save_methods', type=str, choices=["step", "epoch"], default="epoch")
parser.add_argument("--save_freq", type=int, default=1500)
parser.add_argument('--from_local', type=str, choices=["True", "False"], default="False")
parser.add_argument('--lines_per_file', type=int, default=1000)
args = parser.parse_args()
import wandb
if "SuperGLUE" in args.datafolder_path:
    project = "SuperGLUE"
elif "GLUE" in args.datafolder_path:
    project = "GLUE"
else:
    project = "ALL"
wandb.init(project="multitask")

# 分配模型的cuda编号
train_model_device_id=[0, 1]
encode_model_device_id=[1, 2]

def read_all_files(datafolder_path, randomize, project, is_training):
    sub_folders = [os.path.join(datafolder_path, subfolder) for subfolder in os.listdir(datafolder_path)]
    all_datasets = []
    if project == "GLUE":
        for subfolder in sub_folders:
            if os.path.isdir(subfolder):
                task = subfolder.split("/")[-1]
                if task == "MRPC":
                    file_name = "train.txt" if is_training else "dev.txt"
                elif task == "SNLI":
                    continue
                else:
                    file_name = "train.tsv" if is_training else "dev.tsv"
                file_name = os.path.join(subfolder, file_name)
                data = read_tsv(file_name, task, is_training)
                if len(data) < lines_per_file:
                    data = data * math.ceil(lines_per_file / len(data))
                else:
                    random.shuffle(data)
                    data = data
                print(f"Task: {task}, add {len(data)} lines.")
                all_datasets += data
    elif project == "SuperGLUE":
        for subfolder in sub_folders:
            if os.path.isdir(subfolder):
                task = subfolder.split("/")[-1]
                filename = os.path.join(subfolder, "train.jsonl" if is_training else "val.jsonl")
                if task == "RTE":
                    data = read_tsv(filename.replace("SuperGLUE", "GLUE").replace("jsonl", "tsv").replace("val", "dev"), task, is_training)
                elif task == "AX-g" or task == "AX-b":
                    continue
                else:
                    data = read_jsonl(filename, task)
                if len(data) < lines_per_file:
                    data = data * math.ceil(lines_per_file / len(data))
                else:
                    random.shuffle(data)
                    data = data
                print(f"Task: {task}, add {len(data)} lines.")
                all_datasets += data
    if randomize:
        random.shuffle(all_datasets)
    return all_datasets
    
def read_tsv(file_path, task, is_training):
    delimiter = "\t"
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        data = []
        if task != "CoLA":
            next(file)
        for row in file:
            row = row.strip().split(delimiter)
            if task == "CoLA":
                if len(row) == 4:
                    label = "Yes" if int(row[1]) == 1 else "No"
                    sentence = row[3]
                    data.append({"task": task, "sentence": sentence, "label": label})
            elif task == "RTE":
                if len(row) == 4:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    label = "Yes" if target_to_number[row[3]] == 1 else "No"
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": label})
            elif task == "QQP":
                if len(row) == 6:
                    index = row[0]
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = "Yes" if int(row[5]) == 1 else "No"
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": label})
            elif task == "QNLI":
                if len(row) == 4:
                    index = row[0]
                    question = row[1]
                    sentence = row[2]
                    label = "Yes" if target_to_number[row[3]] == 1 else "No"
                    data.append({"task": task, "question": question, "sentence": sentence, "label": label})
            elif task == "SST":
                if len(row) == 2:
                    sentence = row[0]
                    label = "Positive" if int(row[1]) == 1 else "No"
                    data.append({"task": task, "sentence": sentence, "label": label})
            elif task == "MRPC":
                if len(row) == 5:
                    label = "Yes" if int(row[0])==1 else "No"
                    sentence1, sentence2 = row[3], row[4]
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": label})
            elif task == "WNLI":
                if len(row) == 4:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    label = "Yes" if int(row[3]) == 1 else "No"
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": label})
            elif task == "MNLI" and is_training:
                if len(row) == 12:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = target_to_number[row[-1]]
                    if label == 1:
                        label = "Entailment"
                    elif label == 2:
                        label = "Contradiction"
                    else:
                        label = "Neutral"
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": label})
            elif task == "MNLI" and not is_training:
                if len(row) == 16:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = target_to_number[row[-1]]
                    if label == 1:
                        label = "Entailment"
                    elif label == 2:
                        label = "Contradiction"
                    else:
                        label = "Neutral"
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": label})
            elif task == "STS-B":
                if len(row) == 10:
                    sentence1 = row[7]
                    sentence2 = row[8]
                    score = float(row[-1])
                    data.append({"task": task, "sentence1": sentence1, "sentence2": sentence2, "label": score})
            else:
                print(f"Ignoring invalid row: {row}")
                raise NotImplementedError("Unsupported task")
        return data

def read_jsonl(file_path, task):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = json.loads(line.strip())
            if task == "COPA":
                assert len(line) == 6
                choice1, choice2, premise, idx, label, question = line["choice1"], line["choice2"], line["premise"], \
                                                                    line["idx"], int(line["label"]), line["question"]
                data.append({
                    "task": task, 
                    "question": question, 
                    "premise": premise, 
                    "choice1": choice1, 
                    "choice2": choice2, 
                    "label": label
                })
            elif task == "WSC":
                assert len(line) == 4
                text, target, idx, label = line["text"], line["target"], line["idx"], int(bool(line["label"]))
                span1_idx, span2_idx, span1_text, span2_text = \
                                target["span1_index"], target["span2_index"], target["span1_text"], target["span2_text"]
                data.append({
                    "task": task, 
                    "text": text, 
                    "span1_text": span1_text, 
                    "span2_text": span2_text, 
                    "label": label
                })
            elif task == "WiC":
                assert len(line) == 10
                label, sentence1, sentence2, word = int(bool(line["label"])), line["sentence1"], line["sentence2"], line["word"]
                data.append({
                    "task": task, 
                    "label": label, 
                    "sentence1": sentence1, 
                    "sentence2": sentence2, 
                    "word": word
                })
            elif task == "CB":
                assert len(line) == 4
                premise, hypothesis, label = line["premise"], line["hypothesis"], target_to_number[line["label"]]
                data.append({
                    "task": task, 
                    "sentence1": premise, 
                    "sentence2": hypothesis, 
                    "label": label
                })
            elif task == "BoolQ":
                assert len(line) == 4
                question, passage, label = line["question"], line["passage"], int(bool(line["label"]))
                data.append({
                    "task": task, 
                    "question": question, 
                    "passage": passage, 
                    "label": label
                })
            elif task == "MultiRC":
                assert len(line) == 3
                passage = line["passage"]["text"]
                qa_pairs = line["passage"]["questions"]
                for pair in qa_pairs:
                    question = pair["question"]
                    answers = pair["answers"]
                    for answer_sets in answers:
                        answer = answer_sets["text"]
                        label = int(answer_sets["label"])
                        data.append({
                            "task": task, 
                            "passage": passage, 
                            "question": question, 
                            "answer": answer, 
                            "label": label
                        })
            elif task == "ReCoRD":
                entities = line['passage']['entities']
                text = line['passage']['text']
                cleaned_text = text.split('@highlight')[0].strip()
                qas = line['qas']
                for qa in qas:
                    query = qa['query']
                    answers = qa['answers']
                    answer_list = list({answer['text'] for answer in answers})
                    entity_list = list({text[entity['start']:entity['end']+1] for entity in entities})
                    for answer_text in answer_list:
                        query1 = query.replace("@placeholder", answer_text)
                        data.append({
                            "task": task, 
                            "text": cleaned_text, 
                            "query": query1, 
                            "label": "Yes"
                        })
                    for entity_text in set(entity_list).difference(set(answer_list)):
                        query2 = query.replace("@placeholder", entity_text)
                        data.append({
                            "task": task, 
                            "text": cleaned_text, 
                            "query": query2, 
                            "label": "No"
                        })
            else:
                print(f"Ignoring invalid row: {line}")
                raise NotImplementedError("Unsupported task")
        return data

datafolder_path = args.datafolder_path
sample_num = args.batch_sentence_num
randomize = True
start=args.start_epoch
EPOCH=args.end_epoch
repeat_time=args.repeat_time
lr=args.learning_rate
temperature=args.model_temperature
batch_size=repeat_time*sample_num
mini_batch_size = batch_size
gradient_accumulation_steps = 1
save_method = args.save_methods
load_in_8bit = args.load_in_8bit == "True"
save_freq = args.save_freq
from_local = args.from_local == "True"
lines_per_file = args.lines_per_file

import datetime

# 获取当前时间并格式化为字符串
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 结果存储路径
if start == 0:
    save_root = f"./model/multitask_sft/llama3-{sample_num}-{lr}-{current_time}"
else:
    save_root = f"./model/multitask_sft/llama3-{sample_num}-{lr}-start{start}-{current_time}"
# 被训练模型加载
model_name=args.train_model_name
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="all", # "none", #"lora_only", "all"
    task_type="CAUSAL_LM",
    inference_mode=False,
    use_rslora=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.use_cache = False
print(datafolder_path)
if project == "ALL":
    GLUE_dataset = read_all_files(datafolder_path.replace("ALL", "GLUE"), randomize, "GLUE", is_training=True)
    SuperGLUE_dataset = read_all_files(datafolder_path.replace("ALL", "SuperGLUE"), randomize, "SuperGLUE", is_training=True)
    train_dataset = GLUE_dataset + SuperGLUE_dataset
    random.shuffle(train_dataset)

    # GLUE_dataset = read_all_files(datafolder_path.replace("ALL", "GLUE"), randomize=False, project="GLUE", is_training=False)
    # SuperGLUE_dataset = read_all_files(datafolder_path.replace("ALL", "SuperGLUE"), randomize=False, project="SuperGLUE", is_training=False)
    # eval_dataset = GLUE_dataset + SuperGLUE_dataset
    # random.shuffle(eval_dataset)
else:
    train_dataset = read_all_files(datafolder_path, randomize, project=project, is_training=True)
    eval_dataset = read_all_files(datafolder_path, randomize=False, project=project, is_training=False)

chars_per_token = chars_token_ratio(train_dataset, tokenizer)
train_dataset = ConstantLengthDataset(
    tokenizer,
    train_dataset,
    formatting_func=lambda x: prepare_sample_text(x, tokenizer), 
    infinite=True,
    seq_length=1024,
    chars_per_token=chars_per_token,
)
eval_dataset = ConstantLengthDataset(
    tokenizer,
    eval_dataset,
    formatting_func=lambda x: prepare_sample_text(x, tokenizer),
    infinite=False,
    seq_length=1024,
    chars_per_token=chars_per_token,
)
print(f"Training dataset count: {len(train_dataset)}")
total_iter = len(train_dataset) // sample_num
response_template_with_context = "\nAssistant:\n<Judgement>"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
instruction_template = "System_prompt:\n"
instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, instruction_template=instruction_template, tokenizer=tokenizer)
# 初始化PPOTrainer
training_args = TrainingArguments(
    bf16=True, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=save_freq,
    torch_compile=True,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=lr,
    log_level="info",
    logging_steps=10,
    logging_strategy="steps",
    # lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=EPOCH-start,
    output_dir=save_root,
    overwrite_output_dir=True,
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    report_to="wandb",
    save_strategy="steps",
    save_steps=save_freq,
    save_total_limit=20,
    seed=42
)

optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=lr,
)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1) ##TODO: Modify this
lr_scheduler = CosineAnnealingWarmupLR(
    optimizer, 
    T_max=total_iter * (EPOCH - start), 
    eta_min=lr / 10, 
    warmup_steps=total_iter * (EPOCH - start) // 10
) ##TODO: Modify this
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=lambda x: prepare_sample_text(x, tokenizer),
    tokenizer=tokenizer,
    peft_config=lora_config,
    max_seq_length=None,
    optimizers=(None, lr_scheduler),
    data_collator=data_collator,
)

train_result = trainer.train()
metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = max_train_samples
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
if training_args.do_eval:
    metrics = trainer.evaluate(
        metric_key_prefix="eval",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    max_eval_samples = (
        training_args.max_eval_samples if training_args.max_eval_samples is not None else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
