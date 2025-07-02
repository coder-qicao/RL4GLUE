"""
模型训练代码
使用module.py中定义的reward对llama生成的结果判分
使用trl中的PPO方法对llama参数进行更新
"""

import torch
from transformers import AutoTokenizer, Adafactor
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

import os
import re
from prompt import *
from utils import *
import json
import time
import random
import argparse
import csv

# 设置参数
parser = argparse.ArgumentParser(description='加载参数')
parser.add_argument('--train_model_name', type=str, default="daryl149/llama-2-7b-chat-hf")
parser.add_argument('--batch_sentence_num', type=int, default=8)
parser.add_argument('--data_path', type=str, default="./data/CoLA/train.tsv")
parser.add_argument("--task", type=str, default="CoLA")
parser.add_argument('--from_local', type=str, choices=["True", "False"], default="False")
args = parser.parse_args()
project = "SuperGLUE" if args.task not in ["CoLA", "MRPC", "QQP", "MNLI", "QNLI", "RTE", "WNLI", "AX", "STS-B", "SST"] else "GLUE"
# 分配模型的cuda编号
train_model_device_id=[0, 1]
encode_model_device_id=[1, 2]

def read_tsv(file_path, batch_size=32, kaggle=False, task=args.task):
    delimiter = "\t"
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        # tsv_reader = csv.reader(file, delimiter='\t')
        data = []
        next(file)  # 跳过列名行

        for row in file:
            row = row.strip().split(delimiter)
            if task == "CoLA":
                if len(row) == 2:
                    index = row[0]
                    sentence = row[1]
                    data.append((index, sentence))
            elif task == "RTE":
                if len(row) == 3:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    data.append((index, sentence1, sentence2))
            elif task == "QQP":
                if len(row) == 3:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    data.append((index, sentence1, sentence2))
            elif task == "QNLI":
                if len(row) == 3:
                    index = row[0]
                    question = row[1]
                    sentence = row[2]
                    data.append((index, question, sentence))
            elif task == "SST":
                if len(row) == 2:
                    index = row[0]
                    sentence = row[1]
                    data.append((sentence))
            elif task == "MRPC": ## TODO: Needs to be modified.
                if len(row) == 5:
                    sentence1, sentence2 = row[3], row[4]
                    data.append((sentence1, sentence2))
            elif task == "WNLI":
                if len(row) == 3:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    data.append((index, sentence1, sentence2))
            elif task == "MNLI":
                sentence1 = row[8]
                sentence2 = row[9]
                data.append((sentence1, sentence2))
            elif task == "STS-B":
                if len(row) == 9:
                    sentence1 = row[7]
                    sentence2 = row[8]
                    data.append((sentence1, sentence2))
                else:
                    print(f"Ignoring invalid row: {row}")
                    raise NotImplementedError("Unsupported task")
            elif task == "AX":
                if len(row) == 3:
                    sentence1 = row[1]
                    sentence2 = row[2]
                    data.append((sentence1, sentence2))
            else:
                print(f"Ignoring invalid row: {row}")
                raise NotImplementedError("Unsupported task")
        print(len(data))

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            yield batch

def load_jsonl(file_path, task='task'):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = json.loads(line.strip())
            if task == "COPA":
                assert len(line) == 5
                choice1, choice2, premise, idx, question = line["choice1"], line["choice2"], line["premise"], \
                                                                        line["idx"], line["question"]
                data.append((premise, choice1, choice2, question, idx))
            elif task == "WSC":
                assert len(line) == 3
                text, target, idx = line["text"], line["target"], line["idx"]
                span1_idx, span2_idx, span1_text, span2_text = \
                                target["span1_index"], target["span2_index"], target["span1_text"], target["span2_text"]
                
                data.append((text, span1_idx, span2_idx, span1_text, span2_text, idx))
            elif task == "WiC":
                assert len(line) == 9
                idx, sentence1, sentence2, word = line["idx"], line["sentence1"], line["sentence2"], line["word"]
                data.append((idx, sentence1, sentence2, word))
            elif task == "CB":
                assert len(line) == 3
                premise, hypothesis, idx = line["premise"], line["hypothesis"], line["idx"]
                data.append((premise, hypothesis, idx))
            elif task == "BoolQ":
                assert len(line) == 3
                question, passage, idx = line["question"], line["passage"], int(line["idx"])
                data.append((question, passage, idx))
            elif task == "MultiRC":
                assert len(line) == 3
                idx = line["idx"]
                passage = line["passage"]["text"]
                qa_pairs = line["passage"]["questions"]
                for pair in qa_pairs:
                    question = pair["question"]
                    question_idx = pair["idx"]
                    answers = pair["answers"]
                    for answer_sets in answers:
                        answer = answer_sets["text"]
                        answer_idx = answer_sets["idx"]
                        data.append((passage, question, answer, idx, question_idx, answer_idx))
            elif task == "ReCoRD":
                assert len(line) == 4
                entities = line['passage']['entities']
                text = line['passage']['text']
                cleaned_text = text.split('@highlight')[0].strip()
                qas = line['qas']
                idx = line['idx']
                for qa in qas:
                    query = qa['query']
                    query_idx = qa['idx']
                    entity_list = []
                    for answer_idx, entity in enumerate(entities):
                        start = entity['start']
                        end = entity['end']
                        candidate = text[start:end+1]
                        if candidate not in entity_list:
                            entity_list.append(candidate)
                        data.append((cleaned_text, query, candidate, idx, query_idx, answer_idx))
            elif task == "AX-b":
                assert len(line) >= 4
                idx, sentence1, sentence2 = line["idx"], line["sentence1"], line["sentence2"]
                data.append((idx, sentence1, sentence2))
            elif task == "AX-g":
                assert len(line) == 5
                idx, hypothesis, premise = line["idx"], line["hypothesis"], line["premise"]
                data.append((idx, hypothesis, premise))
            elif task == "RTE":
                assert len(line) == 3
                idx, hypothesis, premise = line["idx"], line["hypothesis"], line["premise"]
                data.append((idx, premise, hypothesis))
            else:
                print(f"Ignoring invalid row: {line}")
                raise NotImplementedError("Unsupported task")
        return data

def get_data_length(file_path, task='task'):
    data = load_jsonl(file_path, task)
    return len(data)

def read_jsonl(file_path, batch_size, task=args.task):
    data = load_jsonl(file_path, task)

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield batch


file_path = args.data_path
sample_num = args.batch_sentence_num
from_local = args.from_local == "True"

batch_size = sample_num
mini_batch_size = batch_size
gradient_accumulation_steps = 1

# 被训练模型加载
model_name=args.train_model_name
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="all", #"none", #"lora_only", "all"
    task_type="CAUSAL_LM",
    inference_mode=True,
    use_rslora=True
)
print(model_name, from_local)
if from_local:
    base_model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    base_model = base_model.merge_and_unload(progressbar=True, safe_merge=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        use_auth_token=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        peft_config=lora_config,
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        use_auth_token=True,
        device_map="auto",
        torch_dtype=torch.float16,
        peft_config=lora_config,
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = model.to(f'cuda:{train_model_device_id[0]}')
model.eval()
ppo_config = {
    "batch_size": batch_size,
    "learning_rate": 0,
    "mini_batch_size": mini_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "model_name": model_name,
    "optimize_cuda_cache": True,
}
config = PPOConfig(**ppo_config)
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=config.learning_rate,
)
ppo_trainer = PPOTrainer(config, model, None, tokenizer, optimizer=optimizer)
# eval部分
if project == "GLUE":
    data_generator = read_tsv(file_path, batch_size=sample_num)
    total_iter = count_lines(file_path) // sample_num
elif project == "SuperGLUE":
    data_generator = read_jsonl(file_path, batch_size=sample_num)
    total_iter = get_data_length(file_path, task=args.task) // sample_num
# 统计正确率
total = 0

# 模型输出，产生criteria
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
output_length_sampler = LengthSampler(24, 32)

# 生成一个batch的query
y_pred = []
cnt = {"entailment": 0, "neutral": 0, "contradiction": 0}
cnt_1 = {"yes": 0, "no": 0}
cnt_interval={"0": 0, "1":0, "2":0, "3":0, "4":0, "5":0}
unmatched_count = 0

for iteration, batch in enumerate(data_generator):
    print(f"iter {iteration} / {total_iter}")
    # if iteration < 2250:
    #     continue
    # if iteration >= 2650:
    #     break
    query_tensor = []
    idxes = []
    question_idxes, answer_idxes = [], []
    candidates = []
    # 处理每个批次的数据
    for data_row in batch:
        # 读取数据
        if args.task == "CoLA":
            index, sentence = data_row
            prompt = prompt_tmp["cola"].format(sentence=sentence)
        elif args.task == "RTE":
            idx, sentence1, sentence2 = data_row
            prompt = prompt_tmp["rte"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "QQP":
            idx, sentence1, sentence2 = data_row
            prompt = prompt_tmp["qqp"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "QNLI":
            idx, question, sentence = data_row
            prompt = prompt_tmp["qnli"].format(question=question, sentence=sentence)
        elif args.task == "SST":
            sentence = data_row
            prompt = prompt_tmp["sst"].format(sentence=sentence)
        elif args.task == "MRPC": ## TODO: Needs to be modified.
            sentence1, sentence2 = data_row
            prompt = prompt_tmp["mrpc"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "WNLI":
            index, sentence1, sentence2 = data_row
            prompt = prompt_tmp["wnli"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "MNLI":
            sentence1, sentence2 = data_row
            prompt = prompt_tmp["mnli"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "STS-B":
            sentence1, sentence2 = data_row
            prompt = prompt_tmp["sts-b"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "AX":
            sentence1, sentence2 = data_row
            prompt = prompt_tmp["mnli"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "WSC":
            text, span1_idx, span2_idx, span1_text, span2_text, idx = data_row
            prompt = prompt_tmp["wsc"].format(text=text, span1_idx=span1_idx, span2_idx=span2_idx, span1_text=span1_text, span2_text=span2_text)
        elif args.task == "COPA":
            premise, choice1, choice2, question, idx = data_row
            prompt = prompt_tmp["copa"].format(premise=premise, choice1=choice1, choice2=choice2, question=question)
        elif args.task == "WiC":
            idx, sentence1, sentence2, word = data_row
            prompt = prompt_tmp["wic"].format(word=word, sentence1=sentence1, sentence2=sentence2)
        elif args.task == "CB":
            premise, hypothesis, idx = data_row
            prompt = prompt_tmp["mnli"].format(sentence1=premise, sentence2=hypothesis)
        elif args.task == "BoolQ":
            question, passage, idx = data_row
            prompt = prompt_tmp["boolq"].format(question=question, passage=passage)
        elif args.task == "MultiRC":
            passage, question, answer, idx, question_idx, answer_idx = data_row
            prompt = prompt_tmp["multirc"].format(passage=passage, question=question, answer=answer)
            question_idxes.append(question_idx)
            answer_idxes.append(answer_idx)
        elif args.task == "ReCoRD":
            text, query, candidate, idx, query_idx, answer_idx = data_row
            prompt = prompt_tmp["record"].format(text=text, query=query.replace("@placeholder", candidate))
            question_idxes.append(query_idx)
            answer_idxes.append(answer_idx)
            candidates.append(candidate)
        elif args.task == "AX-b":
            idx, sentence1, sentence2 = data_row
            prompt = prompt_tmp["rte"].format(sentence1=sentence1, sentence2=sentence2)
        elif args.task == "AX-g":
            idx, hypothesis, premise = data_row
            prompt = prompt_tmp["rte"].format(sentence1=premise, sentence2=hypothesis)
        else:
            raise NotImplementedError("Unsupported task")
        
        query_tensor_nlp = tokenizer.encode(prompt, return_tensors="pt")[0].to(f'cuda:{train_model_device_id[0]}')
        query_tensor_nlp = [query_tensor_nlp]
        query_tensor += query_tensor_nlp
        idxes.append(idx)

    start_time = time.time()
    response_tensor_nlp = ppo_trainer.generate(query_tensor, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs) # length_sampler=output_length_sampler,
    input_ids = response_tensor_nlp
    result = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    end_time = time.time()
    print("生成时间：", end_time-start_time, "秒")
    # 计算rewar前只取出最终判断，不关注分析过程
    pattern_0 = r"(.*?)</Judgement>"
    reward_result=[]
    for i, sentence in enumerate(result):
        match_0 = re.search(pattern_0, sentence, re.DOTALL)
        
        if match_0:
            result_value = match_0.group(1).strip()
        else:
            result_value = sentence
            print("None branch:", sentence, "\n----------")
        reward_result.append(result_value)
    # 筛选生成结果
    judgement = []
    if args.task == "QNLI" or (args.task == "RTE" and project == "GLUE"):
        judgement = []
        for x in reward_result:
            if ("Yes" in x) or ("yes" in x):
                judgement.append("entailment")
                cnt_1["yes"] += 1
            elif ("No" in x) or ("no" in x):
                judgement.append("not_entailment")
                cnt_1["no"] += 1
            else:
                judgement.append(random.choice(["not_entailment", "entailment"]))
    elif args.task == "MNLI" or args.task == "AX":
        judgement = []
        for x in reward_result:
            if ("Entailment" in x) or ("entailment" in x):
                judgement.append("entailment")
                cnt["entailment"] += 1
            elif ("Neutral" in x) or ("neutral" in x):
                judgement.append("neutral")
                cnt["neutral"] += 1
            elif ("Contradiction" in x) or ("contradiction" in x):
                judgement.append("contradiction")
                cnt["contradiction"] += 1
            else:
                judgement.append(random.choice(["contradiction", "neutral", "entailment"]))
    elif args.task == "STS-B":
        judgement = []
        for x in reward_result:
            flt_num = find_first_number(x)
            if flt_num is None:
                judgement.append(random.uniform(0, 5))
            else:
                if float(flt_num) > 5:
                    flt_num = 5.0
                elif float(flt_num) < 0:
                    flt_num = 0.0
                judgement.append(float(flt_num))
                cnt_interval[str(int(float(flt_num)))] += 1
    elif args.task in ["CoLA", "QQP", "MRPC", "WNLI", "SST"]:
        if args.task == "SST":
            judgement = [random.choice([0, 1]) if x is None else (1 if "Positive" in x else (0 if "Negative" in x else random.choice([0, 1]))) for x in reward_result]
        else:
            judgement = []
            for x in reward_result:
                if ("Yes" in x) or ("yes" in x):
                    judgement.append(1)
                    cnt_1["yes"] += 1
                elif ("No" in x) or ("no" in x):
                    judgement.append(0)
                    cnt_1["no"] += 1
                else:
                    judgement.append(random.choice([0, 1]))
                    unmatched_count += 1
            # judgement = [random.choice([0, 1]) if x is None else (1 if "Yes" in x else (0 if "No" in x else random.choice([0, 1]))) for x in reward_result]
    elif args.task == "COPA":
        # print("HERE!", idxes, reward_result)
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            # print(x.lower())
            if "option one" in x.lower() or "one" in x.lower():
                rst["label"] = 0
                cnt_1["no"] += 1
            elif "option two" in x.lower() or "two" in x.lower():
                rst["label"] = 1
                cnt_1["yes"] += 1
            else:
                rst["label"] = random.choice([0, 1])
            judgement.append(rst)
    elif args.task == "WSC":
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            if "Yes" in x or "yes" in x:
                rst["label"] = "True"
                cnt_1["yes"] += 1
            elif "No" in x or "no" in x:
                rst["label"] = "False"
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice(["True", "False"])
            judgement.append(rst)
    elif args.task == "WiC":
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            if "They are the same" in x or "same" in x:
                rst["label"] = "true"
                cnt_1["yes"] += 1
            elif "They are different" in x or "different" in x:
                rst["label"] = "false"
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice(["true", "false"])
            judgement.append(rst)
    elif args.task == "CB":
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            if ("Entailment" in x) or ("entailment" in x):
                rst["label"] = "entailment"
                cnt["entailment"] += 1
            elif ("Neutral" in x) or ("neutral" in x):
                rst["label"] = "neutral"
                cnt["neutral"] += 1
            elif ("Contradiction" in x) or ("contradiction" in x):
                rst["label"] = "contradiction"
                cnt["contradiction"] += 1
            else:
                rst["label"] = random.choice(["entailment", "neutral", "contradiction"])
            judgement.append(rst)
    elif args.task == "RTE" and project == "SuperGLUE":
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            if ("Yes" in x) or ("yes" in x):
                rst["label"] = "entailment"
                cnt_1["yes"] += 1
            elif ("No" in x) or ("no" in x):
                rst["label"] = "not_entailment"
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice(["entailment", "not_entailment"])
            judgement.append(rst)
    elif args.task == "BoolQ":
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            if ("Yes" in x) or ("yes" in x):
                rst["label"] = "true"
                cnt_1["yes"] += 1
            elif ("No" in x) or ("no" in x):
                rst["label"] = "false"
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice(["true", "false"])
            judgement.append(rst)
    elif args.task == "AX-b" or args.task == "AX-g":
        for id, x in zip(idxes, reward_result):
            rst = {'idx': id, "label": None}
            if ("Yes" in x) or ("yes" in x):
                rst["label"] = "entailment"
                cnt_1["yes"] += 1
            elif ("No" in x) or ("no" in x):
                rst["label"] = "not_entailment"
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice(["entailment", "not_entailment"])
            judgement.append(rst)
    elif args.task == "MultiRC":
        for id, q_id, a_id, x in zip(idxes, question_idxes, answer_idxes, reward_result):
            rst = {"idx": id, "question_idx": q_id, "answer_idx": a_id, "label": None}
            if ("True" in x) or ("true" in x):
                rst["label"] = 1
                cnt_1["yes"] += 1
            elif ("False" in x) or ("false" in x):
                rst["label"] = 0
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice([0, 1])
            judgement.append(rst)
    elif args.task == "ReCoRD":
        for id, q_id, a_id, candidate, x in zip(idxes, question_idxes, answer_idxes, candidates, reward_result):
            rst = {"idx": id, "question_idx": q_id, "answer_idx": a_id, "label": None}
            if ("Yes" in x) or ("yes" in x):
                rst["label"] = candidate
                cnt_1["yes"] += 1
            elif ("No" in x) or ("no" in x):
                rst["label"] = None
                cnt_1["no"] += 1
            else:
                rst["label"] = random.choice([0, 1])
            judgement.append(rst)

    # 统计正确率
    total += len(judgement)
    y_pred += judgement
    print("--finish one batch--")
    
print(f"Overall: {len(y_pred)} / {total}")
print(cnt)
print(cnt_1)
print(cnt_interval)
print(unmatched_count)
headers = ['index', 'prediction']

if "mpt" in model_name:
    abv = "mpt"
elif "llama" in model_name:
    abv = "llama"
elif "qwen2" in model_name.lower():
    abv = "qwen2"
else:
    abv = "unknown"
if project == "GLUE":
    if not from_local:
        if "mismatched" in args.data_path:
            save_file = f'./test_rst/GLUE/{abv}/baseline/{args.task}-mm.tsv'
        elif "matched" in args.data_path:
            save_file = f'./test_rst/GLUE/{abv}/baseline/{args.task}-m.tsv'
        else:
            save_file = f'./test_rst/GLUE/{abv}/dpo/{args.task}-zeroshot.tsv'
    else:
        if "mismatched" in args.data_path:
            save_file = f'./test_rst/GLUE/{abv}/dpo/{args.task}-mm.tsv'
        elif "matched" in args.data_path:
            save_file = f'./test_rst/GLUE/{abv}/dpo/{args.task}-m.tsv'
        else:
            save_file = f'./test_rst/GLUE/{abv}/dpo/{args.task}-sft.tsv'
    with open(save_file, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(headers)
        for index, value in enumerate(y_pred):
            writer.writerow([index, value])
elif project == "SuperGLUE":
    if not from_local:
        save_file = f'./test_rst/SuperGLUE/{abv}/baseline/{args.task}.jsonl'
    else:
        save_file = f'./test_rst/SuperGLUE/{abv}/single_ppo/{args.task}.jsonl'
    with open(save_file, 'w') as f:
        for item in y_pred:
            json.dump(item, f)
            f.write('\n')
print(f"Result saved to {save_file}")
