"""
模型训练代码
使用module.py中定义的reward对llama生成的结果判分
使用trl中的PPO方法对llama参数进行更新
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, Adafactor, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
# from accelerate import Accelerator

import os
import re
from prompt import *
from utils import *
import json
import time
import random
import argparse

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
parser.add_argument('--data_path', type=str, default="../data/cola_public/raw/in_domain_train.tsv")
parser.add_argument("--task", type=str, default="CoLA")
parser.add_argument('--load_in_8bit', type=str, choices=["True", "False"], default="False")
parser.add_argument('--save_methods', type=str, choices=["step", "epoch"], default="epoch")
parser.add_argument("--save_freq", type=int, default=1500)
parser.add_argument('--from_local', type=str, choices=["True", "False"], default="False")
parser.add_argument('--use_CoT', type=str, choices=["True", "False"], default="True")
args = parser.parse_args()
import wandb
project = "SuperGLUE" if args.task not in ["CoLA", "MRPC", "QQP", "RTE", "MNLI", "QNLI", "WNLI", "AX", "STS-B", "SST"] else "GLUE"
wandb.init(project=project)

# 分配模型的cuda编号
train_model_device_id=[0, 1]

# 加载数据
def read_tsv(file_path, batch_size=32, randomize=False, task=args.task):
    delimiter = "\t"
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        # tsv_reader = csv.reader(file, delimiter='\t')
        data = []
        if task != "CoLA":
            next(file)  # 跳过列名行

        for row in file:
            row = row.strip().split(delimiter)
            if task == "CoLA":
                if len(row) == 4:
                    code = row[0]
                    label = int(row[1])
                    original_label = row[2]
                    sentence = row[3]
                    data.append((code, label, original_label, sentence))
            elif task == "RTE":
                if len(row) == 4:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    label = target_to_number[row[3]]
                    data.append((index, sentence1, sentence2, label))
            elif task == "QQP":
                if len(row) == 6:
                    index = row[0]
                    qid1 = row[1]
                    qid2 = row[2]
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = int(row[5])
                    data.append((index, sentence1, sentence2, label))
            elif task == "QNLI":
                if len(row) == 4:
                    index = row[0]
                    question = row[1]
                    sentence = row[2]
                    label = target_to_number[row[3]]
                    data.append((index, question, sentence, label))
            elif task == "SST":
                if len(row) == 2:
                    sentence = row[0]
                    label = int(row[1])
                    data.append((sentence, label))
            elif task == "MRPC":
                if len(row) == 5:
                    label = int(row[0])
                    id1, id2 = row[1], row[2]
                    sentence1, sentence2 = row[3], row[4]
                    data.append((label, sentence1, sentence2))
            elif task == "WNLI":
                if len(row) == 4:
                    index = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    label = int(row[3])
                    data.append((index, sentence1, sentence2, label))
            elif task == "MNLI":
                if len(row) == 12:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = target_to_number[row[-1]]
                    data.append((sentence1, sentence2, label))
            elif task == "STS-B":
                if len(row) == 10:
                    sentence1 = row[7]
                    sentence2 = row[8]
                    score = float(row[-1])
                    data.append((sentence1, sentence2, score))
            else:
                print(f"Ignoring invalid row: {row}")
                raise NotImplementedError("Unsupported task")
        
        if randomize:
            random.shuffle(data)

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            yield batch

def read_jsonl(file_path, batch_size, randomize=False, task=args.task):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = json.loads(line.strip())
            if task == "COPA":
                assert len(line) == 6
                choice1, choice2, premise, idx, label, question = line["choice1"], line["choice2"], line["premise"], \
                                                                        line["idx"], int(line["label"]), line["question"]
                data.append((premise, choice1, choice2, question, label))
            elif task == "WSC":
                assert len(line) == 4
                text, target, idx, label = line["text"], line["target"], line["idx"], int(bool(line["label"]))
                span1_idx, span2_idx, span1_text, span2_text = \
                                target["span1_index"], target["span2_index"], target["span1_text"], target["span2_text"]
                
                data.append((text, span1_idx, span2_idx, span1_text, span2_text, label))
            elif task == "WiC":
                assert len(line) == 10
                label, sentence1, sentence2, word = int(bool(line["label"])), line["sentence1"], line["sentence2"], line["word"]
                data.append((label, sentence1, sentence2, word))
            elif task == "CB":
                assert len(line) == 4
                premise, hypothesis, label = line["premise"], line["hypothesis"], target_to_number[line["label"]]
                data.append((premise, hypothesis, label))
            elif task == "BoolQ":
                assert len(line) == 4
                question, passage, label = line["question"], line["passage"], int(bool(line["label"]))
                data.append((question, passage, label))
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
                        data.append((passage, question, answer, label))
            elif task == "ReCoRD":
                entities = line['passage']['entities']
                text = line['passage']['text']
                cleaned_text = text.split('@highlight')[0].strip()
                qas = line['qas']
                for qa in qas:
                    query = qa['query']
                    answers = qa['answers']
                    entity_list, answer_list = [], []
                    for answer in answers:
                        answer_text = answer['text']
                        if answer_text not in answer_list:
                            answer_list.append(answer_text)
                    for entity in entities:
                        start = entity['start']
                        end = entity['end']
                        candidate = text[start:end+1]
                        if candidate not in entity_list:
                            entity_list.append(text[start:end+1])
                    data.append((cleaned_text, query, random.choice(answer_list), answer_list))
                    data.append((cleaned_text, query, random.choice(list(set(entity_list).difference(set(answer_list)))), answer_list))
            else:
                print(f"Ignoring invalid row: {line}")
                raise NotImplementedError("Unsupported task")

        if randomize:
            random.shuffle(data)

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            yield batch
        
       
file_path = args.data_path
sample_num = args.batch_sentence_num
randomize = True

start=args.start_epoch
EPOCH=args.end_epoch
repeat_time=args.repeat_time
lr=args.learning_rate
temperature=args.model_temperature
batch_size=repeat_time*sample_num
gradient_accumulation_steps = 2
mini_batch_size = batch_size // gradient_accumulation_steps
save_method = args.save_methods
load_in_8bit = args.load_in_8bit == "True"
save_freq = args.save_freq
from_local = args.from_local == "True"
use_CoT = args.use_CoT == "True"
overall_step_time = 0
import datetime

# 获取当前时间并格式化为字符串
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 结果存储路径
if start == 0:
    save_root = f"./model/{args.task}/qwen2-{sample_num}-{lr}-{current_time}"
else:
    save_root = f"./model/{args.task}/qwen2-{sample_num}-{lr}-start{start}-{current_time}"
# 被训练模型加载
model_name=args.train_model_name
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="all", #"none", #"lora_only", "all"
    task_type="CAUSAL_LM",
    inference_mode=False,
    use_rslora=True
)
if load_in_8bit:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        use_auth_token=True,
        device_map="auto",
        quantization_config=bnb_config,
        peft_config=lora_config,
    )
elif from_local:
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
        torch_dtype=torch.bfloat16,
        peft_config=lora_config,
        use_cache=True
    )
    # model = model.bfloat16().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

total_iter = count_lines(file_path) // sample_num
# 初始化PPOTrainer
ppo_config = {
    "batch_size": batch_size,
    "learning_rate": lr,
    "mini_batch_size": mini_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "log_with": "wandb",
    "model_name": model_name,
    "query_dataset": args.task,
    "optimize_cuda_cache": True,
    "init_kl_coef": 0.2,
    "adap_kl_ctrl": True,
    # "kl_target": 0.1,
}
config = PPOConfig(**ppo_config)
print(f"PPOTrainer config: {ppo_config}")
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=config.learning_rate,
)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1) ##TODO: Modify this
# lr_scheduler = CosineAnnealingWarmupLR(optimizer, T_max=total_iter * (EPOCH - start), eta_min=lr / 10, warmup_steps=total_iter * (EPOCH - start) // 10) ##TODO: Modify this
ppo_trainer = PPOTrainer(config, model, None, tokenizer, optimizer=optimizer, lr_scheduler=lr_scheduler)

# 记录训练过程
iter_all_reward=[]
mean_scores=[]
loss_total=[]
policy_entropy=[]
policy_approxkl=[]
policy_clipfrac=[]
global_step = 0

# 训练部分
for epoch in range(start, EPOCH):
    if project == "GLUE":
        data_generator = read_tsv(file_path, batch_size=sample_num, randomize=randomize)
    elif project == "SuperGLUE":
        data_generator = read_jsonl(file_path, batch_size=sample_num, randomize=randomize)
    
    iter_mean_scores=[]
    iter_loss_total=[]
    iter_policy_entropy=[]
    iter_policy_approxkl=[]
    iter_policy_clipfrac=[]
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    } ## TODO: Modify this
    output_length_sampler = LengthSampler(12, 32)

    # 生成一个batch的query
    for iteration, batch in enumerate(data_generator):
        print(f"epoch {epoch}, iter {iteration} / {total_iter}")
        query_tensor=[]
        ground_truth=[]
        candidate_list = []
        # 处理每个批次的数据
        overall_start_time = time.time()
        start_time = time.time()
        record_batch = {"query":[], "response":[]}
        len_count = 0
        for data_row in batch:
            # 读取数据
            if args.task == "CoLA":
                code, label, original_label, sentence = data_row
                prompt = prompt_tmp["cola"].format(sentence=sentence)
            elif args.task == "RTE":
                idx, sentence1, sentence2, label = data_row
                prompt = prompt_tmp["rte"].format(sentence1=sentence1, sentence2=sentence2)
            elif args.task == "QQP":
                idx, sentence1, sentence2, label = data_row
                prompt = prompt_tmp["qqp"].format(sentence1=sentence1, sentence2=sentence2)
            elif args.task == "QNLI":
                idx, question, sentence, label = data_row
                prompt = prompt_tmp["qnli"].format(question=question, sentence=sentence)
            elif args.task == "SST":
                sentence, label = data_row
                prompt = prompt_tmp["sst"].format(sentence=sentence)
            elif args.task == "MRPC":
                label, sentence1, sentence2 = data_row
                prompt = prompt_tmp["mrpc"].format(sentence1=sentence1, sentence2=sentence2)
            elif args.task == "WNLI":
                index, sentence1, sentence2, label = data_row
                prompt = prompt_tmp["wnli"].format(sentence1=sentence1, sentence2=sentence2)
            elif args.task == "MNLI":
                sentence1, sentence2, label = data_row
                prompt = prompt_tmp["mnli"].format(sentence1=sentence1, sentence2=sentence2)
            elif args.task == "STS-B":
                sentence1, sentence2, score = data_row
                prompt = prompt_tmp["sts-b"].format(sentence1=sentence1, sentence2=sentence2)
            elif args.task == "COPA":
                premise, choice1, choice2, question, label = data_row
                prompt = prompt_tmp["copa"].format(premise=premise, choice1=choice1, choice2=choice2, question=question)
            elif args.task == "WSC":
                text, span1_idx, span2_idx, span1_text, span2_text, label = data_row
                prompt = prompt_tmp["wsc"].format(text=text, span1_idx=span1_idx, span2_idx=span2_idx, span1_text=span1_text, span2_text=span2_text)
            elif args.task == "WiC":
                label, sentence1, sentence2, word = data_row
                prompt = prompt_tmp["wic"].format(word=word, sentence1=sentence1, sentence2=sentence2)
            elif args.task == "CB":
                premise, hypothesis, label = data_row
                prompt = prompt_tmp["mnli"].format(sentence1=premise, sentence2=hypothesis)
            elif args.task == "BoolQ":
                question, passage, label = data_row
                prompt = prompt_tmp["boolq"].format(question=question, passage=passage)
            elif args.task == "MultiRC":
                passage, question, answer, label = data_row
                prompt = prompt_tmp["multirc"].format(passage=passage, question=question, answer=answer)
            elif args.task == "ReCoRD":
                text, query, candidate, answer_list = data_row
                prompt = prompt_tmp["record"].format(text=text, query=query.replace("@placeholder", candidate))
            else:
                raise NotImplementedError("Unsupported task")
            query_tensor_nlp = tokenizer.encode(prompt, return_tensors="pt")[0].to(f'cuda:{train_model_device_id[0]}')
            if len(query_tensor_nlp) > 650 and len_count > 1200:
                continue
            query_tensor_nlp = [query_tensor_nlp]
            query_tensor += repeat_time*query_tensor_nlp
            if args.task == "STS-B":
                ground_truth += repeat_time*[float(score)]
            elif args.task == "ReCoRD":
                ground_truth += [answer_list]
                candidate_list += [candidate]
            else:
                ground_truth += repeat_time*[label if label != 0 else -1]
            record_batch["query"].append(prompt)
            len_count += len(query_tensor_nlp)
        if len(query_tensor) == 0:
            continue
        
        # 模型输出，产生criteria
        end_time = time.time()
        print("数据准备时间：", end_time-start_time, "秒")
        start_time = time.time()
        response_tensor_nlp = ppo_trainer.generate(query_tensor, return_prompt=False, length_sampler=output_length_sampler, batch_size=4, **generation_kwargs) # length_sampler=output_length_sampler,
        result = tokenizer.batch_decode(response_tensor_nlp, skip_special_tokens=True)
        record_batch["response"] = result
        end_time = time.time()
        print("生成时间：", end_time-start_time, "秒")
        start_time = time.time()
        # 计算reward前只取出最终判断，不关注分析过程
        if use_CoT:
            analysis_pattern = r"(.*?)</Analysis>"
            analysis_result = [re.search(analysis_pattern, sentence, re.DOTALL).group(1).strip() if re.search(analysis_pattern, sentence, re.DOTALL) else None for sentence in result]
            judgement_pattern = r"<Judgement>(.*?)</Judgement>"
            result = [re.search(judgement_pattern, sentence, re.DOTALL).group(1).strip() if re.search(judgement_pattern, sentence, re.DOTALL) else None for sentence in result]
        else:
            pattern = r"(.*?)</Judgement>"
            # print(result)
            result = [re.search(pattern, sentence, re.DOTALL).group(1).strip() if re.search(pattern, sentence, re.DOTALL) else None for sentence in result]
            analysis_result = [1] * len(result)
            # print(result)
        # 计算reward
        if args.task != "MNLI" and args.task != "STS-B" and args.task != "CB" and args.task != "ReCoRD":
            if args.task == "SST":
                judgement = [0 if x is None or y is None \
                    else (1 if "Positive" in x else (-1 if "Negative" in x else 0)) for x, y in zip(result, analysis_result)]
            elif args.task == "COPA":
                judgement = [0 if x is None or ("Option two" in x and "Option one" in x) or y is None \
                    else (1 if "Option two" in x else (-1 if "Option one" in x else 0)) for x, y in zip(result, analysis_result)]
            elif args.task == "WiC":
                judgement = [0 if x is None or ("Same" in x and "Different" in x) or y is None \
                    else (1 if "Same" in x else (-1 if "Different" in x else 0)) for x, y in zip(result, analysis_result)]
            elif args.task == "MultiRC":
                judgement = [0 if x is None or ("True" in x and "False" in x) or y is None \
                    else (1 if "True" in x else (-1 if "False" in x else 0)) for x, y in zip(result, analysis_result)]
            else:
                judgement = [0 if x is None or ("Yes" in x and "No" in x) or y is None \
                    else (1 if "Yes" in x else (-1 if "No" in x else 0)) for x, y in zip(result, analysis_result)]
            reward=[torch.tensor(-3).type(torch.float16) if x == 0 \
                else torch.tensor(x * y).type(torch.float16) for x, y in zip(judgement, ground_truth)]
            # print(result, analysis_result, judgement, ground_truth, reward)
        elif args.task == "MNLI" or args.task == "CB":
            judgement = []
            for x in result:
                if x is None:
                    judgement.append(-1)
                else:
                    has_entailment = "Entailment" in x or "entailment" in x
                    has_neutral = "Neutral" in x or "neutral" in x
                    has_contradiction = "Contradiction" in x or "contradiction" in x
                    count = sum([has_entailment, has_neutral, has_contradiction])
                    # print("!", has_entailment, has_neutral, has_contradiction, count)
                    if count >= 2:
                        judgement.append(-1)
                    elif has_entailment:
                        judgement.append(target_to_number["entailment"])
                    elif has_neutral:
                        judgement.append(target_to_number["neutral"])
                    elif has_contradiction:
                        judgement.append(target_to_number["contradiction"])
                    else:
                        judgement.append(-1)
            reward = []
            for x, y in zip(judgement, ground_truth):
                if x == y:
                    reward.append(torch.tensor(1).type(torch.float16))
                elif x == -1:
                    reward.append(torch.tensor(-3).type(torch.float16))
                else:
                    reward.append(torch.tensor(-1).type(torch.float16))
        elif args.task == "STS-B":
            judgement = []
            for x in result:
                flt_num = find_first_number(x)
                if flt_num is None:
                    judgement.append(None)
                else:
                    judgement.append(float(flt_num))
            reward = []
            for x, y in zip(judgement, ground_truth):
                if x is None:
                    reward.append(torch.tensor(-2.5).type(torch.float16))
                else:
                    reward.append(torch.tensor(2.5 - abs(x - y)).type(torch.float16))
        elif args.task == "ReCoRD":
            reward = []
            for x, y, candidate in zip(result, ground_truth, candidate_list):
                if x is None:
                    reward.append(torch.tensor(-3).type(torch.float16))
                elif "Yes" in x or "yes" in x:
                    if candidate in y:
                        reward.append(torch.tensor(1).type(torch.float16))
                    else:
                        reward.append(torch.tensor(-1).type(torch.float16))
                elif "No" in x or "no" in x:
                    if candidate in y:
                        reward.append(torch.tensor(-1).type(torch.float16))
                    else:
                        reward.append(torch.tensor(1).type(torch.float16))
                else:
                    reward.append(torch.tensor(-1).type(torch.float16))
                # print(x, candidate, y, reward)
        # elif args.task == "ReCoRD":
        #     reward = []
        #     for pred, gts in zip(result, ground_truth):
        #         if pred is None:
        #             reward.append(torch.tensor(-3).type(torch.float16))
        #             continue
        #         add_flag = False
        #         for gt in gts:
        #             if pred.lower() in gt.lower():
        #                 reward.append(torch.tensor(1).type(torch.float16))
        #                 add_flag = True
        #                 break
        #         if not add_flag:
        #             reward.append(torch.tensor(-1).type(torch.float16))
                    
        # 更新参数
        ppo_trainer.current_device = torch.device(f'cuda:{train_model_device_id[0]}')
        # 配置实际的batchsize
        ppo_trainer.config.batch_size=len(query_tensor)
        end_time = time.time()
        print("reward生成时间: ", end_time-start_time, "秒")
        start_time = time.time()
        train_stats = ppo_trainer.step(query_tensor, response_tensor_nlp, reward)
        ppo_trainer.log_stats(train_stats, record_batch, reward)
        end_time = time.time()
        print("训练时间：", end_time-start_time, "秒")
        overall_end_time = time.time()
        print("整体运行时间：", overall_end_time-overall_start_time, "秒")
        overall_step_time += overall_end_time - overall_start_time
        # 保存训练中间结果
        print('mean_scores', train_stats['ppo/mean_scores'], 'loss_total', train_stats['ppo/loss/total'], 'policy_entropy', train_stats['ppo/policy/entropy'], 'policy_approxkl', train_stats['ppo/policy/approxkl'], 'policy_clipfrac', train_stats['ppo/policy/clipfrac'])

        iter_mean_scores.append(train_stats['ppo/mean_scores'])
        iter_loss_total.append(train_stats['ppo/loss/total'])
        iter_policy_entropy.append(train_stats['ppo/policy/entropy'])
        iter_policy_approxkl.append(train_stats['ppo/policy/approxkl'])
        iter_policy_clipfrac.append(train_stats['ppo/policy/clipfrac'])
        iter_all_reward+=[x.item() for x in reward]

        if save_method == "step":
            mean_scores.append(sum(iter_mean_scores)/len(iter_mean_scores))
            loss_total.append(sum(iter_loss_total)/len(iter_loss_total))
            policy_entropy.append(sum(iter_policy_entropy)/len(iter_policy_entropy))
            policy_approxkl.append(sum(iter_policy_approxkl)/len(iter_policy_approxkl))
            policy_clipfrac.append(sum(iter_policy_clipfrac)/len(iter_policy_clipfrac))
            # 保存训练中间结果和模型参数
            if (global_step+1)%save_freq==0:
                # 创建保存目录
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                if not os.path.exists(f"{save_root}/{global_step}"):
                    os.makedirs(f"{save_root}/{global_step}")
                # 保存模型的checkpoint
                if  (global_step+1)%save_freq==0:
                    print("save checkpoint")
                    try:
                        ppo_trainer.save_pretrained(f"{save_root}/{global_step}")
                    except:
                        print("save err")
                    torch.save(optimizer.state_dict(), f"{save_root}/{global_step}/optimizer.pth")
                # 保存训练中间结果
                try:
                    iter_train_info= {
                        'mean_scores': mean_scores, 'loss_total': loss_total, 'policy_entropy': policy_entropy, 'policy_approxkl': policy_approxkl, 'policy_clipfrac': policy_clipfrac, 'iter_all_reward': iter_all_reward,
                        'iter_mean_scores': iter_mean_scores, 'iter_loss_total': iter_loss_total, 'iter_policy_entropy': iter_policy_entropy, 'iter_policy_approxkl': iter_policy_approxkl, 'iter_policy_clipfrac': iter_policy_clipfrac,
                    }
                    with open(f'{save_root}/iter_train_info-batch-{iteration}.json', 'w') as f:
                        json.dump(iter_train_info, f)
                except:
                    pass
        global_step += 1
    
    if save_method == "epoch":
        mean_scores.append(sum(iter_mean_scores)/len(iter_mean_scores))
        loss_total.append(sum(iter_loss_total)/len(iter_loss_total))
        policy_entropy.append(sum(iter_policy_entropy)/len(iter_policy_entropy))
        policy_approxkl.append(sum(iter_policy_approxkl)/len(iter_policy_approxkl))
        policy_clipfrac.append(sum(iter_policy_clipfrac)/len(iter_policy_clipfrac))
        # 保存训练中间结果和模型参数
        if (epoch+1)%1==0:
            # 创建保存目录
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            if not os.path.exists(f"{save_root}/{epoch}"):
                os.makedirs(f"{save_root}/{epoch}")
            # 保存模型的checkpoint
            if (epoch+1)%args.checkpoint_rate==0:
                print("save checkpoint")
                try:
                    ppo_trainer.save_pretrained(f"{save_root}/{epoch}")
                except:
                    print("save err")
                torch.save(optimizer.state_dict(), f"{save_root}/{epoch}/optimizer.pth")
            # 保存训练中间结果
            try:
                iter_train_info= {
                    'mean_scores': mean_scores, 'loss_total': loss_total, 'policy_entropy': policy_entropy, 'policy_approxkl': policy_approxkl, 'policy_clipfrac': policy_clipfrac, 'iter_all_reward': iter_all_reward,
                    'iter_mean_scores': iter_mean_scores, 'iter_loss_total': iter_loss_total, 'iter_policy_entropy': iter_policy_entropy, 'iter_policy_approxkl': iter_policy_approxkl, 'iter_policy_clipfrac': iter_policy_clipfrac,
                }
                with open(f'{save_root}/iter_train_info-batch-{epoch}.json', 'w') as f:
                    json.dump(iter_train_info, f)
            except:
                pass
print(f"Average step time: {overall_step_time / global_step} seconds.")
