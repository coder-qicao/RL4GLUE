import re
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from trl import SFTTrainer
from scipy.stats import pearsonr, spearmanr
import math
import torch
from tqdm import tqdm
from prompt import *
import numpy as np
import torch
import random
from rouge_score import rouge_scorer
from sentence_transformers import util

def compute_rouge_f1(prediction: str, reference: str):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    rouge_l = scores['rougeL'].fmeasure
    
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())
    common_tokens = pred_tokens & ref_tokens
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return (rouge_l + f1) / 2

def compute_coverage(prediction: str, reference: str, model):
    pred_embedding = model.encode(prediction, convert_to_tensor=True)
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
    
    tau_c = 0.7
    return similarity - max(0, tau_c - similarity)

def compute_brevity(prediction: str, reference: str):
    pred_len = len(prediction.split())
    ref_len = len(reference.split())
    
    if pred_len > ref_len:
        return max(0, 1 - (pred_len - ref_len) / ref_len)
    else:
        return max(0, pred_len / ref_len)

def compute_final_reward(prediction: str, reference: str, model, alpha=0.5, beta=0.25, gamma=0.25):
    accuracy = compute_rouge_f1(prediction, reference)
    coverage = compute_coverage(prediction, reference, model)
    brevity = compute_brevity(prediction, reference)
    
    final_reward = alpha * accuracy + beta * coverage + gamma * brevity
    return final_reward

import re
import string
from collections import Counter

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s

def exact_match(prediction, ground_truth):
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_squad(predictions, ground_truths):
    total_em, total_f1 = 0, 0
    for pred, gt in zip(predictions, ground_truths):
        total_em += exact_match(pred, gt)
        total_f1 += f1_score(pred, gt)

    num_samples = len(predictions)
    return {
        "Exact Match": round((total_em / num_samples) * 100, 2),
        "F1 Score": round((total_f1 / num_samples) * 100, 2)
    }

class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.step_count = 0
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # Warmup phase
            warmup_factor = self.step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.step_count - self.warmup_steps) / (self.T_max - self.warmup_steps)
            cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cos_factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None:
            self.step_count = epoch
        else:
            self.step_count += 1
        super(CosineAnnealingWarmupLR, self).step()

def find_first_number(text):
    # 定义正则表达式匹配浮点数和整数
    if text is None:
        return None
    
    # 匹配浮点数
    float_pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"
    # 匹配整数
    int_pattern = r"[-+]?\b\d+\b"
    
    # 先匹配浮点数
    float_match = re.search(float_pattern, text)
    
    # 如果找到浮点数，返回浮点数
    if float_match:
        return float_match.group(0)
    
    # 如果没有找到浮点数，再匹配整数
    int_match = re.search(int_pattern, text)
    
    if int_match:
        return int_match.group(0)
    
    return None

def calculate_metric(judgement, ground_truth, metrics):
    if metrics != ["Exact Match", "F1 Score"]:
        scores = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for x, y in zip(judgement, ground_truth):
            if x == 1 and y == 1:
                scores["TP"] += 1
            elif x == 1 and y == -1:
                scores["FP"] += 1
            elif x == -1 and y == 1:
                scores["FN"] += 1
            elif x == -1 and y == -1:
                scores["TN"] += 1
            if metrics == "spearmanr" or metrics == "pearsonr":
                if not 0 <= x <= 5:
                    print(f"Warning: judgement label should be in range [0, 5], but got {x}.")
        TP, FP, FN, TN = scores["TP"], scores["FP"], scores["FN"], scores["TN"]
        scores["precision"] = TP / (TP + FP) if TP + FP > 0 else 0
        scores["recall"] = TP / (TP + FN) if TP + FN > 0 else 0
        for metric in metrics:
            if metric == "acc":
                scores[metric] = accuracy_score(judgement, ground_truth)
            elif metric == "mcc":
                scores[metric] = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            elif metric == "f1":
                # scores[metric] = f1_score(judgement, ground_truth, average="macro")
                scores[metric] = 2 * (scores["precision"] * scores["recall"]) / (scores["precision"] + scores["recall"]) if scores["precision"] + scores["recall"] > 0 else 0
            elif metric == "pearsonr":
                pearsonr_corr, _ = pearsonr(judgement, ground_truth)
                scores[metric] = pearsonr_corr
            elif metric == "spearmanr":
                spearmanr_corr, _ = spearmanr(judgement, ground_truth)
                scores[metric] = spearmanr_corr
    else:
        scores = evaluate_squad(judgement, ground_truth)
    return scores

def bootstrap_metrics(judgement, ground_truth, metrics, n_bootstrap = 1000, confidence_level = 0.95):
    n = len(judgement)
    if metrics == "EM+F1":
        metrics = ["Exact Match", "F1 Score"]
    bootstrap_results = {m: [] for m in metrics}

    for _ in range(n_bootstrap):
        indices = [random.randint(0, n-1) for _ in range(n)]
        sampled_judgement = [judgement[i] for i in indices]
        sampled_truth = [ground_truth[i] for i in indices]

        scores = calculate_metric(sampled_judgement, sampled_truth, metrics)

        for m in metrics:
            bootstrap_results[m].append(scores[m])

    final_summary = {}
    alpha = 1 - confidence_level
    lower_percentile = alpha/2 * 100
    upper_percentile = (1 - alpha/2) * 100

    for m in metrics:
        arr = np.array(bootstrap_results[m])
        mean_val = float(np.mean(arr))
        lower = float(np.percentile(arr, lower_percentile))
        upper = float(np.percentile(arr, upper_percentile))
        final_summary[m] = {
            "mean": mean_val,
            f"{int(confidence_level*100)}%_CI": (lower, upper)
        }

    return final_summary

def extract_evaluation_scores(text):
    evaluation_match = re.search(r"<Evaluation>(.*?)</Evaluation>", text, re.DOTALL)
    
    if not evaluation_match:
        return None
    
    evaluation_content = evaluation_match.group(1).strip() 

    scores = {}
    matches = re.findall(r"Response\s*(\d+):\s*([1-5])", evaluation_content)

    for response_id, score in matches:
        scores[int(response_id)] = int(score)

    return scores

def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file)
    
def prepare_sample_text(example, tokenizer):
    """Prepare the text from a sample of the dataset based on the task and append the EOS token."""
    task = example['task']
    
    if task == "COPA":
        text = sft_prompts["copa"].format(premise=example['premise'], question=example['question'], 
                                          choice1=example['choice1'], choice2=example['choice2'], label=example['label'])
    elif task == "WSC":
        text = sft_prompts["wsc"].format(text=example['text'], span1_text=example['span1_text'], 
                                         span2_text=example['span2_text'], label=example['label'])
    elif task == "WiC":
        text = sft_prompts["wic"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], 
                                         word=example['word'], label=example['label'])
    elif task == "CB":
        text = sft_prompts["mnli"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    elif task == "BoolQ":
        text = sft_prompts["boolq"].format(passage=example['passage'], question=example['question'], label=example['label'])
    elif task == "MultiRC":
        text = sft_prompts["multirc"].format(passage=example['passage'], question=example['question'], 
                                             answer=example['answer'], label=example['label'])
    elif task == "ReCoRD":
        text = sft_prompts["record"].format(text=example['text'], query=example['query'], label=example['label'])
    elif task == "CoLA":
        text = sft_prompts["cola"].format(sentence=example['sentence'], label=example['label'])
    elif task == "RTE":
        text = sft_prompts["rte"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    elif task == "QQP":
        text = sft_prompts["qqp"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    elif task == "QNLI":
        text = sft_prompts["qnli"].format(question=example['question'], sentence=example['sentence'], label=example['label'])
    elif task == "SST":
        text = sft_prompts["sst"].format(sentence=example['sentence'], label=example['label'])
    elif task == "MRPC":
        text = sft_prompts["mrpc"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    elif task == "WNLI":
        text = sft_prompts["wnli"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    elif task == "MNLI":
        text = sft_prompts["mnli"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    elif task == "STS-B":
        text = sft_prompts["sts-b"].format(sentence1=example['sentence1'], sentence2=example['sentence2'], label=example['label'])
    else:
        raise NotImplementedError(f"Unsupported task: {task}")

    # 在生成的文本末尾添加 EOS token
    text += tokenizer.eos_token
    return text

def construct_fewshot_sample(dataset):
    if isinstance(dataset[0], dict):
        task = dataset[0]["task"]
    else:
        task = dataset[0][-1]
    few_shot_prompt = ""
    system_prompt_pattern = r"^.*?(?=Prompt:)"
    user_prompt_pattern = r"Prompt:.*"
    if task == "CB":
        prompt = prompt_tmp["mnli"]
    else:
        prompt = prompt_tmp[task.lower()]
    system_prompt_match = re.search(system_prompt_pattern, prompt, re.DOTALL)
    user_prompt_match = re.search(user_prompt_pattern, prompt, re.DOTALL)
    system_prompt = system_prompt_match.group(0) if system_prompt_match else ""
    user_prompt = user_prompt_match.group(0) if user_prompt_match else ""

    if task == "CoLA" or task == "SST":
        for data in dataset:
            example = user_prompt.format(sentence=data["sentence"])
            example += f" {data['label']} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task in ["RTE", "QQP", "WNLI", "MNLI", "MRPC", "STS-B", "AX"]:
        for data in dataset:
            example = user_prompt.format(sentence1=data["sentence1"], sentence2=data["sentence2"])
            example += f" {data['label']} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "QNLI":
        for data in dataset:
            example = user_prompt.format(question=data["question"], sentence=data["sentence"])
            example += f" {data['label']} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "COPA":
        for data in dataset:
            label = "Option One" if data[4] == 0 else "Option Two"
            example = user_prompt.format(premise=data[0], choice1=data[1], choice2=data[2], question=data[3])
            example += f" {label} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "WiC":
        for data in dataset:
            label = "Same" if data[0] else "Different"
            example = user_prompt.format(word=data[3], sentence1=data[1], sentence2=data[2])
            example += f" {label} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "WSC":
        for data in dataset:
            label = "Yes" if data[5] else "No"
            example = user_prompt.format(text=data[0], span1_idx=data[1], span2_idx=data[2], span1_text=data[3], span2_text=data[4])
            example += f" {label} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "CB":
        for data in dataset:
            example = user_prompt.format(sentence1=data[0], sentence2=data[1])
            example += f" {data[2]} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "BoolQ":
        for data in dataset:
            label = "Yes" if data[2] else "No"
            example = user_prompt.format(question=data[0], passage=data[1])
            example += f" {label} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "MultiRC":
        for data in dataset:
            label = "True" if data[3] == 1 else "False"
            example = user_prompt.format(passage=data[0], question=data[1], answer=data[2])
            example += f" {label} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    elif task == "ReCoRD":
        for data in dataset:
            label = "Yes" if data[3] else "No"
            example = user_prompt.format(text=data[0], query=data[1].replace("@placeholder", data[2]))
            example += f" {label} </Judgement>"
            few_shot_prompt += "[ " + example + " ]\n"
    final_prompt = system_prompt + "\nHere are some examples:\n" + few_shot_prompt + "\n" + user_prompt
    return final_prompt

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, tokenizer)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

class CustomSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs):
        # 打印当前批次的输入数据（原始文本）
        input_ids = inputs['input_ids'].to(model.device)
        print(input_ids, input_ids.shape)
        decoded_input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        print(f"Decoded input: {decoded_input}")
        
        # 前向传播和生成
        outputs = model.generate(input_ids, max_length=50)

        # 解码生成的输出
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"Generated output: {decoded_outputs}")

        # 调用父类的方法继续进行训练步骤
        return super().training_step(model, inputs)