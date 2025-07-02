"""
这是从huggingface上load预训练模型的模块
主要可以选择：
    本地 or 线上(cache)
    是否使用lora adapter
    加载精度,8bit, 16bit, 32bit等

"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model,AutoModelForSeq2SeqLMWithValueHead
from transformers import BitsAndBytesConfig
import os
from peft import PeftModel
from peft import LoraConfig, get_peft_model


# 从huggingface上或本地load预训练模型的模块
def load_model(model_name="daryl149/llama-2-7b-chat-hf",lora_path="/tutorial/model/llama32/31" , need_ref=False, from_local=False, whole_model=False, need_lora=True,
               load_in_8bit=False, fast=False):
    
    # lora adapter的参数选择
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
        # 使用8bit加载
        if fast:
            model = LLM(model="NousResearch/Llama-2-70b-chat-hf", tensor_parallel_size=2)
            return model, None, None
        else:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                device_map='cuda:0',
                load_in_8bit=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            return model, None, tokenizer
    elif from_local:
        # 从本地加载
        print("from_local: ", model_name)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name
        )
    elif whole_model:
        # 32bit全参数加载与训练
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            use_auth_token=True,
        )
    elif need_lora:
        # 32bit全参数但使用lora训练
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            use_auth_token=True,
            peft_config=lora_config,
        )
    else:
        # 16bit全参数加载
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    # 加载模型附带的tokenizer
    if from_local:
        # 本地
        tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, use_fast=False)
    else:
        # 远程
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
    
    # 需要PPO中的reference model
    if need_ref:
        model_ref = create_reference_model(model)
    
    if need_ref:
        return model, model_ref, tokenizer
    else:
        return model, None, tokenizer
    
    
    # 用来load quantization的模型的代码
    # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    # device_map = {
    # "transformer.word_embeddings": 0,
    # "transformer.word_embeddings_layernorm": 0,
    # "lm_head": "cpu",
    # "transformer.h": 0,
    # "transformer.ln_f": 0,
    # }
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, cache_dir=model_path, device_map='auto', quantization_config=quantization_config)
    
    
# 运行将保存示例模型到本地
if __name__ == '__main__':
    model_name="gpt2"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name
    )
    folder_name="/tutorial/llama/gpt2"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model.save_pretrained(folder_name)
    print("model saved")