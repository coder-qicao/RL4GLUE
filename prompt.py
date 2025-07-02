ALL_TASKS = ['sst', 'cola', 'mrpc', 'qqp', 'sts-b', 'mnli', 'qnli', 'rte', 'wnli']

pred_targets = {
    'sst': ['positive','negative'], # 1 0 
    'cola': ['Yes','No'], # 1 0
    'mrpc': ['Yes','No'], # 1 0
    'qqp': ['Yes','No'], # 1 0
    'sts-b': ['number on scale of 0 to 5'], #0-5
    'mnli': ['entailment', 'contradiction' , 'neutral'],
    'qnli': ['Yes','No'],
    'rte': ['Yes','No'],
    'wnli': ['Yes','No']
}

label_targets = {
    'sst': [1,0],
    'cola': [1,0],
    'mrpc': [1,0],
    'qqp': [1,0],
    'sts-b': ['number on scale of 0 to 5'],
    'mnli': ['entailment', 'contradiction' , 'neutral'],
    'qnli': ['entailment','not_entailment'],
    'rte': ['entailment','not_entailment'],
    'wnli': [1,0]
}

target_to_number = {
    'entailment':1,
    'contradiction':2,
    'neutral':3,
    'not_entailment':0
}

""" PPO Prompts """
cola_prompt = "System_prompt:\nYou are an assistant to analyze the linguistic properties of a sentence. The task is to decide the linguistic acceptability of a sentence. If the sentence is linguistically correct then it is acceptable, else it is not."
cola_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
cola_prompt += """Prompt:\nNow analyze and judge if the sentence "{sentence}" is linguistically acceptable.\n"""
cola_prompt += "Assistant:\n<Judgement>"

rte_prompt = "System_prompt:\nYou are an assistant to analyze the logical relationship between two texts. The task is to determine whether the sentence entails each other or not.\n"
rte_prompt += """\nThe result you give should strictly follow the format:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
rte_prompt += """Prompt:\nNow analyze and judge if the sentence "{sentence1}" entails the sentence "{sentence2}".\n"""
rte_prompt += "Assistant:\n<Judgement>"

qqp_prompt = "System_prompt:\nYou are an assistant assigned to analyze sentence pairs. The task is to determine whether each pair of sentences in the corpus is paraphrases of each other.\n"
qqp_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
qqp_prompt += """Prompt:\nNow analyze and judge whether they are paraphrases or not for the pair of sentences "{sentence1}" and "{sentence2}".\n"""
qqp_prompt += "Assistant:\n<Judgement>"

qnli_prompt = "System_prompt:\nYou are an assistant to analyze the provided question and answer pair. The task is to determine whether the sentence entails the answer to the given question.\n"
qnli_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
qnli_prompt += """Prompt:\nNow analyze and judge the logical relationship between the premise and hypothesis for question "{question}" and sentence "{sentence}".\n"""
qnli_prompt += "Assistant:\n<Judgement>"

sst_prompt = "System_prompt:\nYou are an assistant to analyze sentiment of a sentence. The task is to decide whether a series of corpus is positive or negative.\n"
sst_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Positive" or "Negative" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
sst_prompt += """Prompt:\nNow analyze and determine the sentiment expressed in the following text "{sentence}".\n"""
sst_prompt += "Assistant:\n<Judgement>"

mrpc_prompt = "System_prompt:\nYou are an assistant to analyze pairs of sentences. The task is to decide whether a pair of corpus are paraphrases or not.\n"
mrpc_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
mrpc_prompt += """Prompt:\nNow analyze and judge whether the sentence "{sentence1}" paraphrase the sentence "{sentence2}".\n"""
mrpc_prompt += "Assistant:\n<Judgement>"

wnli_prompt = "System_prompt:\nYou are an assistant to analyze the logical relationship between the premise and hypothesis. The task is to determine whether the premise entails the hypothesis.\n"
wnli_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
wnli_prompt += """Prompt:\nNow analyze and judge the logical relationship between the premise "{sentence1}" and the hypothesis "{sentence2}".\n"""
wnli_prompt += "Assistant:\n<Judgement>"

mnli_prompt = "System_prompt:\nYou are an assistant to analyze the logical relationship between the premise and the hypothesis. The task is to classify the relationship between the premise and the hypothesis as entailment, contradiction, or neutral.\n"
mnli_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Entailment", "Contradiction", or "Neutral" here.}} </Judgement>\n"""
mnli_prompt += """Prompt:\nNow judge the logical relationship between the premise "{sentence1}" and the hypothesis "{sentence2}".\n"""
mnli_prompt += "Assistant:\n<Judgement>"

stsb_prompt = "System_prompt:\nYou are an assistant to analyze the semantic similarity between pairs of sentences. The task is to provide a similarity score on a scale of 0 to 5.\n"
stsb_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert a float number between 0 to 5 here}} </Judgement>\n<Analysis> {{Insert your short analysis here}} </Analysis>\n"""
stsb_prompt += """Prompt:\nNow analyze and determine the semantic similarity score for the pair of sentences "{sentence1}" and "{sentence2}".\n"""
stsb_prompt += "Assistant:\n<Judgement>"

copa_prompt = "System_prompt:\nYou are an assistant to determine the {question} of a given premise. The task is to evaluate which of the two options is the {question} of the premise.\n"
copa_prompt += """\nThe result you give should have the following form:\n<Judgement> {{ Insert only "Option one" or "Option two" here }} </Judgement>\n"""
copa_prompt += """Prompt:\nNow choose the most likely {question} for the premise "{premise}" from Option one "{choice1}" and Option two "{choice2}".\n"""
copa_prompt += "Assistant:\n<Judgement>"

wsc_prompt = "System_prompt:\nYou are an assistant to resolve coreferences in a given sentence. The task is to determine whether the pronoun refers to the specified noun phrase.\n"
wsc_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n"""
wsc_prompt += """Prompt:\nNow determine in the sentence "{text}", whether the pronoun "{span2_text}" refers to "{span1_text}"?\n"""
wsc_prompt += "Assistant:\n<Judgement>"

wic_prompt = "System_prompt:\nYou are an assistant to analyze the meaning of a word in two given sentences. The task is to determine whether the word has the same meaning or not in the two sentences.\n"
wic_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Same" or "Different" here}} </Judgement>\n"""
wic_prompt += """Prompt:\nNow determine whether the word '{word}' has the same meaning or not in these two sentences '{sentence1}' and '{sentence2}'.\n"""
wic_prompt += "Assistant:\n<Judgement>"

boolq_prompt = "System_prompt:\nYou are an assistant to answer a question based on a given passage with 'yes' or 'no'.\n"
boolq_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n"""
boolq_prompt += """Prompt:\nPassage: "{passage}"\nQuestion: "{question}?"\n"""
boolq_prompt += "Assistant:\n<Judgement>"

multirc_prompt = "System_prompt:\nYour task is to determine if the answer to the question is consistent with the information in the passage.\n"
multirc_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only "True" or "False" here}} </Judgement>\n"""
multirc_prompt += """Prompt:\nPassage: "{passage}"\nQuestion: "{question}"\nAnswer: "{answer}"\n"""
multirc_prompt += "Assistant:\n<Judgement>"

record_prompt = "System_prompt:\nYour task is to assess the query for consistency with the passage, grammatical correctness, and logical correctness. Output 'No' if any of the above are incorrect, and 'Yes' if all are correct.\n"
record_prompt += """\nThe result you provide should have the following form:\n<Judgement> {{Insert only "Yes" or "No" here}} </Judgement>\n"""
record_prompt += """Prompt:\nPassage: "{text}"\nNow assess the Query: "{query}"\n"""
record_prompt += "Assistant:\n<Judgement>"

squad_prompt = "System_prompt:\nYour task is to answer a question using a short excerpt from the given passage. If there is no answer, respond with 'Unanswerable </Judgement>'.\n"
squad_prompt += """\nThe result you provide should follow this format:\n<Judgement> {{Insert your short answer here}} </Judgement>\n""" 
squad_prompt += """Prompt:\nContext: "{context}"\nQuery: "{question}?"\n""" 
squad_prompt += "Assistant:\n<Judgement>"

''' Llama 2 '''
# boolq_prompt = "[INST] <<SYS>>\nYou are an assistant to answer a question based on a given passage with 'yes' or 'no'."
# boolq_prompt += """ The result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n<</SYS>>\n\n"""
# boolq_prompt += """Passage: "{passage}"\nQuestion: "{question}?" [/INST]\n"""
# boolq_prompt += "<Judgement>"

# multirc_prompt = "[INST] <<SYS>>\nYour task is to determine if the answer to the question is consistent with the information in the passage."
# multirc_prompt += """ The result you give should have the following form:\n<Judgement> {{Insert "True" or "False" here}} </Judgement>\n<</SYS>>\n\n"""
# multirc_prompt += """Passage: "{passage}"\nQuestion: "{question}"\nAnswer: "{answer}" [/INST]\n"""
# multirc_prompt += "<Judgement>"

# record_prompt = """[INST] <<SYS>>\nYour task is to assess the query for consistency with the passage, grammatical correctness, and logical correctness. Output 'No' if any of the above are incorrect, and 'Yes' if all are correct."""
# record_prompt += """ The result you provide should be formatted as follows:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n<</SYS>>\n\n"""
# record_prompt += """Passage:\n"{text}"\nNow assess the Query:\n"{query}" [/INST]\n"""
# record_prompt += "<Judgement>"


''' Llama 3 '''
# boolq_prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are an assistant to answer a question based on a given passage with 'yes' or 'no'.\n"
# boolq_prompt += """ The result you give should have the following form:\n<Judgement> Only answer "Yes" or "No" here </Judgement><|eot_id|>\n"""
# boolq_prompt += """<|start_header_id|>user<|end_header_id|>\n\nNow given the passage "{passage}", "{question}?"<|eot_id|>\n"""
# boolq_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n<Judgement>"

# wic_prompt = "<|start_header_id|>system<|end_header_id|>\n\nYour task is to determine whether the word has the same meaning or not in the two sentences."
# wic_prompt += """ The result you give should have the following form:\n<Judgement> Only answer "Same" or "Different" here </Judgement><|eot_id|>\n"""
# wic_prompt += """<|start_header_id|>user<|end_header_id|>\n\nNow determine if the word '{word}' has the same meaning in the sentence '{sentence1}' as it does in the sentence '{sentence2}'.<|eot_id|>\n"""
# wic_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n<Judgement>"

# multirc_prompt = "<|start_header_id|>system<|end_header_id|>\n\nYour task is to determine if the answer to the question is consistent with the information in the passage.\n"
# multirc_prompt += """ The result you give should have the following form:\n<Judgement> Only answer "True" or "False" here </Judgement><|eot_id|>\n"""
# multirc_prompt += """<|start_header_id|>user<|end_header_id|>\n\nPassage: "{passage}"\nQuestion: "{question}"\nAnswer: "{answer}"<|eot_id|>\n"""
# multirc_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n<Judgement>"

# record_prompt = """<|start_header_id|>system<|end_header_id|>\n\nYour task is to assess the query for consistency with the passage, grammatical correctness, and logical correctness. Output 'No' if any of the above are incorrect, and 'Yes' if all are correct."""
# record_prompt += """ The result you provide should be formatted as follows:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement><|eot_id|>\n"""
# record_prompt += """<|start_header_id|>user<|end_header_id|>\n\nPassage:\n"{text}"\nNow assess the Query:\n"{query}"<|eot_id|>\n"""
# record_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n<Judgement>"

prompt_tmp = {
    'sst': sst_prompt,
    'cola': cola_prompt,
    'mrpc':  mrpc_prompt,
    'qqp': qqp_prompt,
    'sts-b':  stsb_prompt,
    'mnli': mnli_prompt,
    'qnli': qnli_prompt, 
    'rte': rte_prompt,
    'wnli': wnli_prompt, 
    "copa": copa_prompt, 
    "wsc": wsc_prompt,
    "wic": wic_prompt,
    "boolq": boolq_prompt,
    "multirc": multirc_prompt,
    "record": record_prompt,
    "squad": squad_prompt
}


""" SFT prompts """
sft_cola_prompt = "System_prompt:\nYou are an assistant to analyze the linguistic properties of a sentence. The task is to decide the linguistic acceptability of a sentence. If the sentence is linguistically correct then it is acceptable, else it is not."
sft_cola_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_cola_prompt += """Prompt:\nNow judge if the sentence "{sentence}" is linguistically acceptable.\n"""
sft_cola_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_rte_prompt = "System_prompt:\nYou are an assitant to analyze the logical relationship between two texts. The task is to determine whether the sentence entails each other or not."
sft_rte_prompt += """\nThe result you give should strictly follow the format:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_rte_prompt += """Prompt:\nJudge if the sentence "{sentence1}" entails the sentence "{sentence2}".\n"""
sft_rte_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_qqp_prompt = "System_prompt:\nYou are an assistant assigned to analyze sentence pairs. Your task is to determine whether each pair of sentences in the corpus is paraphrases of each other.\n"
sft_qqp_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_qqp_prompt += """Prompt:\nNow judge whether they are paraphrases or not for the pair of sentences "{sentence1}" and "{sentence2}".\n"""
sft_qqp_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_qnli_prompt = "System_prompt:\nYou are an assitant to analyze the provided question and answer pair. The task is to determine whether the sentence entails the answer to the given question.\n"
sft_qnli_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_qnli_prompt += """Prompt:\nNow judge the logical relationship between the premise and hypothesis for question "{question}" and sentence "{sentence}"\n"""
sft_qnli_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_sst_prompt = "System_prompt:\nYou are an assitant to analyze sentiment of a sentence. The task is to decide whether a series of corpus is positive or negative.\n"
sft_sst_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Positive" or "Negative" here}} </Judgement>\n"""
sft_sst_prompt += """Prompt:\nNow analyze the sentiment expressed in the following text "{sentence}".\n"""
sft_sst_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_mrpc_prompt = "System_prompt:\nYou are an assitant to analyze pairs of sentences. The task is to decide whether a pair of corpus are paraphrases or not. \n"
sft_mrpc_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_mrpc_prompt += """Prompt:\nNow judge whether the sentence "{sentence1}" paraphrase the sentence "{sentence2}".\n"""
sft_mrpc_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_wnli_prompt = "System_prompt:\nYou are an assitant to analyze the logical relationship between the premise and hypothesis. The task is to determine whether the premise entails the hypothesis.\n"
sft_wnli_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_wnli_prompt += """Prompt:\nNow judge the logical relationship between the premise "{sentence1}" and the hypothesis "{sentence2}".\n"""
sft_wnli_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_mnli_prompt = "System_prompt:\nYou are an assitant to analyze the logical relationship between the premise and the hypothesis. The task is to classify the relationship between the premise and the hypothesis as entailment, contradiction, or neutral.\n"
sft_mnli_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert only one option from "Entailment", "Contradiction", or "Neutral" here.}} </Judgement>\n"""
sft_mnli_prompt += """Prompt:\nNow judge the logical relationship between the premise "{sentence1}" and the hypothesis "{sentence2}".\n"""
sft_mnli_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_stsb_prompt = "System_prompt:\nYou are an assitant to analyze the semantic similarity between pairs of sentences. The task is to provide a similarity score on a scale of 0 to 5.\n"
sft_stsb_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert a float number between 0 to 5 here}} </Judgement>\n"""
sft_stsb_prompt += """Prompt:\nNow determine whether they are paraphrases or not for the pair of sentences "{sentence1}" and "{sentence2}".\n"""
sft_stsb_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_copa_prompt = "System_prompt:\nYou are an assistant to determine the {question} of a given premise. You need to evaluate which of the two options is the {question} of the premise.\n"
sft_copa_prompt += """\nThe result you give should have the following form:\n<Judgement> {{ Insert "Option one" or "Option two" here }} </Judgement>\n"""
sft_copa_prompt += """Prompt:\nNow choose the most plausible {question} for the premise "{premise}" from option one "{choice1}" and option two "{choice2}".\n"""
sft_copa_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_wsc_prompt = "System_prompt:\nYou are an assistant to resolve coreferences in a given sentence. The task is to determine whether the pronoun refers to the specified noun phrase.\n"
sft_wsc_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_wsc_prompt += """Prompt:\nNow determine in the sentence "{text}", whether the pronoun "{span2_text}" refers to "{span1_text}"?\n"""
sft_wsc_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_wic_prompt = "System_prompt:\nYou are an assistant to analyze the meaning of a word in two given sentences. The task is to determine whether the word has the same meaning or not in the two sentences.\n"
sft_wic_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Same" or "Different" here}} </Judgement>\n"""
sft_wic_prompt += """Prompt:\nNow determine whether the word '{word}' has the same meaning or not in these two sentences '{sentence1}' and '{sentence2}'.\n"""
sft_wic_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_boolq_prompt = "System_prompt:\nYou are an assistant to answer a question based on a given passage with 'yes' or 'no'.\n"
sft_boolq_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_boolq_prompt += """Prompt:\nPassage: "{passage}"\nQuestion: "{question}?"\n"""
sft_boolq_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_multirc_prompt = "System_prompt:\nYour task is to determine if the answer to the question is consistent with the information in the passage.\n"
sft_multirc_prompt += """\nThe result you give should have the following form:\n<Judgement> {{Insert "True" or "False" here}} </Judgement>\n"""
sft_multirc_prompt += """Prompt:\nPassage: "{passage}"\nQuestion: "{question}"\nAnswer: "{answer}"\n"""
sft_multirc_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_record_prompt = "System_prompt:\nYour task is to assess the query for consistency with the passage, grammatical correctness, and logical correctness. Output 'No' if any of the above are incorrect, and 'Yes' if all are correct.\n"
sft_record_prompt += """\nThe result you provide should have the following form:\n<Judgement> {{Insert "Yes" or "No" here}} </Judgement>\n"""
sft_record_prompt += """Prompt:\nPassage: "{text}"\nNow assess the Query: "{query}"\n"""
sft_record_prompt += """Assistant:\n<Judgement>{label}</Judgement>"""

sft_prompts = {
    'sst': sft_sst_prompt, #1 单句子分类任务，正负情感
    'cola': sft_cola_prompt, #1 单句子分类任, 是否符合语法
    'mrpc':  sft_mrpc_prompt, #2 句子对是否语义等效
    'qqp': sft_qqp_prompt, #2 判定句子对是否等效
    'sts-b':  sft_stsb_prompt, #2 判定句子对相似度 0-5
    'mnli': sft_mnli_prompt, #2 给定前提语句和假设，任务是预测前提语句是否（蕴含, entailment），（矛盾，contradiction）或者（中立，neutral）
    'qnli': sft_qnli_prompt, #2 判断问题和句子是否蕴含
    'rte': sft_rte_prompt, #2 句子1和句子2是否互为蕴含
    'wnli': sft_wnli_prompt, #2 句子1和句子2是否互为蕴含
    "copa": sft_copa_prompt, # SuperGLUE 从choice 1/2中选择premise合适的cause/effect
    "wsc": sft_wsc_prompt, # SuperGLUE 判断span text2是否coreference span text1
    "wic": sft_wic_prompt, # SuperGLUE 判断word在sentence 1/2中意思是否一致
    "boolq": sft_boolq_prompt, # BoolQ 根据passage回答问题
    "multirc": sft_multirc_prompt, # MultiRC 根据passage判断答案对错
    "record": sft_record_prompt # RecordLinking 从passage中寻找关键词回答问题
}

""" Downstream task prompt """

ans_only_prompt = "Answer with only one of {{positive or negative}} without any other characters"
CoT_prompt = "Think step by step, but your final answer should strictly be one of {{positive or negative}} without any other characters"
sentiment_analysis = "{inference_prompt}. Judge the sentiment of the following sentence: {sentence} Assistant: "

ans_only_prompt_nli = "Answer with only one of {{contradiction / neutral / entailment}} without any other characters"
CoT_prompt_nli = "Think step by step, but your final answer should strictly be one of {{contradiction / neutral / entailment}} without any other characters"
nli_prompt = "{inference_prompt}. Judge the logical relationship between '{premise}' and '{hypothesis}' Assistant: "