import numpy as np
from numpy import linalg
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch
import re
import pickle 
import json
from tqdm import tqdm

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

import transformers
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./cuda_executable
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)

#####################################################################################################


# Gen resp
def get_completion(msg_in):

    messages = [
        {
            "role": "user",
            "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer.",
        },
        {
            "role":"assistant",
            "content": "I understand.",
        },
        {
            "role": "user", 
            "content": msg_in,
        }
    ]
        
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
        
    # out_text = []
    # for x in range(0, 10):
    #     out_text.append(outputs[x]["generated_text"])
    return outputs[0]["generated_text"]


def compare_llm_outputs(user_query):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query)

    return results

# Self consistency on 10 generated answers
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        ans = ""
        if len(tmp.split("Final Answer:"))>6:
            ans = tmp.split("Final Answer:")[6]
            ans = ans.split("\n")[0]
        ans = re.sub("\D","",ans)
        ans_list.append(ans)

    # print(ans_list)

    d = {}
    for i in ans_list:
        if i=="":
            continue
        if int(i) in d:
            d[int(i)] += 1
        else:
            d[int(i)] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

# Strip answer from sentence
def clean_ans(s):
    ans_s = s.split("#### ")[1]
    ans_s = ans_s.replace(",","")
    return ans_s

def get_prompt(ex):
    s = "\n\n"
    s += "Question:" + ex["question"]+"\n"
    ex["answer"] = re.sub("<<.*?>>", "", ex["answer"])
    ex["answer"] = ex["answer"].replace("#### ", "Final Answer:")
    s += ex["answer"]
    return s

##################################################### Main ######################################3

### Load data
with open("train.jsonl", 'r') as f:
    json_list = list(f)
train_set = [json.loads(x) for x in json_list]

with open("test.jsonl", 'r') as f:
    json_list = list(f)
test_set = [json.loads(x) for x in json_list]

### Random 5 as prompt
prompt = "Follow given examples and solve the Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"

# train_set = train_set[:5]
rand_list = random.sample(range(0,len(train_set)-1), 5)
for i in rand_list:
    prompt += "Question:" + train_set[i]["question"]+"\nAnswer:"
    train_set[i]["answer"] = re.sub("<<.*?>>", "", train_set[i]["answer"])
    train_set[i]["answer"] = train_set[i]["answer"].replace("#### ", "Final Answer:")
    prompt += train_set[i]["answer"] + "\n\n" 

# print(prompt)

########################################## Predictions #########################

matches = 0
mismatches = 0
exnum = 1
# test_set = test_set[85:90]

for ex in tqdm(test_set, total=len(test_set), desc="Generating"):

    user_query = prompt + "Following the examples, generate step by step reasoning in Answer and generate Final Answer for the below question.\n\n" 
    user_query += "Question:" + ex["question"]
    
    tmp_list = compare_llm_outputs(user_query)
    print(tmp_list)
    ans = ""
    if len(tmp_list.split("Final Answer:"))>6:
        ans = tmp_list.split("Final Answer:")[6]
        ans = ans.split("\n")[0]
        ans = re.sub("\D","",ans)
    answer = ans
    # print("\n")
    # n = self_con(tmp_list)
    # print(n)
    # if len(n)==0: answer=""
    # else: answer = n[0][0]
    # if answer=="" and len(n)>1: answer = n[1][0]

    ground_truth = int(clean_ans(ex["answer"]))
    print("\nAnswer:", answer)
    print("Ground Truth:", ground_truth)
    
    if answer!="" and ground_truth==int(answer):
        matches+=1
    else:
        mismatches+=1

    print("Accuracy:", matches/exnum)
    exnum += 1

print("######### End of Execution #########")
