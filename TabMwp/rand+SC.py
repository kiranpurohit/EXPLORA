import numpy as np
from numpy import linalg
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch

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
    outputs = pipeline(prompt, max_new_tokens=200, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
        
    out_text = []
    for x in range(0, 10):
        out_text.append(outputs[x]["generated_text"])
    return out_text
    
def compare_llm_outputs(user_query, hard_code_exception=False):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query)

    return results


dev_set1 = open("problems_dev.json") #7965
dev_set1 = json.load(dev_set1)
dev_set1 = dict(list(dev_set1.items()))

dev_set = []
for i in dev_set1:
    dev_set.append(dev_set1[i])

train_set1 = open("problems_train.json") #23059
train_set1 = json.load(train_set1)
train_set1 = dict(list(train_set1.items()))

train_set = []
for i in train_set1:
    train_set.append(train_set1[i])


prompt =  """Follow the giving Examples each using its Table to find the answer for its Question with the reasoning and solve the Test Question in a similar manner.

            Examples:
            """

rand_list = random.sample(range(0,len(train_set)-1), 5)
for tr in rand_list:
    prompt += "\nTable:\n" + train_set[tr]["table"] + "\nQuestion:" + train_set[tr]["question"]
    if type(train_set[tr]["choices"])==str:
        prompt += "Please select from the following options:"+train_set[tr]["choices"]
    prompt += "\nAnswer:" + train_set[tr]["solution"] + "\nThe answer is:" + train_set[tr]["answer"]

matches = 0
mismatches = 0
counts = []
exnum = 1
for ex in tqdm(dev_set,total=len(dev_set),desc="Generating"):
    
    user_query = prompt + "\nFollowing the given examples generate the answer for:\n Table:\n" + ex["table"] + "\nQuestion:" + ex["question"]
    # tmp_list = []
    tmp_list = compare_llm_outputs(user_query)
    print(len(tmp_list))
    ans_list = []

    for tmp in tmp_list:
        ans = ""
        if len(tmp.split("The answer is:"))>6:
            ans = tmp.split("The answer is:")[6]
            ans = ans.split("\n")[0]
        ans = ans.replace("$", "")
        ans = ans.replace("%", "")
        ans = ans.replace(",", "")
        ans = ans.strip()
        ans_list.append(ans)

    # print(ans_list)
    # Self consistency on 10 generated answers
    d = {}
    for i in ans_list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    # print(n[0][0])
    
    answer = n[0][0]
    if answer=="" and len(n)>1: answer = n[1][0]
    print("\nAnswer: ", answer)
    print("GT: ", ex["answer"])
    ground_truth = ex["answer"]
    if answer!="" and (ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower()):
        matches+=1
    else:
        mismatches+=1
print("EM:", matches/(matches+mismatches))
