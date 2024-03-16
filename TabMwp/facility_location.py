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
import re

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
        
    # prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(msg_in, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
        
    out_text = []
    for x in range(0, 10):
        out_text.append(outputs[x]["generated_text"])
    return out_text



def compare_llm_outputs(user_query, hard_code_exception=False):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)

    return results


# Self consistency on 10 generated answers
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        # tmp_list.append(compare_llm_outputs(user_query))
        # tmp = compare_llm_outputs(user_query)
        # print(tmp)
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

    d = {}
    for i in ans_list:
        if i=="":
            continue
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

# Example usage
if __name__ == "__main__":
    # Generate random embeddings for demonstration
    with open('pickle_train_new.pkl', 'rb') as f:
        l = pickle.load(f)
      
    X_p = pairwise_distances(l, metric="euclidean", squared=True)
    # model_a = GraphCutSelection(5, "euclidean", verbose=True)
    model_b = FacilityLocationSelection(5, "euclidean", verbose=True)
    
    # ans1 = model_a.fit_transform(X_p.max() - X_p)
    ans2 = model_b.fit_transform(X_p.max() - X_p)
    
    # l1 = []
    l2 = []

    for i in range(5):
        # l1.append(model_a.ranking[i])
        l2.append(model_b.ranking[i])
      
    print("Selected Indices:", l2)

    # Load datasets
    dev_set1 = open("problems_dev.json")
    dev_set1 = json.load(dev_set1)
    dev_set1 = dict(list(dev_set1.items())) #7686
    dev_set = []
    for i in dev_set1:
        dev_set.append(dev_set1[i])

    train_set1 = open("problems_train.json")
    train_set1 = json.load(train_set1)
    train_set1 = dict(list(train_set1.items())) #23059
    train_set = []
    for i in train_set1:
        train_set.append(train_set1[i])
    
    exnum = 0
    matches = 0
    mismatches = 0
    for ex in tqdm(dev_set,total=len(dev_set),desc="Generating"):
        user_query = """Follow the giving Examples each using its Table to find the answer for its Question with the reasoning and solve the Test Question in a similar manner.
        Examples:
        """
        selected_indices = l2

        for tr in selected_indices:
            user_query += "\nTable:\n" + train_set[tr]["table"] + "\nQuestion:" + train_set[tr]["question"] + "\Answer:" + train_set[tr]["solution"] + "\The answer is:" + train_set[tr]["answer"]
        # For explora 6,1,10,3
        user_query += "\nTable:\n" + ex["table"] + "\nQuestion:" + ex["question"]
        # print(user_query)
        tmp_list = compare_llm_outputs(user_query)
        # print(len(tmp_list))
        print("\n")
        n = self_con(tmp_list)
        print(n)
        answer = ""
        if len(n)>0: answer = n[0][0]
            
        print("Answer: ", answer)
        gt = ex["answer"]
        
        print("GT: ", gt)
        
        if answer!="" and (gt.lower() in answer.lower() or answer.lower() in gt.lower()):
            matches+=1
        else:
            mismatches+=1
        exnum += 1
        # print("Hits:", matches)
        print("Accuracy:", matches/exnum)
    

print("End of Execution")
