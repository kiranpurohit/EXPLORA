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
        
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, max_new_tokens=200, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
        
    out_text = []
    for x in range(0, 10):
        out_text.append(outputs[x]["generated_text"])
    return out_text


def compare_llm_outputs(user_query):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query)

    return results

# with open('pickle_cs_tr.pkl', 'rb') as f:
#     l = pickle.load(f)

# Replace with your API keys and endpoint URLs
#api_keys = ["<API_KEY_1>", "<API_KEY_2>", "<API_KEY_3>"]
#endpoint_urls = ["<ENDPOINT_URL_1>", "<ENDPOINT_URL_2>", "<ENDPOINT_URL_3>"]
#llm_names = ["LLM 1", "LLM 2", "LLM 3"]

def standardize_normalize_cosine_similarities(cosine_similarities):
    """Normalized cosine similarities"""
    # normalize into 0-1 range
    cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
            np.max(cosine_similarities) - np.min(cosine_similarities))

    # standardize and shift by 0.5
    cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / np.std(cosine_sims_norm)

    return cosine_sims_norm

def max_normalize_cosine_similarities_pairwise(cosine_similarities):
    """Normalized cosine similarities of pairs which is 2d matrix of pairwise cosine similarities"""
    cosine_sims_norm = np.copy(cosine_similarities)
    np.fill_diagonal(cosine_sims_norm, np.NaN)

    # normalize into 0-1 range
    cosine_sims_norm = (cosine_similarities - np.nanmin(cosine_similarities, axis=0)) / (
            np.nanmax(cosine_similarities, axis=0) - np.nanmin(cosine_similarities, axis=0))

    # standardize shift by 0.5
    cosine_sims_norm = \
        0.5 + (cosine_sims_norm - np.nanmean(cosine_sims_norm, axis=0)) / np.nanstd(cosine_sims_norm, axis=0)

    return cosine_sims_norm

def max_normalize_cosine_similarities(cosine_similarities):
    """Normalize cosine similarities using max normalization approach"""
    return 1 / np.max(cosine_similarities) * cosine_similarities.squeeze(axis=1)

def mmr(data_emb, sent_emb, k):
    beta = 0.7
    
    text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
    candidate_sims = cosine_similarity(data_emb)
    text_sims_norm = standardize_normalize_cosine_similarities(text_sims)
    phrase_sims_norm = max_normalize_cosine_similarities_pairwise(candidate_sims)
    # text_sims = emb
    # candidate_sims = l
    selected_data_indices = []
    unselected_data_indices = list(range(len(data_emb)))

    # find the most similar doc (using original cosine similarities)
    best_idx = np.argmax(text_sims)
    selected_data_indices.append(best_idx)
    unselected_data_indices.remove(best_idx)

    # do top_n - 1 cycle to select top N data
    for _ in range(min(len(data_emb), k) - 1):
        unselected_data_distances_to_text = text_sims_norm[unselected_data_indices, :]
        unselected_data_distances_pairwise = phrase_sims_norm[unselected_data_indices][:,
                                                selected_data_indices]

        # if dimension of data distances is 1 we add additional axis to the end
        if unselected_data_distances_pairwise.ndim == 1:
            unselected_data_distances_pairwise = np.expand_dims(unselected_data_distances_pairwise, axis=1)

        # find new candidate with
        idx = int(np.argmax(
            beta * unselected_data_distances_to_text - (1 - beta) * np.max(unselected_data_distances_pairwise,
                                                                                axis=1).reshape(-1, 1)))
        best_idx = unselected_data_indices[idx]

        # select new best docs and update selected/unselected phrase indices list
        selected_data_indices.append(best_idx)
        unselected_data_indices.remove(best_idx)
        top_sent = [idx for idx in selected_data_indices]

    return top_sent

# Self consistency on 10 generated answers
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        ans = ""
        if len(tmp.split("Final Answer:"))>6:
            ans = tmp.split("Final Answer:")[6]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "**************")
            try:
                float(ans)
                ans = round(float(ans))
                ans_list.append(ans)
            except: pass
        # ans_list.append(ans)

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
    # n = [(k, v) for k, v in d.items()]
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
# def get_prompt(ex):
#     s = "\n\n"
#     s += "Question:" + ex[2]+"\n"  # 2
#     ex[3] = re.sub("<<.*?>>", "", ex[3])    # 3
#     ex[3] = ex[3].replace("#### ", "Final Answer:") # 3
#     s += ex[3]   # 3
#     return s

########################################## Main #####################################
if __name__ == "__main__":

    # Pickled embeddings for test and train
    with open('pickle_test.pkl', 'rb') as f:
        l1 = pickle.load(f)
    
    with open('pickle_tr.pkl', 'rb') as f1:
        l2 = pickle.load(f1)

    # test_emb = np.array(l1)
    # train_emb = np.array(l2)
   
    ### Load data
    with open("train.jsonl", 'r') as f:
        json_list = list(f)
    train_set = [json.loads(x) for x in json_list]
    # with open("output/static_subset_selection_llama3.csv", newline='') as f:
    #     reader = csv.reader(f)
    #     train_set = list(reader)
    # train_set = train_set[1:]

    with open("test.jsonl", 'r') as f:
        json_list = list(f)
    test_set = [json.loads(x) for x in json_list]

    # Set the lambda parameter (trade-off between relevance and diversity)
    lambda_param = 0.7

    # Set the number of documents to return
    top_k = 5
    # Apply MMR function
    exnum = 0
    matches = 0
    mismatches = 0
    # test_set = test_set[372:]

    
    for ex in tqdm(test_set,total=len(test_set),desc="Generating"):

        user_query = "Follow given examples and solve the Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"

        selected_indices = mmr(l2, l1[exnum], top_k)
        # selected_indices = test_emb[exnum]

        print("\nSelected Indices:", selected_indices)

        for tr in selected_indices:
            user_query += get_prompt(train_set[tr])

        user_query += "\n\nFollowing the given examples generate step by step reasoning in Answer and generate Final Answer for the below question.\n\n" 
        user_query += "Question:" + ex["question"]
        # print(user_query)
        tmp_list = compare_llm_outputs(user_query) 
        # print(tmp_list)
        n = self_con(tmp_list)
        print(n)
        ground_truth = int(clean_ans(ex["answer"])) 

        answer = ""
        maxf = 0
        if len(n)==0: answer=""
        else: maxf = n[0][1]

        for z in n:
            if z[1]==maxf:
                if ground_truth==z[0]:
                    answer = z[0]

        if answer=="": 
            mismatches += 1
            if len(n)>0: answer = n[0][0]
        else: matches += 1
        
        print("Answer:", answer)
        print("Ground Truth:", ground_truth)
        
        exnum += 1
        print("Accuracy:", matches/exnum)

print("End of Execution")
