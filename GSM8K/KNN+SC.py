import faiss
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

def knn(doc_embeddings, query_embedding, lambda_param, top_k):
    # Normalize embeddings
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Number of documents
    num_docs = doc_embeddings.shape[0]

    # Create an index for Faiss
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings.astype(np.float32))

    # Query the index for similar documents
    _, similarity_indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)

    # Compute relevance scores
    relevance_scores = np.dot(doc_embeddings, query_embedding)

    # Initialize selected set
    selected_set = set()

    # Add the most relevant document to the selected set
    max_relevance_index = np.argmax(relevance_scores)
    selected_set.add(max_relevance_index)

    # Compute KNN scores and select documents iteratively
    while len(selected_set) < top_k:
        remaining_indices = list(set(similarity_indices[0]) - selected_set)
        remaining_embeddings = doc_embeddings[remaining_indices]

        # Compute similarity with the query for remaining documents
        similarity_scores = np.dot(remaining_embeddings, query_embedding)
        # print(similarity_scores)
        # Compute KNN scores
        knn_scores = lambda_param * relevance_scores[remaining_indices] - (1 - lambda_param) * similarity_scores

        # Select document with maximum KNN score
        max_knn_index = remaining_indices[np.argmax(knn_scores)]
        selected_set.add(max_knn_index)

    # Convert selected set to a list of indices
    selected_indices = list(selected_set)

    return selected_indices

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

########################################## Main #####################################
if __name__ == "__main__":
    
    with open('pickle_test.pkl', 'rb') as f:
        l1 = pickle.load(f)
    
    with open('pickle_tr.pkl', 'rb') as f1:
        l2 = pickle.load(f1)

    test_emb = np.array(l1)
    train_emb = np.array(l2)

    ### Load data
    with open("train.jsonl", 'r') as f:
        json_list = list(f)
    train_set = [json.loads(x) for x in json_list]

    with open("test.jsonl", 'r') as f:
        json_list = list(f)
    test_set = [json.loads(x) for x in json_list]

    # Set the lambda parameter (trade-off between relevance and diversity)
    lambda_param = 0.7

    # Set the number of documents to return
    top_k = 5
    # Apply KNN function
    exnum = 0
    matches = 0
    mismatches = 0
    # test_set = test_set[:10]
    
    for ex in tqdm(test_set,total=len(test_set),desc="Generating"):

        user_query = "Follow given examples and solve the Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"

        selected_indices = knn(train_emb, test_emb[exnum], lambda_param, top_k)

        print("\nSelected Indices:", selected_indices)

        for tr in selected_indices:
            user_query += get_prompt(train_set[tr])

        user_query += "\n\nFollowing the given examples generate step by step reasoning in Answer and generate Final Answer for the below question.\n\n" 
        user_query += "Question:" + ex["question"]
        # print(user_query)
        tmp_list = compare_llm_outputs(user_query)
        # print(len(tmp_list))
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
