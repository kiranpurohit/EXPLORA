import numpy as np
from numpy import linalg
import random
import re
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch
import pickle 
import json
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

system_message = """The following is a conversation between a Human and an AI Assistant.
The assistant is helpful, respectful and honest, and it always answers as helpfully as possible, while being safe.
The Assistant's answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that the Assistant's responses are socially unbiased and positive in nature.
If a question by the human does not make any sense, or is not factually coherent, the Assistant should explain why instead of answering something not correct.
If the Assistant does not know the answer to a question, please don't share false information.
####

"""
# with open('pickle_cs_tr.pkl', 'rb') as f:
#     l = pickle.load(f)

# Replace with your API keys and endpoint URLs
#api_keys = ["<API_KEY_1>", "<API_KEY_2>", "<API_KEY_3>"]
#endpoint_urls = ["<ENDPOINT_URL_1>", "<ENDPOINT_URL_2>", "<ENDPOINT_URL_3>"]
#llm_names = ["LLM 1", "LLM 2", "LLM 3"]

api_keys = ["EMPTY", "EMPTY", "EMPTY"]#, "EMPTY"]
endpoint_urls = ["https://3f45-130-75-152-24.ngrok-free.app"]# "https://9451-130-75-87-254.ngrok-free.app", "https://7a6a-130-75-87-254.ngrok-free.app"]#, "https://akdeniz27-llama-2-70b-chat-hf-with-easyllm.hf.space/"]
llm_names = []

for api_key, endpoint_url in zip(api_keys, endpoint_urls):
    if 'hf.space' in endpoint_url:
        model_name = endpoint_url.replace('https://', '').replace('.hf.space', '').replace('/', '')
    else:
        openai.api_key = api_key
        openai.api_base = f"{endpoint_url}/v1"
        model_names = openai.Model.list()
        model_name = model_names["data"][0]["id"]
    llm_names.append(model_name)

# Function to retrieve LLM outputs using the given API key and endpoint
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(prompt, api_key, endpoint_url, hard_code_exception=False):
    # if new_sheet_name=='poem_properties':
    #     if hard_code_exception==True:
    #         max_tokens=128
    #     else:
    #         max_tokens=150
    # else:

    max_tokens=256
    if 'hf.space' in endpoint_url:
        client = Client(endpoint_url)
        result = client.predict(
                        prompt, # str in 'Message' Textbox component
                        api_name="/chat"
        )
        return result.strip()
    openai.api_key = api_key
    openai.api_base = f"{endpoint_url}/v1"
    model_names = openai.Model.list()
    model_name = model_names["data"][0]["id"]

    res = openai.Completion.create(
        model=model_name,  # Replace with your model name
        prompt=system_message + prompt,
        # messages=[
        #     {"role": "system", "content": system_message},
        #     {"role": "user", "content": prompt},
        # ],
        temperature=0.5,
        top_k=10,
        top_p=1.0,
        n=10,
        max_tokens=256,
    )
    
    out_text = []
    for x in range(0, 10):
        out_text.append(res['choices'][x]['text'].strip())
    return out_text


def compare_llm_outputs(user_query, hard_code_exception=False):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)

    return results

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

def mmr(data, emb, k):
    beta = 0.7
    
    text_sims = emb
    candidate_sims = l

    text_sims_norm = standardize_normalize_cosine_similarities(text_sims)
    phrase_sims_norm = max_normalize_cosine_similarities_pairwise(candidate_sims)

    selected_data_indices = []
    unselected_data_indices = list(range(len(data)))

    # find the most similar doc (using original cosine similarities)
    best_idx = np.argmax(text_sims)
    selected_data_indices.append(best_idx)
    unselected_data_indices.remove(best_idx)

    # do top_n - 1 cycle to select top N data
    for _ in range(min(len(data), k) - 1):
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
        if len(tmp.split("Final Answer:"))>1:
            ans = tmp.split("Final Answer:")[1]
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
    
    # with open('pickle_test.pkl', 'rb') as f:
    #     l1 = pickle.load(f)
    
    # with open('pickle_st_mis.pkl', 'rb') as f1:
    #     l2 = pickle.load(f1)

    # test_emb = np.array(l1)
    # train_emb = np.array(l2)
    with open('pickle_mmr_lst.pkl', 'rb') as f:
        test_emb = pickle.load(f)

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

        # selected_indices = mmr(train_set, test_emb[exnum], top_k)
        selected_indices = test_emb[exnum]

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
