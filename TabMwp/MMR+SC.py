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
endpoint_urls = ["https://ba34-203-110-242-13.ngrok-free.app"]# "https://9451-130-75-87-254.ngrok-free.app", "https://7a6a-130-75-87-254.ngrok-free.app"]#, "https://akdeniz27-llama-2-70b-chat-hf-with-easyllm.hf.space/"]
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
        # tmp_list.append(compare_llm_outputs(user_query))
        # tmp = compare_llm_outputs(user_query)
        # print(tmp)
        ans = ""
        if len(tmp.split("The answer is:"))>1:
            ans = tmp.split("The answer is:")[1]
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
    with open('pickle_mmr_final.pkl', 'rb') as f:
        test_emb = pickle.load(f)
  

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
    # with open("output/static_subset_selection_ll2.csv", newline='') as f:
    #     reader = csv.reader(f)
    #     train_set = list(reader)
    # train_set = train_set[1:]

    # Set the lambda parameter (trade-off between relevance and diversity)
    lambda_param = 0.7

    # Set the number of documents to return
    top_k = 5
    # Apply MMR function
    exnum = 0
    matches = 0
    mismatches = 0
    for ex in tqdm(dev_set,total=len(dev_set),desc="Generating"):
        user_query = """Follow the giving Examples each using its Table to find the answer for its Question with the reasoning and solve the Test Question in a similar manner.
        Examples:
        """
        selected_indices = test_emb[exnum]

        print("\nSelected Indices:", selected_indices)

        for tr in selected_indices:
            user_query += "\nTable:\n" + train_set[tr]["table"] + "\nQuestion:" + train_set[tr]["question"] + "\Answer:" + train_set[tr]["solution"] + "\The answer is:" + train_set[tr]["answer"]
        # For explora 6,1,10,3
        user_query += "\nTable:\n" + ex["table"] + "\nQuestion:" + ex["question"]
        # print(user_query)
        tmp_list = compare_llm_outputs(user_query)
        # print(len(tmp_list))
        # print("\n")
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
