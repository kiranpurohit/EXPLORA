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
# dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]
dev_set1 = dict(list(dev_set1.items())) #7686

# train_set = []
dev_set = []
for i in dev_set1:
    dev_set.append(dev_set1[i])

matches = 0
mismatches = 0
counts = []
exnum = 1
for ex in tqdm(dev_set,total=len(dev_set),desc="Generating"):
    user_query = """Follow the giving Examples each using its Table to find the answer for its Question with the reasoning and solve the Test Question in a similar manner.

            Examples:

            Table:
            Stem | Leaf
            3 | 3, 3, 3, 5, 5
            4 | 6
            5 | 4, 5, 7, 8
            6 | 7, 8
            7 | 2, 3, 7, 9
            8 | 6, 8, 9
            Question:The members of the local garden club tallied the number of plants in each persons garden. How many gardens have at least 47 plants?
            Answer: Find the row with stem 4. Count all the leaves greater than or equal to 7.

            Count all the leaves in the rows with stems 5, 6, 7, and 8.

            You counted 13 leaves, which are blue in the stem-and-leaf plots above. 13 gardens have at least 47 plants.
            The answer is:13
            Table:
            Day | Number of tickets
            Friday | 71
            Saturday | 74
            Sunday | 75
            Monday | 72
            Question:The transportation company tracked the number of train tickets sold in the past 4 days. On which day were the fewest train tickets sold?
            Answer: Find the least number in the table. Remember to compare the numbers starting with the highest place value. The least number is 71.

            Now find the corresponding day. Friday corresponds to 71.
            The answer is:Friday

            Table:
            Donation level | Number of donors
            Gold | 15
            Silver | 68
            Bronze | 58
            Question:The Burlington Symphony categorizes its donors as gold, silver, or bronze depending on the amount donated. What fraction of donors are at the bronze level? Simplify your answer.
            Answer: Find how many donors are at the bronze level.

            58

            Find how many donors there are in total.

            15 + 68 + 58 = 141

            Divide 58 by 141.

            58/141

            58/141 of donors are at the bronze level.
            The answer is:58/141

            Table:
            Number of times | Frequency
            0 | 1
            1 | 18
            2 | 12
            3 | 13
            4 | 0
            Question:Employees at Eve's Movies tracked the number of movies that customers rented last month. How many customers are there in all?
            Answer: Add the frequencies for each row.

            Add:

            1 + 18 + 12 + 13 + 0 = 44

            There are 44 customers in all.
            The answer is:44

            Table:
            Day | Number of hammers
            Thursday | 7
            Friday | 6
            Saturday | 6
            Sunday | 9
            Monday | 4
            Tuesday | 2
            Question:A hardware store monitored how many hammers it sold in the past 6 days. What is the range of the numbers?
            Answer: Read the numbers from the table.

            7, 6, 6, 9, 4, 2

            First, find the greatest number. The greatest number is 9.

            Next, find the least number. The least number is 2.

            Subtract the least number from the greatest number:

            9 - 2 = 7

            The range is 7.
            The answer is:7

            """+"Following the given examples generate the answer for:\n Table:\n" + ex["table"] + "\nQuestion:" + ex["question"]
    # tmp_list = []
    tmp_list = compare_llm_outputs(user_query)
    print(len(tmp_list))
    ans_list = []

    for tmp in tmp_list:
        ans = ""
        if len(tmp.split("The answer is:"))>1:
            ans = tmp.split("The answer is:")[-1]
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
