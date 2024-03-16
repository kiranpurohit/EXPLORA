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
    outputs = pipeline(prompt, max_new_tokens=200, do_sample=True, num_return_sequences=1, temperature=0.5, top_k=10, top_p=1.0)
        
    # out_text = []
    # for x in range(0, 10):
    #     out_text.append(outputs[x]["generated_text"])
    return outputs[0]["generated_text"]



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


#################################################################################

count = 0
counts = []
exnum = 1
for ex in tqdm(dev_set,total=len(dev_set),desc="Generating"):
    
    user_query ="""Question: A dog show enthusiast recorded the weight of the winning dog at recent dog shows. According to the table, what was the rate of change between 2015 and 2016?,
                        Table: Year | Weight (kg)\n2014 | 29\n2015 | 28\n2016 | 25\n2017 | 23\n2018 | 31,
                        Rationale: Plug the numbers into the formula for rate of change and simplify.\nRate of change\n = \\frac{change in value}{change in time}\n = \\frac{25 kilograms - 28 kilograms}{2016 - 2015}\n = \\frac{25 kilograms - 28 kilograms}{1 year}\n = \\frac{-3 kilograms}{1 year}\n = -3 kilograms per year\nThe rate of change between 2015 and 2016 was - 3 kilograms per year.,
                        Answer: -3 \n
                        Question: Todd buys 2 kilograms of orange erasers and 1 kilogram of white erasers . What is the total cost?,
                        Table: red erasers | $0.94 per kilogram\nyellow erasers | $1.49 per kilogram\nwhite erasers | $1.18 per kilogram\nblue erasers | $1.71 per kilogram\norange erasers | $1.53 per kilogram\npurple erasers | $1.38 per kilogram\ngreen erasers | $0.94 per kilogram,
                        Rationale: Find the cost of the orange erasers. Multiply:\n\n$1.53 \u00d7 2 = $3.06\n\nFind the cost of the white erasers. Multiply:\n\n$1.18 \u00d7 1 = $1.18\n\nNow find the total cost by adding:\n\n$3.06 + $1.18 = $4.24\n\nThe total cost is $4.24.,
                        Answer: 4.24 \n
                          Question: An insurance company compiled the number of doctor visits made by some patients last year. How many patients are there in all?,
                          Table: Number of trips | Frequency\n0 | 8\n1 | 4\n2 | 12\n3 | 14\n4 | 6\n5 | 18\n6 | 2,
                          Rationale: Add the frequencies for each row.\n\nAdd:\n\n8 + 4 + 12 + 14 + 6 + 18 + 2 = 64\n\nThere are 64 patients in all.,
                          Answer: 64 \n
                          Question: A stock broker followed the stock prices of a certain set of companies. How much more did Thompson Corporation's stock cost on Monday than on Tuesday?,
                          Table: Company | Monday | Tuesday\nWhite and Company | $11 | $17\nThompson Corporation | $16 | $9\nWatson and Partners | $1 | $16\nHouston Sporting Goods | $15 | $7\nComputer Data Company | $4 | $20
                          Rationale: Find the Thompson Corporation row. Find the numbers in this row for Monday and Tuesday.\n\nMonday: $16.00\nTuesday: $9.00\n\nNow subtract:\n\n$16.00 \u2212 $9.00 = $7.00\n\nThompson Corporation's stock cost $7 more on Monday than on Tuesday.
                          Answer: 7 \n
                          Question: A real estate agent counted the number of offices per floor in the building she is selling. How many floors have at least 30 offices? 
                          Table: Stem | Leaf \n2 | 8, 9\n3 | 8\n4 | 5, 6\n5 | 1\n6 | 2, 2, 9\n7 | 8\n8 | 5, 5, 7, 8\n9 | 0, 0
                          Rationale: Count all the leaves in the rows with stems 3, 4, 5, 6, 7, 8, and 9.\n\nYou counted 16 leaves, which are blue in the stem-and-leaf plot above. 16 floors have at least 30 offices,
                          Answer: 16 \n
                          Question: A manager at Stafford Rentals took an inventory of the company's dishes. How many more clay platters than clay cups are there?
                          Table: Material | Cups | Platters\nSteel | 18 | 13\nClay | 1 | 15\nPlastic | 18 | 2\nCeramic | 14 | 17
                          Rationale:Find the Clay row. Find the numbers in this row for platters and cups.\n\nplatters: 15\ncups: 1\n\nNow subtract:\n\n15 \u2212 1 = 14\n\nThere are 14 more clay platters than clay cups.,
                          Answer: 14.0
            """+"give best concise answer by generating rationales  Rationale..  Answer:.. answer following above examples to solve question using data in both table (In table columns are separated by | and rows by \n (newline).) and text. For answer give only numbers or single words without narrative for Question:" + ex["question"]+",Table: "+ex["table"]+"Output in format Rationale:, Answer:"
    tmp_list = compare_llm_outputs(user_query)
    answer = ""
    if len(tmp_list.split("The answer is:"))>6:
        answer = tmp_list.split("The answer is:")[6]
        answer = answer.split("\n")[0]
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    answer = answer.replace(",", "")
    
    print("\nAnswer: ", answer)
    print("GT: ", ex["answer"])
    ground_truth = ex["answer"]
    if answer!="" and (ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower()):
        matches+=1
    else:
        mismatches+=1
print("EM:", matches/(matches+mismatches))
