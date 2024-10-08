# -*- coding: utf-8 -*-

from huggingface_hub import login
access_token_read = "YOUR_TOKEN"
login(token = access_token_read)

#import numpy as np
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
#np.random.seed(7)
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

def prompt_for_manual_prediction(ex, shots):
    stop_signal = "\n\n"
    showcase_examples = [
            "Q: {}\nO: {} \nA: {}. The option is {}\n".format(
                 s["question"],s["options"],
                s["rationale"], s["correct"]) for s in shots
        ]




   


    input_example = "\nQ: {}\n O: {}\nA:".format(ex['question'], ex['options'])
    prompt = "\n".join(showcase_examples + [input_example])

    return prompt, stop_signal






def in_context_manual_prediction(ex, training_data):
    template,stop = prompt_for_manual_prediction(ex, training_data)

    messages=[{
                "role": "user",
                "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic.Do not generate examples in your answer",
            }]
    text={"role": "assistant", "content":""" Follow given examples and solve the Test Question at end in similar manner by decomposing the original questions
         Examples:{}""".format(template)}
    messages.append(text)


    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, max_new_tokens=200, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
        
    out_text = []
    for x in range(0, 10):
        out_text.append(outputs[x]["generated_text"])
    return out_text




#import faiss
from transformers import BertTokenizer, BertModel, logging
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import json
import pickle
#import openai
#SSfrom tenacity import retry, stop_after_attempt, wait_random_exponential
import json


# tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# model_bert = BertModel.from_pretrained('bert-base-uncased')

# logging.set_verbosity_error()




def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        # tmp_list.append(compare_llm_outputs(user_query))
        # tmp = compare_llm_outputs(user_query)
        # print(tmp)
        ans = ""
        if len(tmp.split("The option is "))>6:
            ans = tmp.split("The option is ")[6][0]
            print(ans)
            # ans = ans.split("\n")[0]
        # ans = ans.replace("$", "")
        # ans = ans.strip()
        ans_list.append(ans)

    # print(ans_list)

    d = {}
    for i in ans_list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n
    








def test_few_shot_manual_prediction():
    print("Running prediction")
    with open('dev.jsonl', 'r') as json_file1:
        json_list1 = list(json_file1)
    dev_set=[]


    for json_str in json_list1:
        result1 = json.loads(json_str)
        dev_set.append(result1)
    dev_set=dev_set[:2]


    with open('train.jsonl', 'r') as json_file:
        json_list = list(json_file)

    train_set=[]


    random.seed(7)


    for json_str in json_list:
        result = json.loads(json_str)
        train_set.append(result)
    #train_set=train_set[:20]
    rand_list = random.sample(range(0,len(train_set)-1), 5)
    new_rand_train=[]
    for i in rand_list:
      new_rand_train.append(train_set[i])

    
    print("started Running:")
    matches=0
    mismatches=0

    for ex in tqdm(dev_set,total=len(dev_set),desc="predicting"):
        #user_query,stop=prompt_for_manual_prediction(ex,new_rand_train)

        tmp_list = in_context_manual_prediction(ex,new_rand_train)
        #print(tmp[0])
        
        n = self_con(tmp_list)
        answer = n[0][0]
        if answer=="" and len(n)>1: answer = n[1][0]
        ans = ""
        # if len(tmp[0].split("The option is "))>1:
        #     ans = tmp[0].split("The option is ")[1][0]
        # answer=ans
        
        print("\nAnswer: ", answer)
        gt = ex["correct"]
        print("GT: ", gt)
        if(answer==gt):
          matches+=1
        else:
          mismatches+=1
    print("EM:",matches/(matches+mismatches))

test_few_shot_manual_prediction()
