from huggingface_hub import login
access_token_read = "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"
login(token = access_token_read)

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
            "Read the following table, and then answer the question:\nTable: {}\nQuestion: {}\nEquation: {}\n. The answer is {}\n".format(
                 s["table"],s["question"],
                s["program"], s["answer"]) for s in shots
        ]




   


    input_example = "Read the following table, and then answer the question:\nTable: {}\nQuestion: {}\nEquation:".format(ex['table'], ex['question'])
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








import faiss
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










def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        # tmp_list.append(compare_llm_outputs(user_query))
        # tmp = compare_llm_outputs(user_query)
        # print(tmp)
        ans = ""
        if len(tmp.split("The answer is "))>6:
            ans = tmp.split("The answer is ")[6]
            #print(ans)
            ans = ans.split("\n")[0]
        try:
            ans = ans.replace("$", "")
            ans = ans.replace("%","")
            ans = ans.strip()
        except:
            pass

        if 'yes' in ans.lower() or 'true' in ans.lower():
            ans = 'yes'
        elif 'no' in ans.lower() or 'false' in ans.lower():
            ans = 'no'
        try : 
            float(ans) 
            ans = float(ans)
        except : 
            pass
        

        

        if type(ans)==float:
            #print("**")
            ans=round(ans,2)
        
       
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
    dev_set = open("finQA_test.json") #7965
    dev_set = json.load(dev_set)
    dev_set=dev_set[:1]
    train_set=open("finQA_train.json")
    train_set=json.load(train_set)
    

    
    dbfile3 = open('pickle_mmr_finqa.pkl', 'rb')
    l2 = pickle.load(dbfile3)


    
    print("started Running:")
    matches=0
    mismatches=0
    i=0

    for ex in tqdm(dev_set,total=len(dev_set),desc="predicting"):
        #user_query,stop=prompt_for_manual_prediction(ex,new_rand_train)

        # query = tokenizer_bert(ex['question'], return_tensors='pt', padding=True, truncation=True)
        # with torch.no_grad():
        #     outputs_sentence1 = model_bert(**query)
        # query_embedding=outputs_sentence1.last_hidden_state.mean(dim=1).numpy()[0]
        

        # Set the lambda parameter (trade-off between relevance and diversity)
        lambda_param = 0.5

        # Set the number of documents to return
        top_k = 5

        # Apply KNN function
        selected_indices = l2[i]
        i+=1
        print("selected indices:",selected_indices)

        train_set_1=[]
        for i in selected_indices:
          train_set_1.append(train_set[i])
        #print("1-done")
        user_query,stop=prompt_for_manual_prediction(ex,train_set_1)
        while len(user_query)>4096:
            print('*************************************\n')
            train_set_1.pop()
            user_query,stop=prompt_for_manual_prediction(ex,train_set_1)
        #print("2-done")

        tmp_list = in_context_manual_prediction(ex,train_set_1)
        #print("3-done")
        # if matches+mismatches==20:
        #     break
        #print(tmp_list[0])
        
        n = self_con(tmp_list)
        answer = n[0][0]
        if answer=="" and len(n)>1: answer = n[1][0]
        #print("4-done")
        
        
        print("\nAnswer: ", answer)
        gt = ex["answer"]
        ans=gt
        try:
            ans = ans.replace("$", "")
            ans = ans.replace("%","")
            ans = ans.strip()
        except:
            pass
        try : 
            float(ans) 
            ans = float(ans)
        except : 
            pass
        

        #ans = ans.strip()

        if type(ans)==float:
            #print("**")
            ans=round(ans,2)
        gt =ans
        print("GT: ", gt)
        
        if(answer==gt):
          matches+=1
        else:
          mismatches+=1
        print("Accuracy till now:",matches/(matches+mismatches))
    print("EM:",matches/(matches+mismatches))

test_few_shot_manual_prediction()