import numpy as np
from numpy import linalg
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch
import func_timeout
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


# Self consistency on 10 generated answers
def self_con(tmp_list):

    n = len(tmp_list)
    f_list = []
    for i in range(0, n):
        if type(tmp_list[i])==float:
            if tmp_list[i] in [float("-inf"),float("inf"),float("nan")]: continue
            f_list.append(round(tmp_list[i]))
        elif type(tmp_list[i])==int:
            f_list.append(tmp_list[i])
    
    d = {}
    for i in f_list:
        if int(i) in d:
            d[int(i)] += 1
        else:
            d[int(i)] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

def llm_output(user_query):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query)
    return results

def clean_ans(s):
    ans_s = s.split("#### ")[1]
    ans_s = ans_s.replace(",","")
    return ans_s

def execute(x):
    try:
        exec(x)
        return locals()['ans']
    except:
        return ""

######################################################################################################

def prompt_for_manual_prediction(ex):

    # Prompt manually written for the subset selected using our method, explora
    prompt = """
Write Python code to answer the question:
Question:The difference between the number of boys and girls in a tree planting event is 400. If there are 600 boys at the event, and the number of girls is more than the number of boys, what's 60% of the total number of boys and girls at the event?
# Python Code, return ans
total_boys = 600
total_girls = total_boys + 400
ans = 0.6 * (total_boys + total_girls)

Write Python code to answer the question:
Question:Casey is trying to decide which employee she wants to hire. One employee works for $20 an hour. The other employee works for $22 an hour, but Casey would also get a $6/hour subsidy from the government for hiring a disabled worker. How much money per week would Casey save by hiring the cheaper employee, if they both work 40 hours per week?
# Python Code, return ans
hourly_wage_employee_1 = 20
hourly_wage_employee_2 = 22
subsidy_per_hour = 6
hours_per_week = 40
ans = (hourly_wage_employee_1 - hourly_wage_employee_2) * hours_per_week + subsidy_per_hour * hours_per_week

Write Python code to answer the question:
Question:Cara has 60 marbles in a bag. 20 of them are yellow, half as many are green, and the remaining marbles are equally divided between red and blue. If Cara picks a marble at random, what are the odds it's blue (expressed as a percentage)?
# Python Code, return ans
yellow_marbles = 20
green_marbles = yellow_marbles / 2
remaining_marbles = 60 - yellow_marbles - green_marbles
blue_marbles = remaining_marbles / 2
red_marbles = remaining_marbles / 2
total_marbles = yellow_marbles + green_marbles + blue_marbles + red_marbles
ans = (blue_marbles / total_marbles) * 100

Write Python code to answer the question:
Question:Samir just turned half the age Hania was 10 years ago. If in five years Hania will be 45 years old, what will Samir's age be five years from now?
# Python Code, return ans
hania_age_in_5_years = 45
hania_age_10_years_ago = hania_age_in_5_years - 15
samir_current_age = hania_age_10_years_ago / 2
ans = samir_current_age + 5

Write Python code to answer the question:
Question:If the normal hours of operation of Jean's business are 4 pm to 10p every day Monday through Friday, and from 6 pm to 10 pm on weekends, how many hours is the business open in a week?
# Python Code, return ans
weekdays_hours = 6
weekend_hours = 4
ans = 0
for i in range(7):
    if i < 5:
        ans += weekdays_hours
    else:
        ans += weekend_hours
"""
    prompt += "\nWrite Python code to answer the question:\nQuestion:" + ex["question"]
    prompt += "\n# Python Code, return ans\n"
    
    return prompt


def get_open_source_completions(test_data):

    # stop_signal = "\n\n"
    matches = 0
    mismatches = 0

    question_df = {"question":[],"answer":[]}
    
    no_ans = 0
    acc_records = []
    exnum = 1

    # test_data = test_data[:5]
    
    for row in tqdm(test_data, total=len(test_data), desc="Generating"):

        prompt = prompt_for_manual_prediction(row)
        answer_list = llm_output(prompt)
        ans_list = []
        for i in answer_list:
            code = ""
            z = i.split("# Python Code, return ans\n")
            if len(z)>6:
                code = z[6]
            elif len(z)>1:
                code = z[-1]
            code = code.split("\n\n")[0]
            if "</s>" in code:
                code = code.split("</s>")[1].strip()
            ans_list.append(code)
            
        # print(ans_list)
        # Getting code and output
        tmp_list = []
        for i in ans_list:
            # code = i.split("\n\n")[0]
            code = i.split("\n\n")[0].strip()
            # print(code)
            # try:
            #     exec(code)
            #     tmp_list.append(locals()['ans'])
            try:
                ans = func_timeout.func_timeout(5, execute, args=[code])
                tmp_list.append(ans)
            except func_timeout.FunctionTimedOut:
                pass
                # ans = None
            # except: pass
            
        # print(tmp_list)
        print("\n")
        n = self_con(tmp_list)
        print(n)

        ground_truth = int(clean_ans(row["answer"]))
        
        answer = ""
        maxf = 0
        if len(n)==0: 
            answer=""
            no_ans += 1
        else: maxf = n[0][1]

        for z in n:
            if z[1]==maxf:
                if ground_truth==z[0] or ground_truth==(z[0]*-1):
                    answer = z[0]

        if answer=="": 
            mismatches += 1
            if len(n)>0: answer = n[0][0]
        else: matches += 1      

        question_df["question"].append(row["question"])
        question_df["answer"].append(answer)
        
        print("Answer:", answer)
        print("Ground Truth:", ground_truth)

        print("Accuracy:", matches/exnum)
        exnum += 1

    print("EM:", matches/(matches+mismatches))
    print("No PoT:", no_ans)

    final_questions = pd.DataFrame(question_df)
    final_questions.to_csv("output/mistral_pot_question_answer.tsv",sep="\t",index=False)

    return final_questions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    # Load test data
    with open("test.jsonl", 'r') as f:
        json_list = list(f)
    test_set = [json.loads(x) for x in json_list]

    # exemplars = pd.read_csv("output/static_subset_selection_llama3.csv")
    # exemplars = exemplars[:5]
       
    final_df = get_open_source_completions(test_set)
    # print(final_df)
    print("End of execution")


if __name__=='__main__':
    test_few_shot_prediction()
