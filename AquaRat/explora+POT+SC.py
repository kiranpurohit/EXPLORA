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
#import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

from transformers import BertTokenizer, BertModel, logging



#for-pot

from typing import Dict, Any
import os
import json
from tqdm import tqdm
from datetime import datetime
import openai
from time import sleep
import sympy
from sympy.solvers import solve
from sympy import Symbol
import math
import argparse
from tool import simplify_ans, parse_api_result, safe_execute
from sympy import simplify
from collections import Counter

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



from huggingface_hub import login
access_token_read = "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"
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
                s["rationale"], s["correct"]) for index, s in shots.iterrows()
        ]




    # prompt = "\n".join(showcase_examples)
    # prompt=prompt+"\n\n{text}\n"


    input_example = "\nQ: {}\n O: {}\nA:".format(ex['question'], ex['options'])
    prompt = "\n".join(showcase_examples + [input_example])

    return prompt



def read_AHOTPOT_test_data():
    data = pd.read_csv("dev.csv")
    return data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_AHOTPOT_train_data():
    data = pd.read_csv("train.csv")
    return data


def in_context_manual_prediction(ex, training_data,flag):
    template = prompt_for_manual_prediction(ex, training_data)

    messages=[{
                "role": "user",
                "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic.Do not generate examples in your answer",
            }]
    text={"role": "assistant", "content":""" Follow given examples and solve the Test Question at end in similar manner by decomposing the original questions
         Examples:{}""".format(template)}
    messages.append(text)


    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if(flag==0):
        outputs = pipeline(prompt, max_new_tokens=200, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
    else:
        outputs = pipeline(prompt, max_new_tokens=10, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)   
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
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
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
    

def get_open_source_completions(test_data, data):

    stop_signal = "\n\n"
    matches = 0
    mismatches =0
    print("started running:")

    question_df = {"question":[],"answers":[]}

    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data=val_data[:20]

    #exemplars = static_subset_selection(val_data, train_data, 5, test_data)
    merged_exemplars = pd.read_csv("output/aquarat/static_subset_selection_mistral_aquarat_latest1.csv")
    # # print("exemplars:",exemplars)
    exemplars=np.array_split(merged_exemplars, 10)

    # merged_exemplars = pd.concat(exemplars)
    # merged_exemplars.to_csv("output/aquarat/static_subset_selection_Llama_aquarat_latest12.csv")
    # #*****************************************************************************
    # print("\n\n\n_____________Take the exemplar with minimum validation loss and use it as the exemplar")
    # avg_err = LLM_avg_error(exemplars, val_data)
    # print("\n\navg_err ",avg_err)
    # ind = np.argmin(avg_err)
    # print("\n\nmin ind ",ind)
    exemplars = exemplars[1]

    index=0
    acc_records = []

    for index, row in test_data.iterrows():
        prompt="""
# Write Python Code to solve the question at end using above examples. Store your result as a variable named 'ans'.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.



#Question:The average age of three boys is 15 years and their ages are in proportion 3:5:7. What is the age in years of the youngest boy?
#Answer options:['A)9', 'B)10', 'C)11', 'D)12', 'E)13']
x = symbols('x')
age_ratio = [3*x, 5*x, 7*x]
average_age = sum(age_ratio) / 3
equation1 = Eq(average_age, 15)
solution = solve_it(equation1, x)
ans = 3*solution[x]
print(ans)



#Question:Can you deduce the pattern and find the next number in the series? 6, 14, 26, 98, __?
#Answer options:['A)276', 'B)277', 'C)278', 'D)279', 'E)None of these']
n = symbols('n')
n=5
previous_term=98
ans=n**2 + (previous term + 1)
print(ans)



#Question:In covering a distance of 42 km, A takes 2 hours more than B. If A doubles his speed, then he would take 1 hour less than B. A's speed is:
#Answer options:['A)5 km/h', 'B)7 km/h', 'C)10 km/h', 'D)15 km/h', 'E)25 km/h']
a, b = symbols('a b')
equation1 = Eq(42 / a, 42 / b + 2)
equation2 = Eq(42 / (2 * a), 42 / b - 1)
solution = solve_it([equation1, equation2], [a, b])
ans = solution[a]
print(ans)



#Question:Let the number which when multiplied by 15 is increased by 196.
#Answer options:['A)14', 'B)20', 'C)26', 'D)28', 'E)30']
x = symbols('x')
equation = Eq(15*x, x + 196)
solution = solve_it(equation, x)
ans=solution[x]
print(ans)



#Question:A certain sum of money at simple interest amounted Rs.980 in 3 years at 5% per annum, find the sum?
#Answer options:['A)867', 'B)855', 'C)299', 'D)852', 'E)903']
P = symbols('P')
A = 980  
R = 5
T = 3
simple_interest = (P * R * T) / 100
equation = Eq(P + simple_interest, A)
solution = solve_it(equation, P)
ans=solution[P]
print(ans)




"""



        # if index==0 or index==2:
        #     continue
        # if(index==1):
        #     break

        #prompt = prompt_for_manual_prediction(row, exemplars)
        
        # #chain_answer = safe_completion(prompt=prompt, max_tokens=_MAX_TOKENS, stop=stop_signal, temp=0.0, logprobs=5)
       
        # answer=""
        #answer=in_context_manual_prediction(row,exemplars)
        #prompt=prompt+prompt_for_manual_prediction(row,exemplars)
        #print("prompt",prompt)

        outputs=in_context_manual_prediction(row,exemplars,0)
        #print(outputs[0])
        d={}
        for i in outputs:
            i=i.split("\n\n")
            #print(i)
            #print(i[0]+"\n\n")
            output=i[6]
            #print(output+"\n\n\n\n")
            output=output.split('\n')
            for i in range(len(output)):
                if '=' in output[i]:
                    output[i]=output[i].strip()+'\n'
                else:
                    output[i]=output[i]+'\n'

            #print(outputs)
            output=["from tool import simplify_ans, parse_api_result, safe_execute,solve_it\n","import sympy\n",
    "from sympy.solvers import solve\n",
    "from sympy import symbols,Eq\n"]+output
            #print(outputs)
            file1 = open('myfile.py', 'w')
            file1.writelines(output)
            try:
                with subprocess_lock:
                    command=["python", "myfile.py"]
                    result = subprocess.run(command, stdout=subprocess.PIPE, text=True,check=True)
                    ans=result.stdout
                    ans=ans.split('\n')
                    if len(ans)>1:
                        ans=ans[1].strip()
                    else:
                        ans=''
                    

                    try:
                        ans = round(float(ans), 2)
                    except Exception:
                        ans = str(ans)

                    #print("ans:",ans)
                    # if ans!=None and ans!='':
                    if ans in d:
                        d[ans] += 1
                    else:
                        d[ans] = 1
                    time.sleep(1)
            except subprocess.CalledProcessError as e:
        # Handle subprocess errors
                print(f"Subprocess failed with return code {e.returncode}.")
                print("Error output from the subprocess:")
                print(e.stderr)

            except Exception as e:
        # Handle other exceptions that might occur
                print(f"An error occurred: {e}")
                
        print(d)
        n = sorted(d.items(), key=lambda x:x[1], reverse=True)
        answer=''
        if len(n)==1:
            answer=n[0][0]
        if (answer==None or answer=='') and len(n)>1:
            answer=n[1][0]
        if (answer==None or answer=='') and len(n)>2:
            answer=n[2][0]
        option_prompt="""
Find the closest options based on the question and prediction.

Question: A company produces 420 units of a particular computer component every month, at a production cost to the company of $110 per component, and sells all of the components by the end of each month. What is the minimum selling price per component that will guarantee that the yearly profit (revenue from sales minus production costs) will be at least $626,400 ?
Options: ['A)226', 'B)230', 'C)240', 'D)260', 'E)280']
Prediction: 234.28571428571428
Closest Option: B

Question: In how many ways can the letters of the word "PROBLEC" be rearranged to make 7 letter words such that none of the letters repeat?
Options: ['A)2!', 'B)3!', 'C)7!', 'D)8!', 'E)9!']
Prediction: 5040
Closest Option: C

Question: Find the total no. of distinct bike no.'s that can beformed using 2 letters followed by 2 no.'s. How many letters need to be distinct?
Options: ["A)74453", "B)64543", "C)74325", "D)65000", "E)97656"]
Prediction = 67600
Closest Option: D

Question: A wire in the shape of rectangle of length 27 cm and breadth 17 cm is rebent to form a square. What will be the measure of each side?
Options: ['A)9', 'B)11', 'C)22', 'D)25', 'E)31']
Prediction = [-21.42428528562855, 21.42428528562855]
Closest Option: C

Question: A point on the edge of a fan blade that is rotating in a plane 10 centimeters from the center of the fan. What is the distance traveled, in centimeters, by this point after 30 seconds when the fan runs at the rate of 300 revolutions per minutes?
Options: ['A)750pi', 'B)1500pi', 'C)1875pi', 'D)3000pi', 'E)7500pi']
Prediction: 9424.77
Closest Option: D


"""
        question=row["question"]
        options=row["options"]
        option_prompt += f'\nQuestion: {question}\nOptions: {options}\nPrediction: {answer}\nClosest Option:' 
        option=in_context_manual_prediction(row,exemplars,1)
        print("option:",option)
        

        

        # option=option[0].strip()
        # option=option[0]
        # print("the option is:",option)
        ans_list = []
        for tmp in option:
            # tmp_list.append(compare_llm_outputs(user_query))
            # tmp = compare_llm_outputs(user_query)
            # print(tmp)
            ans = ""
            tmp=tmp.split("Closest Option: ")
                # ans = ans.split("\n")[0]
            # ans = ans.replace("$", "")
            # ans = ans.strip()
            #print(tmp[0])
            if(len(tmp)>0):
                ans_list.append(tmp[6][0])

        #print(ans_list)

        d1 = {}
        for i in ans_list:
            if i in d1:
                d1[i] += 1
            else:
                d1[i] = 1
        print(d1)
        n1 = sorted(d1.items(), key=lambda x:x[1], reverse=True)
        answer = n1[0][0]
        if answer=="" and len(n1)>1: answer = n1[1][0]
        option=answer


        




        
        

        
        question_df['question'].append(row["question"])
        question_df["answers"].append(outputs)
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("output/aquarat/Llama_static_aquarat_question_answer_latest_pot.tsv",sep="\t",index=False)


        ground_truth = row["correct"]

    #     print("\nanswer:",answer)
    #     print("ground_truth:",ground_truth)

        if option==ground_truth:
            matches+=1
        else:
            mismatches+=1
        print("Accuracy till now:", matches/(matches+mismatches))

    print("EM", matches/(matches+mismatches))

    return final_questions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    # ADVHOTPOT train dataset
    train_data = read_AHOTPOT_train_data()

    # ADVHOTPOT test dataset
    test_data = read_AHOTPOT_test_data()

    final_df = get_open_source_completions(test_data, train_data)
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()
