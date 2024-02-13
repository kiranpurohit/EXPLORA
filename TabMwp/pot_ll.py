import numpy as np
import random
import pandas as pd
import torch
import json
import openai
import func_timeout
from tqdm import tqdm
from math import *
from time import sleep
from tenacity import retry, stop_after_attempt, wait_random_exponential

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#####################################################################################################
system_message = """The following is a conversation between a Human and an AI Assistant.
The assistant is helpful, respectful and honest, and it always answers as helpfully as possible, while being safe.
The Assistant's answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that the Assistant's responses are socially unbiased and positive in nature.
If a question by the human does not make any sense, or is not factually coherent, the Assistant should explain why instead of answering something not correct.
If the Assistant does not know the answer to a question, please don't share false information.
####

"""

api_keys = ["EMPTY", "EMPTY", "EMPTY"]#, "EMPTY"]
endpoint_urls = ["https://6621-203-110-242-13.ngrok-free.app"]# https://68e6-14-139-109-7.ngrok-free.app/ "https://9451-130-75-87-254.ngrok-free.app", "https://7a6a-130-75-87-254.ngrok-free.app"]#, "https://akdeniz27-llama-2-70b-chat-hf-with-easyllm.hf.space/"]
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

# Gen response from API
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(prompt, api_key, endpoint_url, hard_code_exception=False):

    max_tokens=200
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

# Self consistency on 10 generated answers
def self_con(tmp_list):

    n = len(tmp_list)
    for i in range(0, n):
        if type(tmp_list[i])==float:
            tmp_list[i] = round(tmp_list[i], 2)
        tmp_list[i] = str(tmp_list[i])
    
    d = {}
    for i in tmp_list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

def llm_output(user_query, hard_code_exception=False):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)
    # res = self_con(results)
    # answer = res[0][0]
    # if answer=="" and len(res)>1: answer = res[1][0]
    return results


def safe_execute(codes:str):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            return locals_.get('ans', None)
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(5, execute, args=(codes))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans

######################################################################################################

def prompt_for_manual_prediction(ex):

    prompt = """
There is no user input required for any question in Python Code.

Read the following table and then write Python code to answer a question:

Day | Number of tickets
Monday | 36
Tuesday | 43
Wednesday | 46
Thursday | 59
Friday | 37
Saturday | 46
Sunday | 51

Question: The transportation company tracked the number of train tickets sold in the past 7 days. What is the range of the numbers?
# Python Code, return ans
tickets = [36, 43, 46, 59, 37, 46, 51]
min_tickets = min(tickets)
max_tickets = max(tickets)
range_tickets = max_tickets - min_tickets
ans = range_tickets


Read the following table and then write Python code to answer a question:

Stem | Leaf 
4 | 2, 7, 9, 9, 9
5 | 1
6 | 9
7 | 2, 2, 3, 3, 5
8 | 
9 | 0

Question: A pottery factory kept track of the number of broken plates per shipment last week. How many shipments had exactly 73 broken plates?
# Python Code, return ans
broken_plates = [2, 7, 9, 9, 9, 1, 9, 2, 2, 3, 3, 5, 0]
broken_plates = sorted(broken_plates)
count = 0
for i in range(2, len(broken_plates)):
    if broken_plates[i] - broken_plates[i-1] == 73:
        count += 1
ans = count


Read the following table and then write Python code to answer a question:

purple and red clay bead | $0.02
small pink bead | $0.04
pearl bead | $0.07
round silver bead | $0.01
brown cat's eye bead | $0.08
orange glass bead | $0.07

Question: Kylie has $0.05. Does she have enough to buy a small pink bead and a purple and red clay bead? Please select from the following options: ['yes', 'no']
# Python Code, return ans
small_pink_bead = 0.04
purple_and_red_clay_bead = 0.02
total_money = 0.05
if total_money > small_pink_bead + purple_and_red_clay_bead:
    ans = "yes"
else:
    ans = "no"


Read the following table then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$165 | 17,900 | 6,400
$345 | 15,100 | 8,900
$525 | 12,300 | 11,400
$705 | 9,500 | 13,900
$885 | 6,700 | 16,400
Question: Look at the table. Then answer the question. At a price of $885, is there a shortage or a surplus? Please select from the following options: ['shortage', 'surplus']
# Python Code, return ans
price_885 = 885
quantity_demanded_price_885 = 6700
quantity_supplied_price_885 = 16400
if quantity_demanded_price_885 > quantity_supplied_price_885:
    ans = "shortage"
else:
    ans = "surplus"


Read the following table then write Python code to answer a question:

Chickenville | 3:00 A.M. | 12:00 P.M. | 3:30 P.M.
Floral Gardens | 3:45 A.M. | 12:45 P.M. | 4:15 P.M.
Pleasant River Campground | 4:45 A.M. | 1:45 P.M. | 5:15 P.M.
Happy Cow Farm | 5:15 A.M. | 2:15 P.M. | 5:45 P.M.
Rocky Ravine Town | 5:45 A.M. | 2:45 P.M. | 6:15 P.M.

Question: Look at the following schedule. Doug just missed the 3.00 A.M. train at Chickenville. How long does he have to wait until the next train? Please select from the following options: ['6 hours and 30 minutes', '6 hours and 45 minutes', '8 hours and 45 minutes', '9 hours']
# Python Code, return ans
schedule = {
"Chickenville": {"3:00 A.M.": "3:45 A.M."},
"Floral Gardens": {"3:45 A.M.": "4:15 P.M."},
"Pleasant River Campground": {"4:45 A.M.": "1:45 P.M."},
"Happy Cow Farm": {"5:15 A.M.": "5:45 P.M."},
"Rocky Ravine Town": {"5:45 A.M.": "6:15 P.M."}
}
train = schedule["Chickenville"]["3:00 A.M."]
next_train = schedule["Chickenville"]["4:45 A.M."]
waiting_time = next_train - train
if waiting_time < 60:
    ans = "6 hours and 30 minutes"
elif waiting_time < 180:
    ans = "6 hours and 45 minutes"
else:
    ans = "8 hours and 45 minutes"
"""
    prompt += "Read the following table then write Python code to answer the question:\nTable:\n" + ex["table"] + "\nQuestion:" + ex["question"]
    if type(ex["choices"])==str:
        prompt += "Please select from the following options:"+ex["choices"]
    prompt += "\n# Python Code, return ans\n"
    
    return prompt


def get_open_source_completions(test_data, data):

    # stop_signal = "\n\n"
    matches = 0
    mismatches = 0

    question_df = {"question":[],"answers":[]}
    
    index=0
    acc_records = []
    exnum = 1

    # test_data = test_data[:10]
    # for index, row in test_data.iterrows():
    # codes = []
    for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Generating"):

        prompt = prompt_for_manual_prediction(row)
        answer_list = llm_output(prompt)
        ans_list = []
        for i in answer_list:
            code = i.split("\n\n")[0]
            ans_list.append(code)
            # print(code)
            
        # print(ans_list)
        # Getting code and output
        tmp_list = []
        for i in ans_list:
            code = i.split("\n\n")[0]
            # print(code)
            try:
                exec(code)
                tmp_list.append(locals()['ans'])
                # ans = locals_.get('ans', None)
            except: pass
            # t = safe_execute(str(code))
            # if ans!=None: tmp_list.append(ans)
        
        print(tmp_list)
        res = self_con(tmp_list)
        print(res)
        if len(res)>0: answer = res[0][0]
        else: answer = None

        # question_df['question'].append(row["question"])
        # question_df["answers"].append(answer)
        # final_questions = pd.DataFrame(question_df)
        # final_questions.to_csv("output/mistral_pot_question_answer.tsv",sep="\t",index=False)

        ground_truth = row["answer"]

        if '/' in ground_truth:
            ground_truth = int(ground_truth.split('/')[0]) / int(ground_truth.split('/')[1])
        elif '%' in ground_truth:
            ground_truth = float(ground_truth.split('%')[0]) / 100
        if type(ground_truth)==float: ground_truth = str(round(ground_truth, 2))
        else: ground_truth = str(ground_truth)

        print("\nGen Answer:", answer)
        print("Ground Truth:", ground_truth)

        if answer!=None and (ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower()):
            matches+=1
        else:
            mismatches+=1

        print("Accuracy after", exnum, "ex:", matches/(matches+mismatches))
        exnum += 1
        # sleep(5)

    print("EM:", matches/(matches+mismatches))

    final_questions = 0
    return final_questions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    # ADVHOTPOT train dataset
    # train_data = read_AHOTPOT_train_data()
    train_data = pd.read_json('problems_train.json', orient='index')

    # ADVHOTPOT test dataset
    # test_data = read_AHOTPOT_test_data()
    test_data = pd.read_json('problems_dev.json', orient='index')

    final_df = get_open_source_completions(test_data, train_data)
    # print(final_df)
    print("End of execution")


if __name__=='__main__':
    test_few_shot_prediction()