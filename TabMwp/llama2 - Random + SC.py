# -*- coding: utf-8 -*-
"""Copy of Open-Source-LLMs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10SYnCPLvNjwGS59RLkZTrzgBBS061p6x
"""

!pip install openai==0.28 tenacity

# import gradio as gr
# from gradio_client import Client
# !pip install openai==0.28 tenacity
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json

system_message = """The following is a conversation between a Human and an AI Assistant.
The assistant is helpful, respectful and honest, and it always answers as helpfully as possible, while being safe.
The Assistant's answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that the Assistant's responses are socially unbiased and positive in nature.
If a question by the human does not make any sense, or is not factually coherent, the Assistant should explain why instead of answering something not correct.
If the Assistant does not know the answer to a question, please don't share false information.
####

"""

# Replace with your API keys and endpoint URLs
#api_keys = ["<API_KEY_1>", "<API_KEY_2>", "<API_KEY_3>"]
#endpoint_urls = ["<ENDPOINT_URL_1>", "<ENDPOINT_URL_2>", "<ENDPOINT_URL_3>"]
#llm_names = ["LLM 1", "LLM 2", "LLM 3"]

api_keys = ["EMPTY", "EMPTY", "EMPTY"]#, "EMPTY"]
endpoint_urls = ["https://d06d-130-75-87-254.ngrok-free.app"]# "https://9451-130-75-87-254.ngrok-free.app", "https://7a6a-130-75-87-254.ngrok-free.app"]#, "https://akdeniz27-llama-2-70b-chat-hf-with-easyllm.hf.space/"]
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
        max_tokens=200,
    )
    out_text = []
    for x in range(0, 10):
        out_text.append(res['choices'][x]['text'].strip())
    return out_text
    # return res['choices'][0]['message']['content'].strip()

# Define a function for Gradio to call
def compare_llm_outputs(user_query, hard_code_exception=False):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)

    return results

llm_names

from google.colab import drive
drive.mount('/content/drive')

dev_set1 = open("drive/MyDrive/tabmwp/problems_dev.json") #7965
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

            """+"Generate the answer now:\n Table:\n" + ex["table"] + "\nQuestion:" + ex["question"]
    # tmp_list = []
    tmp_list = compare_llm_outputs(user_query)
    print(len(tmp_list))
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
        ans = ans.strip()
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
    # print(n[0][0])
    '''
    answer = ""
    if len(tmp_list[0].split("The answer is:"))>1:
        answer = tmp_list[0].split("The answer is:")[1]
        answer = answer.split("\n")[0]
    answer = answer.replace("$", "")
    '''
    answer = n[0][0]
    if answer=="" and len(n)>1: answer = n[1][0]
    print("\nAnswer: ", answer)
    print("GT: ", ex["answer"])
    ground_truth = ex["answer"]
    if ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower():
        matches+=1
    else:
        mismatches+=1
    print(matches)
    # if exnum%1000==0:
    #     counts.append(matches)
    # exnum += 1
    # print("Counts: ", counts)

for i in tmp_list:
  print(i)
  print("*****************")