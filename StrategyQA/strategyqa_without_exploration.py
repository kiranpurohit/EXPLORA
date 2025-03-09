import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json
import torch
import random
import pickle 
from sklearn.model_selection import train_test_split
import transformers
# import os
# from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2TokenizerFast, BertTokenizer, BertModel, logging

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
logging.set_verbosity_error()

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

###########################################################################################
# model_name = "google/flan-t5-large"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
# # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
# model = model.to(device)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# def get_completion(msg_in):

#     messages=[{
#                 "role": "user",
#                 "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic.Do not generate examples in your answer",
#             }]
#     text={"role": "assistant", "content":""" Follow given examples and solve the Test Question at end in similar manner by decomposing the original questions
#          Examples:{}""".format(msg_in)}
#     messages.append(text)

#     prompt = msg_in
#     #prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
#     print("OUTPUTS", outputs)
#     out_text = []
#     for x in range(0, 10):
#         out_text.append(outputs[x]["generated_text"])
#     return out_text


#####################################################################################################
system_message = """The following is a conversation between a Human and an AI Assistant.
The assistant is helpful, respectful and honest, and it always answers as helpfully as possible, while being safe.
The Assistant's answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that the Assistant's responses are socially unbiased and positive in nature.
If a question by the human does not make any sense, or is not factually coherent, the Assistant should explain why instead of answering something not correct.
If the Assistant does not know the answer to a question, please don't share false information.
####

"""
api_keys = ["EMPTY"]
endpoint_urls = ["https://827b-130-75-152-24.ngrok-free.app"]
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



###Gen response from API
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(prompt, api_key, endpoint_url, hard_code_exception=False):

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
        temperature=0.3,
        top_k=10,
        top_p=1.0,
        n=10,
        max_tokens=256,
    )
    out_text = res['choices'][0]['text'].strip()
    # out_text = []
    # for x in range(0, 10):
    #     out_text.append(res['choices'][x]['text'].strip())
    return out_text
#####################################################################################################




def llm_output(user_query, hard_code_exception=False):
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)
    #results = get_completion(user_query)
    return results


def read_strategyqa():
    with open('../data/StrategyQA/strategyqa_train.json', encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    for d in data:

        if d["answer"]:
            answer = "Yes"
        else:
            answer = "No"

        ex = {
            "id": d["qid"],
            "question": d["question"].lstrip(),
            "answer": answer,
            "facts": d["facts"],
            "decomposition" : d["decomposition"]
        }
        examples.append(ex)

    examples = pd.DataFrame(examples)

    return examples


def get_embeddings1(list1):
    doc_embeddings=[]
    for i in list1:
        inputs_sentence1 = tokenizer_bert(i, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs_sentence1 = model_bert(**inputs_sentence1)
        embedding_sentence1 = outputs_sentence1.last_hidden_state.mean(dim=1).numpy()
        embedding_sentence1=embedding_sentence1.tolist()
        doc_embeddings.append(embedding_sentence1[0])
    return doc_embeddings


def get_prompt(ex):
    count=0
    s = "\n\n"
    for p in ex["facts"]:
        if count==0:
            s += "Facts:" + p
        else:
            s += p
        count+=1
    # s.join(["{}".format(p) for p in ex["facts"]])
    s += "\nQuestion:" + ex["question"]
    s += "\nAnswer:\n"
    for i, p in enumerate(ex["decomposition"]):
        s += "Sub-question {}: {}\n".format(i+1,p)
    # s.join(["Sub-question {}: {}\n".format(i+1,p) for i, p in enumerate(ex["decomposition"])])
    s += "The answer is:" + ex["answer"]
    return s



def get_prompt_test(ex):
    count=0
    s = "\n"
    for p in ex["facts"]:
        if count==0:
            s += "Facts:" + p
        else:
            s += p
        count+=1
    # s.join(["{}".format(p) for p in ex["facts"]])
    s += "\nQuestion:" + ex["question"]
    return s

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prompt_for_manual_prediction(ex, shots):
    #print(shots)
    prompt = """\n\nFollow the given examples that use the facts to answer a question by decomposing into sub questions first and then predicting the final answer as "Yes" or "No" only."""  
    # prompt = "Follow given examples and solve the given Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"
    for index, s in shots.iterrows():
        prompt += get_prompt(s)

    prompt += "\n\nGenerate the answer for the test example now:\n"
    # prompt += "\n\nFollowing the given examples generate step by step reasoning strictly preceded by Answer and generate Final Answer preceded by Final Answer: for the below question.\n\n" 
    prompt += get_prompt_test(ex)
    
    return prompt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_avg_error(exemplars_set, val_data):
    stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="predicting"):
        matches = 0
        mismatches =0
        acc_records = []
        for index, row in val_data.iterrows():
            prompt = prompt_for_manual_prediction(row, exemplars)
            #chain_answer = safe_completion(prompt=prompt, max_tokens=_MAX_TOKENS, stop=stop_signal, temp=0.0, logprobs=5)

            tmp = llm_output(prompt)

            answer = ""
            if len(tmp.split("The answer is:"))>1:
                answer = tmp.split("The answer is:")[1]
                answer = answer.split("\n")[0]

            # print("\nAnswer: ", answer)
            # print("GT: ", row["answer"])
            ground_truth = row["answer"]

            if ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower():
              matches+=1
            else:
              mismatches+=1

        error.append(mismatches/(matches+mismatches))

    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_error_indicator(exemplars_set, val_data):
    stop_signal = "\n\n"
    error=[]

    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="predicting"):

        for index, row in val_data.iterrows():
            prompt = prompt_for_manual_prediction(row, exemplars)
            # print(prompt)
            tmp = llm_output(prompt)
            
            answer = ""
            if len(tmp.split("The answer is:"))>1:
                answer = tmp.split("The answer is:")[1]
                answer = answer.split("\n")[0]

            # print("\nAnswer: ", answer)
            # print("GT: ", row["answer"])
            ground_truth = row["answer"]
            if ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower():
                loss=0
            else:
                loss=1

            error.append(loss)

    return error


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#################################################################################################
#************************************************************************************************

def static_subset_selection(val_data, train_data, k, test_data):

    val_data = val_data[:20]

    # dbfile3 = open('../data/AQUA_RAT/aquarat_train_emb.pkl', 'rb')    
    # train_emb = pickle.load(dbfile3)

    #Calculate embeddings for all validation questions
    val_emb = get_embeddings1(val_data["question"].tolist())
    with open("../data/StrategyQA/strategy_val_emb.pkl", 'wb') as f:
            pickle.dump(val_emb, f)
    print("Valiation embeddings calculated")


    train_emb = get_embeddings1(train_data["question"].tolist())
    with open("../data/StrategyQA/strategy_train_emb.pkl", 'wb') as f:
            pickle.dump(train_emb, f)
    print("Train embeddings calculated")

    # k-means clustering on train_data with k=5
    kmeans = KMeans(n_clusters=5, random_state=0).fit(train_emb)
    #print(kmeans.labels_)
    # Make clusters of train_data with same cluster label
    train_data['cluster'] = kmeans.labels_

    # # Do stratified sampling from train_data based on the first word of the question
    # train_data['w1'] = train_data['question'].str.split().str[0]

    # Create index column in train_data using arange
    train_data['index'] = np.arange(len(train_data))

    # Create L=_ subsets of size k total, with each group having k/num_gr examples
    num_gr = len(train_data['cluster'].unique())
    L = []
    L_indices = []

    # Initialize L, 40 random set of subsets from train_data 
    for idx1 in range(40):
        subset = []
        for name, group in train_data.groupby('cluster'):
            # subset.append(group.sample(k//num_gr))   
            subset.append(group.sample(1))   
        subsets = pd.concat(subset)
        L.append(subsets)
        L_indices.append(subsets['index'].tolist())

    
    # Calculate the similarity matrix, Eij = cosine similarity between exemplar x_i=train_data and test example u_j=val_data
    E_val = cosine_similarity(train_emb, val_emb)

    # Calculate Loss(Y,S) for all S in L
    LLM_loss = LLM_error_indicator(L, val_data)


    
    # Storing the indices of the subsets in L
    L_indices = []
    for i in range(len(L)):
        L_indices.append(L[i]['index'].tolist())


    
    # fill W = E_{ij} for all i in L
    l = len(L)
    L_val = np.zeros((l*len(val_emb), len(train_emb)))
    for u in range(l):
        for i in range(len(train_emb)):
            if i in L_indices[u]:
                L_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]



    # min_alpha_i (LLM_loss-L_val*alpha)^2
    alpha = np.linalg.lstsq(L_val, LLM_loss, rcond=None)[0]
    print("\nalpha shape ",alpha.shape)
    print("\nalpha ",alpha)


    # Calculate the best subset S_best ∈ L that minimizes the approximate loss
    mul1 = np.matmul(L_val, alpha)
    mul_new1 = np.reshape(mul1, (len(L),-1))
    S_best_ind = np.argmin(np.sum(mul_new1, axis=1))
    S_best = L[S_best_ind]


    return S_best


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_open_source_completions(data):


    print("started running:")

    train_num = 1800 # 490 test ex
    labels = [x["answer"] for index, x in data.iterrows()]
    train_split, test_data, _, _ = train_test_split(data, labels, train_size=train_num, stratify=labels, random_state=7)

    #-------------
    val_size = 20
    #-------------
    labels = [x["answer"] for index, x in train_split.iterrows()]
    val_data, train_data, _, _ = train_test_split(train_split, labels, train_size=val_size, stratify=labels, random_state=7)
    print("val_data size = ",len(val_data))
    print("train_data size = ",len(train_data))

    exemplars = static_subset_selection(val_data, train_data, 5, test_data)

    print(exemplars)

    exemplars.to_csv("output/without_exploration_strategyqa.csv")
    
    exit(0)

    return exemplars
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    # train dataset
    train_data = read_strategyqa()

    print("Data read successfully")

    final_df = get_open_source_completions(train_data)
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()