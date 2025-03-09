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
endpoint_urls = ["https://a37017d2bb74.ngrok.app"]
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



    # Initialize U, 10 random set of subsets from L 
    ind_L = np.arange(0,len(L)).tolist()
    
    ind_total = random.sample(ind_L, 15)
    ind_U = ind_total[:10]
    ind_V = ind_total[10:]

    ind_L_minus_U = [x for x in ind_L if x not in ind_U]    

    U = []
    for i in ind_U:
        U.append(L[i])
    V = []
    for i in ind_V:
        V.append(L[i])
        
    L_minus_U = []
    for i in ind_L_minus_U:
        L_minus_U.append(L[i])

    
    # Calculate the similarity matrix, Eij = cosine similarity between exemplar x_i=train_data and test example u_j=val_data
    E_val = cosine_similarity(train_emb, val_emb)
    #E_test = cosine_similarity(train_emb, test_emb)

    # Calculate Loss(Y,S) for all S in U
    LLM_loss = LLM_error_indicator(U, val_data)
    #LLM_loss_on_test_data = LLM_error_indicator(U, test_data)
    # LLM_loss_on_L_minus_U = LLM_error_indicator(L_minus_U, val_data)
    LLM_loss_on_V = LLM_error_indicator(V, val_data)

    


    approx_error_on_U = []
    approx_error_on_U_after_update = []
    # approx_error_on_L_minus_U = []
    # approx_error_on_L_minus_U_after_update = []
    approx_error_on_V = []
    approx_error_on_V_after_update = []
    
    # approx_error_on_U_on_test_data = []
    # approx_error_on_U_after_update_on_test_data = []
    # approx_error_on_L_minus_U_on_test_data = []
    # approx_error_on_L_minus_U_after_update_on_test_data = []
    # approx_error_on_V_on_test_data = []
    # approx_error_on_V_after_update_on_test_data = []

    approx_value_on_U = []
    approx_value_on_U_after_update = []
    # approx_value_on_L_minus_U = []
    # approx_value_on_L_minus_U_after_update = []
    approx_value_on_V = []
    approx_value_on_V_after_update = []
    
    # approx_value_on_U_on_test_data = []
    # approx_value_on_U_after_update_on_test_data = []
    # approx_value_on_L_minus_U_on_test_data = []
    # approx_value_on_L_minus_U_after_update_on_test_data = []
    # approx_value_on_V_on_test_data = []
    # approx_value_on_V_after_update_on_test_data = []

    LLM_loss_on_val = []
    avg_LLM_loss_on_val = []
    min_LLM_loss_on_val = []
    max_LLM_loss_on_val = []
    LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss, (len(U),-1))
    LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
    LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
    avg_LLM_loss_on_val.append(LLM_loss_on_val_avg.tolist()) 
    min_LLM_loss_on_val.append(LLM_loss_on_val_min.tolist())
    max_LLM_loss_on_val.append(LLM_loss_on_val_max.tolist())
    print("\n********* LLM LOSS ON U FOR VALIDATION DATA *********")
    print("\nLLM_loss_on_val ",LLM_loss_on_val)
    print("AVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_val)
    print("MIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_val)
    print("MAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_val)
    #===============================================================================



    #LLM_loss_on_test = []
    # avg_LLM_loss = []
    # min_LLM_loss = []
    # max_LLM_loss = []
    # LLM_loss_for_each_subset = np.reshape(LLM_loss_on_test_data, (len(U),-1))
    # LLM_loss_for_each_subset = np.average(LLM_loss_for_each_subset, axis=1)
    # LLM_loss_avg = np.average(LLM_loss_for_each_subset)
    # LLM_loss_min = np.min(LLM_loss_for_each_subset)
    # LLM_loss_max = np.max(LLM_loss_for_each_subset)
    # #LLM_loss_on_test.append(LLM_loss_for_each_subset.tolist())
    # avg_LLM_loss.append(LLM_loss_avg.tolist()) 
    # min_LLM_loss.append(LLM_loss_min.tolist())
    # max_LLM_loss.append(LLM_loss_max.tolist())
    # print("\n********* LLM LOSS ON U FOR TEST DATA *********")
    # print("\nLLM_loss_on_test ",LLM_loss_on_test)
    # print("AVG_LLM_loss_on_TEST_data ",avg_LLM_loss)
    # print("MIN_LLM_loss_on_TEST_data ",min_LLM_loss)
    # print("MAX_LLM_loss_on_TEST_data ",max_LLM_loss)
    #===============================================================================

   

    # LLM_loss_on_L_minus_U_on_val = []
    # avg_LLM_loss_on_L_minus_U_on_val = []
    # min_LLM_loss_on_L_minus_U_on_val = []
    # max_LLM_loss_on_L_minus_U_on_val = []
    # LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss_on_L_minus_U, (len(L_minus_U),-1))
    # LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
    # LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
    # LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
    # LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
    # LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
    # avg_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_avg.tolist()) 
    # min_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_min.tolist())
    # max_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_max.tolist())
    # print("\n********* LLM LOSS ON L_minus_U FOR VALIDATION DATA *********")
    # print("\nLLM_loss_on_val ",LLM_loss_on_L_minus_U_on_val)
    # print("AVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_L_minus_U_on_val)
    # print("MIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_L_minus_U_on_val)
    # print("MAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_L_minus_U_on_val)
    # #===============================================================================

    # LLM_loss_on_L_minus_U_on_test_data = LLM_error_indicator(L_minus_U, test_data)
    # LLM_loss_on_L_minus_U_on_test = []
    # avg_LLM_loss_on_L_minus_U = []
    # min_LLM_loss_on_L_minus_U = []
    # max_LLM_loss_on_L_minus_U = []
    # LLM_loss_for_each_subset = np.reshape(LLM_loss_on_L_minus_U_on_test_data, (len(L_minus_U),-1))
    # LLM_loss_for_each_subset = np.average(LLM_loss_for_each_subset, axis=1)
    # LLM_loss_avg = np.average(LLM_loss_for_each_subset)
    # LLM_loss_min = np.min(LLM_loss_for_each_subset)
    # LLM_loss_max = np.max(LLM_loss_for_each_subset)
    # LLM_loss_on_L_minus_U_on_test.append(LLM_loss_for_each_subset.tolist())
    # avg_LLM_loss_on_L_minus_U.append(LLM_loss_avg.tolist()) 
    # min_LLM_loss_on_L_minus_U.append(LLM_loss_min.tolist())
    # max_LLM_loss_on_L_minus_U.append(LLM_loss_max.tolist())
    # print("\n********* LLM LOSS ON L_minus_U FOR TEST DATA *********")
    # print("\nLLM_loss_on_test ",LLM_loss_on_L_minus_U_on_test)
    # print("AVG_LLM_loss_on_TEST_data ",avg_LLM_loss_on_L_minus_U)
    # print("MIN_LLM_loss_on_TEST_data ",min_LLM_loss_on_L_minus_U)
    # print("MAX_LLM_loss_on_TEST_data ",max_LLM_loss_on_L_minus_U)
    #===============================================================================


    LLM_loss_on_V_on_val = []
    avg_LLM_loss_on_V_on_val = []
    min_LLM_loss_on_V_on_val = []
    max_LLM_loss_on_V_on_val = []
    LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss_on_V, (len(V),-1))
    LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
    LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_V_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
    avg_LLM_loss_on_V_on_val.append(LLM_loss_on_val_avg.tolist()) 
    min_LLM_loss_on_V_on_val.append(LLM_loss_on_val_min.tolist())
    max_LLM_loss_on_V_on_val.append(LLM_loss_on_val_max.tolist())
    print("\n********* LLM LOSS ON V FOR VALIDATION DATA *********")
    print("\nLLM_loss_on_val ",LLM_loss_on_V_on_val)
    print("AVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_V_on_val)
    print("MIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_V_on_val)
    print("MAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_V_on_val)


    #===============================================================================
    # Calculate the pairwise overlap between the subsets in U
    overlaps=[]
    for i in range(len(U)):
        inner_overlaps=[]
        for j in range(len(U)):
            if i!=j:
                overlap=0
                for index_j, s_i in U[i].iterrows():
                    for index_j, s_j in U[j].iterrows():
                        if s_i["question"].lower() in s_j["question"].lower() or s_j["question"].lower() in s_i["question"].lower():
                            overlap+=1
                inner_overlaps.append(overlap)
        overlaps.append(inner_overlaps)
            
    print("\noverlaps ",overlaps)
    print("len overlaps ",len(overlaps))

    overlap_for_subset = []
    avg_overlap = []
    min_overlap = []
    max_overlap = []
    overlap_for_each_subset = np.average(overlaps, axis=1)
    overlap_avg = np.average(overlap_for_each_subset)
    overlap_min = np.min(overlap_for_each_subset)
    overlap_max = np.max(overlap_for_each_subset)
    overlap_for_subset.append(overlap_for_each_subset.tolist())
    avg_overlap.append(overlap_avg.tolist()) 
    min_overlap.append(overlap_min.tolist())
    max_overlap.append(overlap_max.tolist())
    print("\n********* PAIRWISE OVERLAP *********")
    print("\noverlap_for_subset ",overlap_for_subset)
    print("\nAVG_overlap ",avg_overlap)
    print("MIN_overlap ",min_overlap)
    print("MAX_overlap ",max_overlap)
    #===============================================================================








    ################################################################################
    # Storing the indices of the subsets in U_t
    U_indices = []
    for i in range(len(U)):
        U_indices.append(U[i]['index'].tolist())

    V_indices = []
    for i in range(len(V)):
        V_indices.append(V[i]['index'].tolist())

    L_minus_U_indices = []
    for i in range(len(L_minus_U)):
        L_minus_U_indices.append(L_minus_U[i]['index'].tolist())


    ################################################################################
    # fill W = E_{ij} for all i in U_t
    l = len(U)
    W_val = np.zeros((l*len(val_emb), len(train_emb)))
    #W_test = np.zeros((l*len(test_emb), len(train_emb)))
    for u in range(l):
        for i in range(len(train_emb)):
            if i in U_indices[u]:
                W_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                #W_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


    ################################################################################
    # fill V = E_{ij} for all i in V_t
    v = len(V)
    V_val = np.zeros((v*len(val_emb), len(train_emb)))
    #V_test = np.zeros((v*len(test_emb), len(train_emb)))
    for u in range(v):
        for i in range(len(train_emb)):
            if i in V_indices[u]:
                V_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                #V_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]



    ##############################################################################
    # make X = E_{ij} for all i in L-U_t 
    L_minus_U_len = len(L)-len(U)
    X_val = np.zeros((L_minus_U_len*len(val_emb), len(train_emb)))
    # X_test = np.zeros((L_minus_U_len*len(test_emb), len(train_emb)))
    for u in range(L_minus_U_len):
        for i in range(len(train_emb)):
            if i in L_minus_U_indices[u]:
                X_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                # X_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]








    # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WHILE LOOP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    t=0
    while t<10:#t=10

        ################################################################################
        # min_alpha_i (LLM_loss-W*alpha)^2
        #alpha = np.linalg.lstsq(W_val, LLM_loss, rcond=None)[0]
        # increase rows of llm loss by appending llm loss of V
        # increase rows of W by appending V
 
        LLM_loss_on_U_V = LLM_loss + LLM_loss_on_V
        W_V_val = np.concatenate((W_val, V_val), axis=0)

        print("\n LLM_loss_on_U_V_len",len(LLM_loss_on_U_V))
        print("\n LLM_loss_on_U_V ",LLM_loss_on_U_V)
        print("\n W_V_val_shape ",W_V_val.shape)
        print("\n W_V_val ",W_V_val)

        alpha = np.linalg.lstsq(W_V_val, LLM_loss_on_U_V, rcond=None)[0]
        # Print alpha

        print("\nalpha shape ",alpha.shape)
        print("\nalpha ",alpha)



        ################################################################################
        # Calculate the worst subset S_worst ∈ U_t that maximizes the approximate loss
        mul1 = np.matmul(W_val, alpha)
        mul_new1 = np.reshape(mul1, (len(U),-1))
        S_worst_ind = np.argmax(np.sum(mul_new1, axis=1))
        S_worst = U[S_worst_ind]

        ##############################################################################
        #mul1_test = np.matmul(W_test, alpha)

        ###############################################################################
        # Calculate the top set S_star ∈ L \ U_t that minimizes the approximate loss
        mul2 = np.matmul(X_val, alpha)
        mul_new2 = np.reshape(mul2, (L_minus_U_len,-1))
        S_star_ind = np.argmin(np.sum(mul_new2, axis=1))
        S_star = L_minus_U[S_star_ind]

        ###############################################################################
        mul3 = np.matmul(V_val, alpha)

        ###############################################################################

        
        #********* APPROX VALUE ON U ON VALIDATION DATA *********
        approx_value_on_U_for_each_subset = np.reshape(mul1, (len(U),-1))
        approx_value_on_U_for_each_subset = np.average(approx_value_on_U_for_each_subset, axis=1)
        approx_value_on_U.append(approx_value_on_U_for_each_subset.tolist())


        #********* APPROX VALUE ON U ON TEST DATA *********
        # approx_value_on_U_on_test_for_each_subset = np.reshape(mul1_test, (-1,len(test_data)))
        # approx_value_on_U_on_test_for_each_subset = np.average(approx_value_on_U_on_test_for_each_subset, axis=1)
        # approx_value_on_U_on_test_data.append(approx_value_on_U_on_test_for_each_subset.tolist())


        # #********* APPROX VALUE ON L-U ON VALIDATION DATA *********
        # approx_value_on_L_minus_U_for_each_subset = np.reshape(mul2, (-1,len(val_data)))
        # approx_value_on_L_minus_U_for_each_subset = np.average(approx_value_on_L_minus_U_for_each_subset, axis=1)
        # approx_value_on_L_minus_U.append(approx_value_on_L_minus_U_for_each_subset.tolist())

        #********* APPROX VALUE ON V ON VALIDATION DATA *********
        approx_value_on_V_for_each_subset = np.reshape(mul3, (-1,len(val_data)))
        approx_value_on_V_for_each_subset = np.average(approx_value_on_V_for_each_subset, axis=1)
        approx_value_on_V.append(approx_value_on_V_for_each_subset.tolist())


        #calculate the approximate error = (LLM_loss-W*alpha) on U
        print("\n*************Approximation error of Validation Data on U ************")
        print("\nLLM Loss ",LLM_loss) 
        print("\napproximation \n",mul1)
        # mape = MAPE(np.array(LLM_loss), mul1)
        # approx_error_on_U.append(mape.tolist())
        # print("\napprox error on U ",approx_error_on_U)
        error1 = np.abs(np.array(LLM_loss) - mul1)
        error1 = np.reshape(error1, (-1,len(val_data)))
        error1 = np.mean(error1, axis=1)
        approx_error_on_U.append(error1.tolist())
        print("\napprox error on U on val data ",approx_error_on_U) 


        # print("\n*************Approximation error of Test Data on U ************")
        # print("\nLLM Loss ",LLM_loss_on_test_data) 
        # print("\napproximation \n",mul1_test)
        # error1 = np.abs(np.array(LLM_loss_on_test_data) - mul1_test)
        # error1 = np.reshape(error1, (-1,len(test_data)))
        # error1 = np.mean(error1, axis=1)
        # approx_error_on_U_on_test_data.append(error1.tolist())
        # print("\napprox error on U on test data ",approx_error_on_U_on_test_data) 


        # #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        # print("\n*************Approximation error of Validation Data on L_minus_U ************")
        # print("\nLLM Loss on L_minus_U ",LLM_loss_on_L_minus_U)
        # print("\napproximation \n",mul2)
        # error2 = np.abs(np.array(LLM_loss_on_L_minus_U) - mul2)
        # error2 = np.reshape(error2, (-1,len(val_data)))
        # error2 = np.mean(error2, axis=1)
        # approx_error_on_L_minus_U.append(error2.tolist())
        # print("\napprox error on L_minus_U on Val data ",approx_error_on_L_minus_U) 


        # print("\n*************Approximation error of Test Data on L_minus_U ************")
        # print("LLM Loss on L_minus_U ",LLM_loss_on_L_minus_U_on_test_data)
        # mul2_test = np.matmul(X_test, alpha)     
        # print("\napproximation \n",mul2_test)
        # error2 = np.abs(np.array(LLM_loss_on_L_minus_U_on_test_data) - mul2_test)
        # error2 = np.reshape(error2, (-1,len(test_data)))
        # error2 = np.mean(error2, axis=1)
        # approx_error_on_L_minus_U_on_test_data.append(error2.tolist())
        # print("\napprox error on L_minus_U on test data ",approx_error_on_L_minus_U_on_test_data) 
        
        print("\n*************Approximation error of Validation Data on V ************")
        print("\nLLM Loss on V ",LLM_loss_on_V)
        print("\napproximation \n",mul3)
        error3 = np.abs(np.array(LLM_loss_on_V) - mul3)
        error3 = np.reshape(error3, (-1,len(val_data)))
        error3 = np.mean(error3, axis=1)
        approx_error_on_V.append(error3.tolist())
        print("\napprox error on V on Val data ",approx_error_on_V) 

        #################################################################################
        # Calculate the new U_{t+1} by removing worst subset from U_t and adding the best subset from L \ U_t
        U.pop(S_worst_ind)
        U.append(S_star)


        #################################################################################
        # Make new L_minus_U by removing best subset from it and adding worst subset of U_t to it
        L_minus_U.pop(S_star_ind)
        L_minus_U.append(S_worst) 


        #################################################################################
        # Calculate Loss(Y,Star) 
        LLM_loss = LLM_loss[0:S_worst_ind*len(val_data)] + LLM_loss[(S_worst_ind*len(val_data))+len(val_data):]
        S_star_list = []
        S_star_list.append(S_star)
        new_LLM_loss = LLM_error_indicator(S_star_list, val_data)
        LLM_loss.extend(new_LLM_loss)


        # LLM_loss_on_test_data = LLM_loss_on_test_data[0:S_worst_ind*len(test_data)] + LLM_loss_on_test_data[(S_worst_ind*len(test_data))+len(test_data):]
        # new_LLM_loss = LLM_error_indicator(S_star_list, test_data)
        # LLM_loss_on_test_data.extend(new_LLM_loss)


        # LLM_loss_on_L_minus_U = LLM_loss_on_L_minus_U[0:S_star_ind*len(val_data)] + LLM_loss_on_L_minus_U[(S_star_ind*len(val_data))+len(val_data):] 
        # S_worst_list = []
        # S_worst_list.append(S_worst)
        # LLM_loss_of_worst_on_val_data = LLM_error_indicator(S_worst_list, val_data)
        # LLM_loss_on_L_minus_U.extend(LLM_loss_of_worst_on_val_data)


        # LLM_loss_on_L_minus_U_on_test_data = LLM_loss_on_L_minus_U_on_test_data[0:S_star_ind*len(test_data)] + LLM_loss_on_L_minus_U_on_test_data[(S_star_ind*len(test_data))+len(test_data):] 
        # LLM_loss_of_worst_on_test_data = LLM_error_indicator(S_worst_list, test_data)
        # LLM_loss_on_L_minus_U_on_test_data.extend(LLM_loss_of_worst_on_test_data)

        
        ################################################################################
        # Storing the indices of the subsets in U_t
        U_indices = []
        for i in range(len(U)):
            U_indices.append(U[i]['index'].tolist())

        L_minus_U_indices = []
        for i in range(len(L_minus_U)):
            L_minus_U_indices.append(L_minus_U[i]['index'].tolist())

        ################################################################################
        # fill W = identity(x_i \in S) E_{ij} 
        l = len(U)
        W_val = np.zeros((l*len(val_emb), len(train_emb)))
       # W_test = np.zeros((l*len(test_emb), len(train_emb)))
        for u in range(l):
            for i in range(len(train_emb)):
                if i in U_indices[u]:
                    W_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                    #W_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


        ##############################################################################
        # make X = E_{ij} for all i in L \ U_t 
        L_minus_U_len = len(L)-len(U)
        X_val = np.zeros((L_minus_U_len*len(val_emb), len(train_emb)))
        # X_test = np.zeros((L_minus_U_len*len(test_emb), len(train_emb)))
        for u in range(L_minus_U_len):
            for i in range(len(train_emb)):
                if i in L_minus_U_indices[u]:
                    X_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                    # X_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


        ################################################################################
        mul1 = np.matmul(W_val, alpha)
        #mul1_test = np.matmul(W_test, alpha)
        mul2 = np.matmul(X_val, alpha)
 

        #################################################################################
        #################################################################################
        print("\nMake new V by taking top v highest loss subsets from L \ U")
        #################################################################################

        mul_new2 = np.reshape(mul2, (L_minus_U_len,-1))
        sum_mul_new2 = np.sum(mul_new2, axis=1)
        v_worst_ind = np.argpartition(sum_mul_new2,-len(V))[-len(V):]
        V_new = [L_minus_U[i] for i in v_worst_ind]

        V_new_indices = []
        for i in range(len(V_new)):
            V_new_indices.append(V_new[i]['index'].tolist())

#         print("\n All the error values =",sum_mul_new2)
#         print("\n Top v highest error indices =",v_worst_ind)

#         print("\n V_old_indices =",V_indices)
#         print(" V_new_indices =",V_new_indices)
#         print("\n LLM_loss_on_old_V = ",LLM_loss_on_V)

#         set_difference = [item for item in V_new_indices if item not in V_indices]

#         # #new_items_in_V = [train_data[i] for i in set_difference]
#         new_items_in_V = []
#         for pos, ind in enumerate(L_indices):
#             if ind in set_difference:
#                 new_items_in_V.append(L[pos])


#         LLM_loss_of_new_items_in_V = LLM_error_indicator(new_items_in_V, val_data)
#         V_indices_overlap = []
#         LLM_loss_on_V_overlap = []
#         for pos, ind in enumerate(V_indices):
#             if ind in V_new_indices:
#                 V_indices_overlap.append(ind)
#                 LLM_loss_on_V_overlap.extend(LLM_loss_on_V[pos*len(val_data):(pos*len(val_data)+len(val_data))])
    

#         LLM_loss_on_V = LLM_loss_on_V_overlap + LLM_loss_of_new_items_in_V
#         V_indices = V_indices_overlap + set_difference

#         print("\n V_indices_overlap =",V_indices_overlap)
#         print(" V_new-V_old = set_difference =",set_difference)
#         print("\n V_latest_UPDATED_indices =",V_indices)
#         print("\n LLM_loss_on_V_overlap = ",LLM_loss_on_V_overlap)
#         print(" LLM_loss_of_new_items_in_V = ",LLM_loss_of_new_items_in_V)
#         print("\n LLM_loss_on_latest_UPDATED_V = ",LLM_loss_on_V)

#         #V = [train_data[i] for i in V_indices]
#         V = []
#         for pos, ind in enumerate(L_indices):
#             if ind in V_indices:
#                 V.append(L[pos])

        ######################################################
        V = V_new
        V_indices = V_new_indices
        LLM_loss_on_V = LLM_error_indicator(V, val_data)
        ######################################################
        

        # fill V = E_{ij} for all i in V_t
        v = len(V)
        V_val = np.zeros((v*len(val_emb), len(train_emb)))
        #V_test = np.zeros((v*len(test_emb), len(train_emb)))
        for u in range(v):
            for i in range(len(train_emb)):
                if i in V_indices[u]:
                    V_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                    #V_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]

        #####################################################################################################
        #####################################################################################################
        
        LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss, (len(U),-1))
        LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
        LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
        avg_LLM_loss_on_val.append(LLM_loss_on_val_avg.tolist()) 
        min_LLM_loss_on_val.append(LLM_loss_on_val_min.tolist())
        max_LLM_loss_on_val.append(LLM_loss_on_val_max.tolist())
        print("\n***********************************")
        print("S_worst_ind ",S_worst_ind)
        print("\n********* LLM LOSS ON U ON VALIDATION DATA *********")
        print("\nLLM_loss_on_val ",LLM_loss)
        print("\nAVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_val)
        print("\nMIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_val)
        print("\nMAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_val)



        
        # LLM_loss_for_each_subset = np.reshape(LLM_loss_on_test_data, (len(U),-1))
        # LLM_loss_for_each_subset = np.average(LLM_loss_for_each_subset, axis=1)
        # LLM_loss_avg = np.average(LLM_loss_for_each_subset)
        # LLM_loss_min = np.min(LLM_loss_for_each_subset)
        # LLM_loss_max = np.max(LLM_loss_for_each_subset)
        # #LLM_loss_on_test.append(LLM_loss_for_each_subset.tolist())
        # avg_LLM_loss.append(LLM_loss_avg.tolist()) 
        # min_LLM_loss.append(LLM_loss_min.tolist())
        # max_LLM_loss.append(LLM_loss_max.tolist())
        # print("\n***********************************")
        # print("S_worst_ind ",S_worst_ind)
        # print("\n********* LLM LOSS ON U FOR TEST DATA *********")
        # print("\nLLM_loss_on_test ",LLM_loss_on_test)
        # print("\nAVG_LLM_loss_on_TEST_data ",avg_LLM_loss)
        # print("\nMIN_LLM_loss_on_TEST_data ",min_LLM_loss)
        # print("\nMAX_LLM_loss_on_TEST_data ",max_LLM_loss)
        



        # LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss_on_L_minus_U, (len(L_minus_U),-1))
        # LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
        # LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
        # LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
        # LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
        # LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
        # avg_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_avg.tolist()) 
        # min_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_min.tolist())
        # max_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_max.tolist())
        # print("\n***********************************")
        # print("S_best_ind ",S_star_ind)
        # print("\n********* LLM LOSS ON L_minus_U FOR VALIDATION DATA *********")
        # print("\nLLM_loss_on_val ",LLM_loss_on_L_minus_U_on_val)
        # print("\nAVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_L_minus_U_on_val)
        # print("\nMIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_L_minus_U_on_val)
        # print("\nMAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_L_minus_U_on_val)

        # LLM_loss_for_each_subset = np.reshape(LLM_loss_on_L_minus_U_on_test_data, (len(L_minus_U),-1))
        # LLM_loss_for_each_subset = np.average(LLM_loss_for_each_subset, axis=1)
        # LLM_loss_avg = np.average(LLM_loss_for_each_subset)
        # LLM_loss_min = np.min(LLM_loss_for_each_subset)
        # LLM_loss_max = np.max(LLM_loss_for_each_subset)
        # LLM_loss_on_L_minus_U_on_test.append(LLM_loss_for_each_subset.tolist())
        # avg_LLM_loss_on_L_minus_U.append(LLM_loss_avg.tolist()) 
        # min_LLM_loss_on_L_minus_U.append(LLM_loss_min.tolist())
        # max_LLM_loss_on_L_minus_U.append(LLM_loss_max.tolist())
        # print("\n***********************************")
        # print("S_best_ind ",S_star_ind)
        # print("\n********* LLM LOSS ON L_minus_U FOR TEST DATA *********")
        # print("\nLLM_loss_on_test ",LLM_loss_on_L_minus_U_on_test)
        # print("\nAVG_LLM_loss_on_TEST_data ",avg_LLM_loss_on_L_minus_U)
        # print("\nMIN_LLM_loss_on_TEST_data ",min_LLM_loss_on_L_minus_U)
        # print("\nMAX_LLM_loss_on_TEST_data ",max_LLM_loss_on_L_minus_U)


        LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss_on_V, (len(V),-1))
        LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
        LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_V_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
        avg_LLM_loss_on_V_on_val.append(LLM_loss_on_val_avg.tolist()) 
        min_LLM_loss_on_V_on_val.append(LLM_loss_on_val_min.tolist())
        max_LLM_loss_on_V_on_val.append(LLM_loss_on_val_max.tolist())
        print("\n********* LLM LOSS ON V FOR VALIDATION DATA *********")
        print("\nLLM_loss_on_val ",LLM_loss_on_V_on_val)
        print("\nAVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_V_on_val)
        print("\nMIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_V_on_val)
        print("\nMAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_V_on_val)
        #===============================================================================





        #********* APPROX VALUE ON U ON VALIDATION DATA after updating U *********
        approx_value_on_U_for_each_subset = np.reshape(mul1, (len(U),-1))
        approx_value_on_U_for_each_subset = np.average(approx_value_on_U_for_each_subset, axis=1)
        approx_value_on_U_after_update.append(approx_value_on_U_for_each_subset.tolist())


        #********* APPROX VALUE ON U ON TEST DATA after updating U *********
        # approx_value_on_U_on_test_for_each_subset = np.reshape(mul1_test, (-1,len(test_data)))
        # approx_value_on_U_on_test_for_each_subset = np.average(approx_value_on_U_on_test_for_each_subset, axis=1)
        # approx_value_on_U_after_update_on_test_data.append(approx_value_on_U_on_test_for_each_subset.tolist())


        # #********* APPROX VALUE ON L-U ON VALIDATION DATA after updating L-U *********
        # approx_value_on_L_minus_U_for_each_subset = np.reshape(mul2, (-1,len(val_data)))
        # approx_value_on_L_minus_U_for_each_subset = np.average(approx_value_on_L_minus_U_for_each_subset, axis=1)
        # approx_value_on_L_minus_U_after_update.append(approx_value_on_L_minus_U_for_each_subset.tolist())


        #********* APPROX VALUE ON V ON VALIDATION DATA after updating V *********
        mul3 = np.matmul(V_val, alpha)
        approx_value_on_V_for_each_subset = np.reshape(mul3, (-1,len(val_data)))
        approx_value_on_V_for_each_subset = np.average(approx_value_on_V_for_each_subset, axis=1)
        approx_value_on_V_after_update.append(approx_value_on_V_for_each_subset.tolist())

        ###############################################################################


        #calculate the approximate error = (LLM_loss-W*alpha) on U
        print("\n*************Approximation error of Validation Data on U after updating U************")
        print("\nUpdated LLM Loss on U for Validation Data ",LLM_loss)
        print("\napproximation \n",mul1)
        # mape = MAPE(np.array(LLM_loss), mul1)
        # approx_error_on_U.append(mape.tolist())
        # print("\napprox error on U ",approx_error_on_U)
        error1 = np.abs(np.array(LLM_loss) - mul1)
        error1 = np.reshape(error1, (-1,len(val_data)))
        error1 = np.mean(error1, axis=1)
        approx_error_on_U_after_update.append(error1.tolist())
        print("\napprox error on U for Validation Data after updating U ",approx_error_on_U_after_update) 



        # print("\n*************Approximation error of Test Data on U after updating U************")
        # print("\nUpdated LLM Loss on U for Test Data ",LLM_loss_on_test_data)
        # print("\napproximation \n",mul1_test)
        # error1 = np.abs(np.array(LLM_loss_on_test_data) - mul1_test)
        # error1 = np.reshape(error1, (-1,len(test_data)))
        # error1 = np.mean(error1, axis=1)
        # approx_error_on_U_after_update_on_test_data.append(error1.tolist())
        # print("\napprox error on U for Test Data after updating U ",approx_error_on_U_after_update_on_test_data) 
        


        # #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        # print("\n*************Approximation error of Validation Data on L_minus_U after updating L_minus_U************")
        # print("\nUpdated LLM Loss on L_minus_U for Validation Data ",LLM_loss_on_L_minus_U)
        # print("\napproximation \n",mul2)
        # error2 = np.abs(np.array(LLM_loss_on_L_minus_U) - mul2)
        # error2 = np.reshape(error2, (-1,len(val_data)))
        # error2 = np.mean(error2, axis=1)
        # approx_error_on_L_minus_U_after_update.append(error2.tolist())
        # print("\napprox error on L_minus_U for Validation Data after updating L_minus_U ",approx_error_on_L_minus_U_after_update) 

        # #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        # print("\n*************Approximation error of Test Data on L_minus_U after updating L_minus_U************")
        # print("Updated LLM Loss on L_minus_U for Test Data ",LLM_loss_on_L_minus_U_on_test_data)
        # print("\napproximation \n",mul2_test)
        # error2 = np.abs(np.array(LLM_loss_on_L_minus_U_on_test_data) - mul2_test)
        # error2 = np.reshape(error2, (-1,len(test_data)))
        # error2 = np.mean(error2, axis=1)
        # approx_error_on_L_minus_U_after_update_on_test_data.append(error2.tolist())
        # print("\napprox error on L_minus_U for Test Data after updating L_minus_U ",approx_error_on_L_minus_U_after_update_on_test_data) 


        #calculate the approximate error = (LLM_loss_on_V - V*alpha) on V
        print("\n*************Approximation error of Validation Data on V after updating V************")
        print("\nUpdated LLM Loss on V for Validation Data ",LLM_loss_on_V)
        print("\napproximation \n",mul3)
        error3 = np.abs(np.array(LLM_loss_on_V) - mul3)
        error3 = np.reshape(error3, (-1,len(val_data)))
        error3 = np.mean(error3, axis=1)
        approx_error_on_V_after_update.append(error3.tolist())
        print("\napprox error on V for Validation Data after updating V ",approx_error_on_V_after_update) 
    



        ###########################################################################################
        #Calculate pairwise overlap between subsets in U
        # overlaps=[]
        # for i in range(len(U)):
        #     for index_j, s_i in U[i].iterrows():
        #         overlap=0
        #         for j in range(len(U)):
        #             if i!=j:
        #                 for index_j, s_j in U[j].iterrows():
        #                     if s_i["question"].lower() in s_j["question"].lower() or s_j["question"].lower() in s_i["question"].lower():
        #                         overlap+=1
        #         overlaps.append(overlap)

        overlaps=[]
        for i in range(len(U)):
            inner_overlaps=[]
            for j in range(len(U)):
                if i!=j:
                    overlap=0
                    for index_j, s_i in U[i].iterrows():
                        for index_j, s_j in U[j].iterrows():
                            if s_i["question"].lower() in s_j["question"].lower() or s_j["question"].lower() in s_i["question"].lower():
                                overlap+=1
                    inner_overlaps.append(overlap)
            overlaps.append(inner_overlaps)
                

        print("\noverlaps ",overlaps)
        print("len overlaps ",len(overlaps))


        overlap_for_each_subset = np.average(overlaps, axis=1)
        overlap_avg = np.average(overlap_for_each_subset)
        overlap_min = np.min(overlap_for_each_subset)
        overlap_max = np.max(overlap_for_each_subset)

        overlap_for_subset.append(overlap_for_each_subset.tolist())
        avg_overlap.append(overlap_avg.tolist()) 
        min_overlap.append(overlap_min.tolist())
        max_overlap.append(overlap_max.tolist())
        print("\n********* PAIRWISE OVERLAP *********")
        print("\noverlap_for_subset ",overlap_for_subset)
        print("\nAVG_overlap ",avg_overlap)
        print("MIN_overlap ",min_overlap)
        print("MAX_overlap ",max_overlap)
        

        folder1 = f"output/loss_folder_llama"
        np.savez(f'{folder1}', LLM_loss_on_val = LLM_loss_on_val, avg_LLM_loss_on_val = avg_LLM_loss_on_val, min_LLM_loss_on_val = min_LLM_loss_on_val, max_LLM_loss_on_val = max_LLM_loss_on_val, LLM_loss_on_V_on_val = LLM_loss_on_V_on_val, avg_LLM_loss_on_V_on_val = avg_LLM_loss_on_V_on_val, min_LLM_loss_on_V_on_val = min_LLM_loss_on_V_on_val, max_LLM_loss_on_V_on_val = max_LLM_loss_on_V_on_val, approx_error_on_U = approx_error_on_U, approx_error_on_V = approx_error_on_V, approx_error_on_U_after_update = approx_error_on_U_after_update, approx_error_on_V_after_update = approx_error_on_V_after_update,  approx_value_on_U = approx_value_on_U, approx_value_on_U_after_update = approx_value_on_U_after_update, approx_value_on_V = approx_value_on_V, approx_value_on_V_after_update = approx_value_on_V_after_update,  overlap_for_subset = overlap_for_subset , avg_overlap = avg_overlap, min_overlap = min_overlap, max_overlap = max_overlap)
        #==============================================================================================================

        # Increment t
        t+=1

    return U

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_open_source_completions(data):

    matches = 0
    mismatches = 0
    # counts = []
    # exnum = 1

    print("started running:")
    question_df = {"question":[],"answers":[]}

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

    print("while loop completed!")

    merged_exemplars = pd.concat(exemplars)
    merged_exemplars.to_csv("output/static_subset_selection_llama_strategyqa.csv")
    
    # merged_exemplars = pd.read_csv("output/case_aquarat_exemplars.csv")
    # exemplars = [merged_exemplars[0:5],merged_exemplars[5:10],merged_exemplars[10:15],merged_exemplars[15:20],merged_exemplars[20:25],merged_exemplars[25:30],merged_exemplars[30:35],merged_exemplars[35:40],merged_exemplars[40:45],merged_exemplars[45:50]]

    print("\n\n\n_____________Take the exemplar with minimum validation loss and use it as the exemplar")
    avg_err = LLM_avg_error(exemplars, val_data)
    print("\n\navg_err ",avg_err)
    ind = np.argmin(avg_err)
    print("\n\nmin ind ",ind)
    exemplars = exemplars[ind]

    index=0
    acc_records = []



    for index, row in test_data.iterrows():

        prompt = prompt_for_manual_prediction(row, exemplars)
        tmp = llm_output(prompt)

        answer = ""
        if len(tmp.split("The answer is:"))>1:
            answer = tmp.split("The answer is:")[1]
            answer = answer.split("\n")[0]

        print("\nAnswer: ", answer)
        print("GT: ", row["answer"])
        ground_truth = row["answer"]
        
        if ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower():
            matches+=1
        else:
            mismatches+=1

        question_df['question'].append(row["question"])
        question_df["answers"].append(answer)
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("output/static_llama_strategyqa_question_answer.tsv",sep="\t",index=False)


    print("EM:",matches/(matches+mismatches))

    return final_questions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    # train dataset
    train_data = read_strategyqa()

    print("Data read successfully")

    final_df = get_open_source_completions(train_data)
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()