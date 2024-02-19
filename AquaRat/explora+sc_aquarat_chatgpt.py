import numpy as np
from numpy import linalg
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch
import time
import pickle 
import json
import openai
openai.api_key = "OPEN-AI-KEY"
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import BertTokenizer, BertModel, logging
import transformers
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


random.seed(7)
np.random.seed(7)
torch.manual_seed(7)



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



def prompt_for_manual_prediction(ex, shots):

    stop_signal = "\n\n"
    showcase_examples = [
            "Q: {}\nO: {} \nRationale: {}\n Answer: {}\n".format(
                 s["question"],s["options"],
                s["rationale"], s["correct"]) for index, s in shots.iterrows()
        ]

    # prompt = "\n".join(showcase_examples)
    # prompt=prompt+"\n\n{text}\n"


    input_example = "\nQ: {}\n O: {}\nRationale: [INS] \n Answer: [INS]".format(ex['question'], ex['options'])
    prompt = "\n".join(showcase_examples + [input_example])

    return prompt,stop_signal






def in_context_manual_prediction(ex, training_data):
    template,stop = prompt_for_manual_prediction(ex, training_data)

    messages=[{
                "role": "system",
                "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic.Do not generate examples in your answer",
            }]
    text={"role": "user", "content":""" Follow given examples and solve the Test Question at end in similar manner step by step and output Rationale under Rationale: and strictly only answer option preceded by Answer: 
         Examples:{}""".format(template)}
    print("template",template)
    messages.append(text)

    try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= messages ,
                temperature=0.3,
                max_tokens=256,
                top_p=1.0,
                n=10,
                frequency_penalty=0.8,
                presence_penalty=0.6            
                )
    except:
        time.sleep(60)
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= messages ,
                temperature=0.3,
                max_tokens=256,
                top_p=1.0,
                n=10,
                frequency_penalty=0.8,
                presence_penalty=0.6            
                )
    out_text = []
    for index, ans in enumerate(response['choices']):
        text = ans['message']["content"]
        out_text.append(text)
    return out_text




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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



def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        # tmp_list.append(compare_llm_outputs(user_query))
        # tmp = compare_llm_outputs(user_query)
        #print(tmp)
        ans = ""
        if len(tmp.split("Answer:"))>0:
            ans = tmp.split("Answer:")[-1]
        else:
            if len(tmp.split("answer is")) >0:
                ans = tmp.split("answer is")[-1]
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




tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

logging.set_verbosity_error()



######################################################################################################

def read_test_data():
    data = pd.read_json(path_or_buf="/Users/venktesh_1/Downloads/Explora_iitkgp/data/AQUA_RAT/dev.jsonl", lines=True)
    return data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_train_data():
    data = pd.read_json(path_or_buf="/Users/venktesh_1/Downloads/Explora_iitkgp/data/AQUA_RAT/train.jsonl", lines=True)
    return data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




def LLM_avg_error(exemplars_set, val_data):
    stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="predicting"):
        matches = 0
        mismatches =0
        acc_records = []
        for index, row in val_data.iterrows():
            #prompt,stop = prompt_for_manual_prediction(row, exemplars)
            #chain_answer = safe_completion(prompt=prompt, max_tokens=_MAX_TOKENS, stop=stop_signal, temp=0.0, logprobs=5)

            tmp_list = in_context_manual_prediction(row,exemplars)
            #print(tmp_list)
            #print(tmp[0])
            
            n = self_con(tmp_list)
            answer = n[0][0]
            if answer=="" and len(n)>1: answer = n[1][0]
            # ans = ""
            # if len(tmp_list[0].split("The option is "))>6:
            #     ans = tmp_list[0].split("The option is ")[6][0]
            # answer=ans
            
            print("\nAnswer:", answer)
            gt = row["correct"]
            print("GT:", gt)
            if(answer.strip().lower()[0]==gt.strip().lower()):
              matches+=1
            else:
              mismatches+=1
        print("mismatches/(matches+mismatches)",mismatches/(matches+mismatches))
        error.append(mismatches/(matches+mismatches))

    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




def LLM_error_indicator(exemplars_set, val_data):
    stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="predicting"):
        for index, row in val_data.iterrows():
            tmp_list = in_context_manual_prediction(row,exemplars)
            #print(tmp[0])
            
            #n = self_con(tmp_list)
            # answer = n[0][0]
            # if answer=="" and len(n)>1: answer = n[1][0]
            # ans = ""
            if len(tmp_list[0].split("The option is "))>6:
                ans = tmp_list[0].split("The option is ")[6][0]
            answer=ans
            
            print("\nAnswer:", answer)
            gt = ex["correct"]
            print("GT:", gt)

            if answer.strip().lower()==gt.strip().lower():
                loss=0
            else:
                loss=1

            error.append(loss)

    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




def static_subset_selection(val_data, train_data, k, test_data):

    #test_data = test_data[:15]
    val_data = val_data[:20]
    # val_data=20, k=5, L=100, U=10, V=5, L-U=90
    

    print("checkpoint-1:")


    # dbfile1 = open('val_emb.pkl', 'rb')    
    # val_emb = pickle.load(dbfile1)
    # val_emb=val_emb[:2]



    #dbfile2 = open('test_emb.pkl', 'rb')    
    #test_emb = pickle.load(dbfile2)


    dbfile3 = open('/Users/venktesh_1/Downloads/Explora_iitkgp/data/aquarat_embeddings/transfer_emb.pkl', 'rb')    
    transfer_emb = pickle.load(dbfile3)

    #Calculate embeddings for all validation questions
    #val_emb = get_embeddings1(val_data["question"].tolist())
    with open("/Users/venktesh_1/Downloads/Explora_iitkgp/data/aquarat_embeddings/val_emb.pkl", 'rb') as f:
            val_emb = pickle.load(f)
    
    # print("1-done")
    # test_emb = get_embeddings1(test_data["question"].tolist())
    # with open("test_emb.pkl", 'wb') as f:
    #         pickle.dump(test_emb, f)

    # print("2-done")
    # transfer_emb = get_embeddings1(train_data["question"].tolist())
    # with open("transfer_emb.pkl", 'wb') as f:
    #         pickle.dump(transfer_emb, f)
    # print("3-done")


    # k-means clustering on train_data with k=5
    kmeans = KMeans(n_clusters=5, random_state=0).fit(transfer_emb)
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
    for i in range(100):
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

    print("check-point-2:")
    
    # Calculate the similarity matrix, Eij = cosine similarity between exemplar x_i=train_data and test example u_j=val_data
    E_val = cosine_similarity(transfer_emb, val_emb)
    #E_test = cosine_similarity(transfer_emb, test_emb)

    # Calculate Loss(Y,S) for all S in U
    LLM_loss = LLM_error_indicator(U, val_data)
    #LLM_loss_on_test_data = LLM_error_indicator(U, test_data)
    LLM_loss_on_L_minus_U = LLM_error_indicator(L_minus_U, val_data)
    LLM_loss_on_V = LLM_error_indicator(V, val_data)
    print("check-point-3:")
    


    approx_error_on_U = []
    approx_error_on_U_after_update = []
    approx_error_on_L_minus_U = []
    approx_error_on_L_minus_U_after_update = []
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
    approx_value_on_L_minus_U = []
    approx_value_on_L_minus_U_after_update = []
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
    avg_LLM_loss = []
    min_LLM_loss = []
    max_LLM_loss = []
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

   

    LLM_loss_on_L_minus_U_on_val = []
    avg_LLM_loss_on_L_minus_U_on_val = []
    min_LLM_loss_on_L_minus_U_on_val = []
    max_LLM_loss_on_L_minus_U_on_val = []
    LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss_on_L_minus_U, (len(L_minus_U),-1))
    LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
    LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
    LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
    avg_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_avg.tolist()) 
    min_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_min.tolist())
    max_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_max.tolist())
    print("\n********* LLM LOSS ON L_minus_U FOR VALIDATION DATA *********")
    print("\nLLM_loss_on_val ",LLM_loss_on_L_minus_U_on_val)
    print("AVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_L_minus_U_on_val)
    print("MIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_L_minus_U_on_val)
    print("MAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_L_minus_U_on_val)
    #===============================================================================

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
    W_val = np.zeros((l*len(val_emb), len(transfer_emb)))
    #W_test = np.zeros((l*len(test_emb), len(transfer_emb)))
    for u in range(l):
        for i in range(len(transfer_emb)):
            if i in U_indices[u]:
                W_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                #W_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


    ################################################################################
    # fill V = E_{ij} for all i in V_t
    v = len(V)
    V_val = np.zeros((v*len(val_emb), len(transfer_emb)))
    #V_test = np.zeros((v*len(test_emb), len(transfer_emb)))
    for u in range(v):
        for i in range(len(transfer_emb)):
            if i in V_indices[u]:
                V_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                #V_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]



    ##############################################################################
    # make X = E_{ij} for all i in L-U_t 
    L_minus_U_len = len(L)-len(U)
    X_val = np.zeros((L_minus_U_len*len(val_emb), len(transfer_emb)))
    # X_test = np.zeros((L_minus_U_len*len(test_emb), len(transfer_emb)))
    for u in range(L_minus_U_len):
        for i in range(len(transfer_emb)):
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


        #********* APPROX VALUE ON L-U ON VALIDATION DATA *********
        approx_value_on_L_minus_U_for_each_subset = np.reshape(mul2, (-1,len(val_data)))
        approx_value_on_L_minus_U_for_each_subset = np.average(approx_value_on_L_minus_U_for_each_subset, axis=1)
        approx_value_on_L_minus_U.append(approx_value_on_L_minus_U_for_each_subset.tolist())

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


        #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        print("\n*************Approximation error of Validation Data on L_minus_U ************")
        print("\nLLM Loss on L_minus_U ",LLM_loss_on_L_minus_U)
        print("\napproximation \n",mul2)
        error2 = np.abs(np.array(LLM_loss_on_L_minus_U) - mul2)
        error2 = np.reshape(error2, (-1,len(val_data)))
        error2 = np.mean(error2, axis=1)
        approx_error_on_L_minus_U.append(error2.tolist())
        print("\napprox error on L_minus_U on Val data ",approx_error_on_L_minus_U) 


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


        LLM_loss_on_L_minus_U = LLM_loss_on_L_minus_U[0:S_star_ind*len(val_data)] + LLM_loss_on_L_minus_U[(S_star_ind*len(val_data))+len(val_data):] 
        S_worst_list = []
        S_worst_list.append(S_worst)
        LLM_loss_of_worst_on_val_data = LLM_error_indicator(S_worst_list, val_data)
        LLM_loss_on_L_minus_U.extend(LLM_loss_of_worst_on_val_data)


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
        W_val = np.zeros((l*len(val_emb), len(transfer_emb)))
       # W_test = np.zeros((l*len(test_emb), len(transfer_emb)))
        for u in range(l):
            for i in range(len(transfer_emb)):
                if i in U_indices[u]:
                    W_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                    #W_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


        ##############################################################################
        # make X = E_{ij} for all i in L \ U_t 
        L_minus_U_len = len(L)-len(U)
        X_val = np.zeros((L_minus_U_len*len(val_emb), len(transfer_emb)))
        # X_test = np.zeros((L_minus_U_len*len(test_emb), len(transfer_emb)))
        for u in range(L_minus_U_len):
            for i in range(len(transfer_emb)):
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


        
        V = V_new
        V_indices = V_new_indices
        LLM_loss_on_V = LLM_error_indicator(V, val_data)
        
        

        # fill V = E_{ij} for all i in V_t
        v = len(V)
        V_val = np.zeros((v*len(val_emb), len(transfer_emb)))
        #V_test = np.zeros((v*len(test_emb), len(transfer_emb)))
        for u in range(v):
            for i in range(len(transfer_emb)):
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
        



        LLM_loss_on_val_for_each_subset = np.reshape(LLM_loss_on_L_minus_U, (len(L_minus_U),-1))
        LLM_loss_on_val_for_each_subset = np.average(LLM_loss_on_val_for_each_subset, axis=1)
        LLM_loss_on_val_avg = np.average(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val_min = np.min(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_val_max = np.max(LLM_loss_on_val_for_each_subset)
        LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_for_each_subset.tolist())
        avg_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_avg.tolist()) 
        min_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_min.tolist())
        max_LLM_loss_on_L_minus_U_on_val.append(LLM_loss_on_val_max.tolist())
        print("\n***********************************")
        print("S_best_ind ",S_star_ind)
        print("\n********* LLM LOSS ON L_minus_U FOR VALIDATION DATA *********")
        print("\nLLM_loss_on_val ",LLM_loss_on_L_minus_U_on_val)
        print("\nAVG_LLM_loss_on_VAL_data ",avg_LLM_loss_on_L_minus_U_on_val)
        print("\nMIN_LLM_loss_on_VAL_data ",min_LLM_loss_on_L_minus_U_on_val)
        print("\nMAX_LLM_loss_on_VAL_data ",max_LLM_loss_on_L_minus_U_on_val)

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


        #********* APPROX VALUE ON L-U ON VALIDATION DATA after updating L-U *********
        approx_value_on_L_minus_U_for_each_subset = np.reshape(mul2, (-1,len(val_data)))
        approx_value_on_L_minus_U_for_each_subset = np.average(approx_value_on_L_minus_U_for_each_subset, axis=1)
        approx_value_on_L_minus_U_after_update.append(approx_value_on_L_minus_U_for_each_subset.tolist())


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
        


        #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        print("\n*************Approximation error of Validation Data on L_minus_U after updating L_minus_U************")
        print("\nUpdated LLM Loss on L_minus_U for Validation Data ",LLM_loss_on_L_minus_U)
        print("\napproximation \n",mul2)
        error2 = np.abs(np.array(LLM_loss_on_L_minus_U) - mul2)
        error2 = np.reshape(error2, (-1,len(val_data)))
        error2 = np.mean(error2, axis=1)
        approx_error_on_L_minus_U_after_update.append(error2.tolist())
        print("\napprox error on L_minus_U for Validation Data after updating L_minus_U ",approx_error_on_L_minus_U_after_update) 

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
        

        folder1 = f"./loss_folder1"
        np.savez(f'{folder1}',  avg_LLM_loss = avg_LLM_loss, min_LLM_loss = min_LLM_loss, max_LLM_loss = max_LLM_loss, LLM_loss_on_val = LLM_loss_on_val, avg_LLM_loss_on_val = avg_LLM_loss_on_val, min_LLM_loss_on_val = min_LLM_loss_on_val, max_LLM_loss_on_val = max_LLM_loss_on_val, LLM_loss_on_L_minus_U_on_val = LLM_loss_on_L_minus_U_on_val, avg_LLM_loss_on_L_minus_U_on_val = avg_LLM_loss_on_L_minus_U_on_val, min_LLM_loss_on_L_minus_U_on_val = min_LLM_loss_on_L_minus_U_on_val, max_LLM_loss_on_L_minus_U_on_val = max_LLM_loss_on_L_minus_U_on_val, LLM_loss_on_V_on_val = LLM_loss_on_V_on_val, avg_LLM_loss_on_V_on_val = avg_LLM_loss_on_V_on_val, min_LLM_loss_on_V_on_val = min_LLM_loss_on_V_on_val, max_LLM_loss_on_V_on_val = max_LLM_loss_on_V_on_val, approx_error_on_U = approx_error_on_U, approx_error_on_L_minus_U = approx_error_on_L_minus_U, approx_error_on_V = approx_error_on_V, approx_error_on_U_after_update = approx_error_on_U_after_update, approx_error_on_L_minus_U_after_update = approx_error_on_L_minus_U_after_update, approx_error_on_V_after_update = approx_error_on_V_after_update,  approx_value_on_U = approx_value_on_U, approx_value_on_U_after_update = approx_value_on_U_after_update, approx_value_on_L_minus_U = approx_value_on_L_minus_U, approx_value_on_L_minus_U_after_update = approx_value_on_L_minus_U_after_update, approx_value_on_V = approx_value_on_V, approx_value_on_V_after_update = approx_value_on_V_after_update,  overlap_for_subset = overlap_for_subset , avg_overlap = avg_overlap, min_overlap = min_overlap, max_overlap = max_overlap)
        
        # Increment t
        t+=1

    return U
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




def get_open_source_completions(test_data, data):

    stop_signal = "\n\n"
    matches = 0
    mismatches =0
    print("started running:")

    question_df = {"question":[],"answers":[]}

    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data=val_data[:20]

    #exemplars = static_subset_selection(val_data, train_data, 5, test_data)

    print("while loop completed!")

    #merged_exemplars = pd.concat(exemplars)
    #merged_exemplars.to_csv("output/aquarat/explora_subset_selection_mistral_aquarat.csv")
    merged_exemplars = pd.read_csv("/Users/venktesh_1/Downloads/Explora_iitkgp/data/explora_subset_selection_mistral_aquarat.csv")
    #merged_exemplars.to_csv("output/static_subset_selection_mistral.csv")
    exemplars = [merged_exemplars[0:5],merged_exemplars[5:10],merged_exemplars[10:15],merged_exemplars[15:20],merged_exemplars[20:25],merged_exemplars[25:30],merged_exemplars[30:35],merged_exemplars[35:40],merged_exemplars[40:45],merged_exemplars[45:50]]

    #*****************************************************************************
    print("\n\n\n_____________Take the exemplar with minimum validation loss and use it as the exemplar")
    avg_err = LLM_avg_error(exemplars, val_data)
    print("\n\navg_err ",avg_err)
    ind = np.argmin(avg_err)
    print("\n\nmin ind ",ind)
    exemplars = exemplars[ind]

    index=0
    acc_records = []
    question_df = {"question":[],"answers":[]}

    for index, row in test_data.iterrows():

        tmp_list = in_context_manual_prediction(row,exemplars)
        #print(tmp[0])
        
        n = self_con(tmp_list)
        answer = n[0][0]
        if answer=="" and len(n)>1: answer = n[1][0]
        ans = ""
        # if len(tmp[0].split("The option is "))>1:
        #     ans = tmp[0].split("The option is ")[1][0]
        # answer=ans
        
        print("\nAnswer:", answer)
        gt = row["correct"]
        print("GT:", gt)
        if(answer.strip().lower()[0]==gt.strip().lower()):
          matches+=1
        else:
          mismatches+=1
        question_df['question'].append(row["question"])
        question_df["answers"].append(answer)
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("static_aquarat_mistral_gpt3_question_answer_1.tsv",sep="\t",index=False)
        print("EM:",matches/(matches+mismatches))

    return final_questions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





def test_few_shot_prediction():

    # train dataset
    train_data = read_train_data()

    # test dataset
    test_data = read_test_data()

    final_df = get_open_source_completions(test_data, train_data)
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()
