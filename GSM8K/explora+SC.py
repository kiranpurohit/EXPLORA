import numpy as np
from numpy import linalg
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch
import re
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
    outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)
        
    out_text = []
    for x in range(0, 10):
        out_text.append(outputs[x]["generated_text"])
    return out_text

# Self consistency on 10 generated answers
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        ans = ""
        if len(tmp.split("Final Answer:"))>0:
            ans = tmp.split("Final Answer:")[-1]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "**************")
            try:
                float(ans)
                ans = round(float(ans))
                ans_list.append(ans)
            except: pass
        # ans_list.append(ans)

    # print(ans_list)
    d = {}
    for i in ans_list:
        if i=="":
            continue
        if int(i) in d:
            d[int(i)] += 1
        else:
            d[int(i)] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

# Strip answer from sentence
def clean_ans(s):
    ans_s = s.split("#### ")[1]
    ans_s = ans_s.replace(",","")
    return ans_s

def get_prompt(ex):
    s = "\n\n"
    s += "Question:" + ex["question"]+"\n"
    ex["answer"] = re.sub("<<.*?>>", "", ex["answer"])
    ex["answer"] = ex["answer"].replace("#### ", "Final Answer:")
    s += ex["answer"]
    return s

def llm_output(user_query):
    # results = [get_completion(user_query, api_keys[i], endpoint_urls[i], hard_code_exception=hard_code_exception) for i in range(len(endpoint_urls))]
    results = get_completion(user_query)
    
    return results

######################################################################################################

# Deleted Fsbd search code

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prompt_for_manual_prediction(ex, shots):

    prompt = "Follow given examples and solve the Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"
    for index, s in shots.iterrows():
        prompt += get_prompt(s)

    prompt += "\n\nFollowing the given examples generate step by step reasoning in Answer and generate Final Answer for the below question.\n\n" 
    prompt += "Question:" + ex["question"]
    
    return prompt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_avg_error(exemplars_set, val_data):
    # stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="LLM Loss Fn Exemplar"):
        matches = 0
        mismatches = 0
        exnum = 0
        # acc_records = []
        for index, row in val_data.iterrows():
            prompt = prompt_for_manual_prediction(row, exemplars)
            tmp_list = llm_output(prompt)

            n = self_con(tmp_list)
    
            ground_truth = int(clean_ans(row["answer"]))

            answer = ""
            maxf = 0
            if len(n)==0: answer=""
            else: maxf = n[0][1]

            for z in n:
                if z[1]==maxf:
                    if ground_truth==z[0]:
                        answer = z[0]

            if answer=="": 
                mismatches += 1
                if len(n)>0: answer = n[0][0]
            else: matches += 1           
            
            exnum+=1

        error.append(mismatches/exnum)

    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_error_indicator(exemplars_set, val_data):
    # stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="LLM Error Fn Exemplar"):
        for index, row in val_data.iterrows():
            prompt = prompt_for_manual_prediction(row, exemplars)
            # print("*********************************")
            # print(prompt)
            # print("*********************************")
            tmp_list = llm_output(prompt)
            n = self_con(tmp_list)
            
            ground_truth = int(clean_ans(row["answer"]))

            answer = ""
            maxf = 0
            if len(n)==0: answer=""
            else: maxf = n[0][1]

            for z in n:
                if z[1]==maxf:
                    if ground_truth==z[0]:
                        answer = z[0]

            if answer=="": loss = 1
            else: loss = 0

            error.append(loss)


    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def static_subset_selection(val_data, train_data, k, test_data):

    # test_data = test_data[:50]
    val_data = val_data[:20]
    # test_data = 15, val_data=9, train_data=30-9=21, k=5, L=40, U=10, V=10, L-U=30
    # test_emb = torch.Tensor(test_emb)

    with open('transfer_emb.pkl', 'rb') as f:
        transfer_emb = pickle.load(f)
    
    with open('val_emb.pkl', 'rb') as f1:
        val_emb = pickle.load(f1)
    
    val_emb = val_emb[:20]
    
    val_emb = torch.Tensor(val_emb)
    transfer_emb = torch.Tensor(transfer_emb)

    print("*****Embeddings loaded*****")
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

    # print("*****Initializing L*****")
    # Initialize L, 100 random set of subsets from train_data 
    for i in range(100):
        subset = []
        for name, group in train_data.groupby('cluster'):
            # subset.append(group.sample(k//num_gr))   
            subset.append(group.sample(1))   
        subsets = pd.concat(subset)
        L.append(subsets)
        L_indices.append(subsets['index'].tolist())

    # print("*****Initializing U*****")
    # Initialize U, 15 random set of subsets from L 
    ind_L = np.arange(0,len(L)).tolist()
    
    ind_total = random.sample(ind_L, 15)
    ind_U = ind_total[:10] #10
    ind_V = ind_total[10:] #5

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
    E_val = cosine_similarity(transfer_emb, val_emb)
    # E_test = cosine_similarity(transfer_emb, test_emb)

    print("*****Calculating LLM Loss*****")
    # Calculate Loss(Y,S) for all S in U
    LLM_loss = LLM_error_indicator(U, val_data)
    # LLM_loss_on_test_data = LLM_error_indicator(U, test_data)
    LLM_loss_on_L_minus_U = LLM_error_indicator(L_minus_U, val_data)
    LLM_loss_on_V = LLM_error_indicator(V, val_data)


    approx_error_on_U = []
    approx_error_on_U_after_update = []
    approx_error_on_L_minus_U = []
    approx_error_on_L_minus_U_after_update = []
    approx_error_on_V = []
    approx_error_on_V_after_update = []

    '''
    approx_error_on_U_on_test_data = []
    approx_error_on_U_after_update_on_test_data = []
    approx_error_on_L_minus_U_on_test_data = []
    approx_error_on_L_minus_U_after_update_on_test_data = []
    approx_error_on_V_on_test_data = []
    approx_error_on_V_after_update_on_test_data = []
    '''

    approx_value_on_U = []
    approx_value_on_U_after_update = []
    approx_value_on_L_minus_U = []
    approx_value_on_L_minus_U_after_update = []
    approx_value_on_V = []
    approx_value_on_V_after_update = []

    '''
    approx_value_on_U_on_test_data = []
    approx_value_on_U_after_update_on_test_data = []
    approx_value_on_L_minus_U_on_test_data = []
    approx_value_on_L_minus_U_after_update_on_test_data = []
    approx_value_on_V_on_test_data = []
    approx_value_on_V_after_update_on_test_data = []
    '''

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
    print("\nLLM_loss_on_val:",LLM_loss_on_val)
    print("AVG_LLM_loss_on_VAL_data:",avg_LLM_loss_on_val)
    print("MIN_LLM_loss_on_VAL_data:",min_LLM_loss_on_val)
    print("MAX_LLM_loss_on_VAL_data:",max_LLM_loss_on_val)
    #===============================================================================


    '''
    LLM_loss_on_test = []
    avg_LLM_loss = []
    min_LLM_loss = []
    max_LLM_loss = []
    LLM_loss_for_each_subset = np.reshape(LLM_loss_on_test_data, (len(U),-1))
    LLM_loss_for_each_subset = np.average(LLM_loss_for_each_subset, axis=1)
    LLM_loss_avg = np.average(LLM_loss_for_each_subset)
    LLM_loss_min = np.min(LLM_loss_for_each_subset)
    LLM_loss_max = np.max(LLM_loss_for_each_subset)
    LLM_loss_on_test.append(LLM_loss_for_each_subset.tolist())
    avg_LLM_loss.append(LLM_loss_avg.tolist()) 
    min_LLM_loss.append(LLM_loss_min.tolist())
    max_LLM_loss.append(LLM_loss_max.tolist())
    print("\n********* LLM LOSS ON U FOR TEST DATA *********")
    print("\nLLM_loss_on_test:",LLM_loss_on_test)
    print("AVG_LLM_loss_on_TEST_data:",avg_LLM_loss)
    print("MIN_LLM_loss_on_TEST_data:",min_LLM_loss)
    print("MAX_LLM_loss_on_TEST_data:",max_LLM_loss)
    '''
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
    print("\nLLM_loss_on_val:",LLM_loss_on_L_minus_U_on_val)
    print("AVG_LLM_loss_on_VAL_data:",avg_LLM_loss_on_L_minus_U_on_val)
    print("MIN_LLM_loss_on_VAL_data:",min_LLM_loss_on_L_minus_U_on_val)
    print("MAX_LLM_loss_on_VAL_data:",max_LLM_loss_on_L_minus_U_on_val)
    #===============================================================================

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
    print("\nLLM_loss_on_val:",LLM_loss_on_V_on_val)
    print("AVG_LLM_loss_on_VAL_data:",avg_LLM_loss_on_V_on_val)
    print("MIN_LLM_loss_on_VAL_data:",min_LLM_loss_on_V_on_val)
    print("MAX_LLM_loss_on_VAL_data:",max_LLM_loss_on_V_on_val)


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
            
    print("\nOverlaps:",overlaps)
    print("Len of overlaps:",len(overlaps))

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
    print("\nOverlap_for_subset:",overlap_for_subset)
    print("\nAVG_overlap:",avg_overlap)
    print("MIN_overlap:",min_overlap)
    print("MAX_overlap:",max_overlap)
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
    # W_test = np.zeros((l*len(test_emb), len(transfer_emb)))
    for u in range(l):
        for i in range(len(transfer_emb)):
            if i in U_indices[u]:
                W_val[u*len(val_emb):(u*len(val_emb)+len(val_emb)),i] = E_val[i]
                # W_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


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
    
    # while t<15:
    for i in tqdm(range(0, 15), desc="Iterating Loop of Static Selection"):

        ################################################################################
        # min_alpha_i (LLM_loss-W*alpha)^2
        #alpha = np.linalg.lstsq(W_val, LLM_loss, rcond=None)[0]
        # increase rows of llm loss by appending llm loss of V
        # increase rows of W by appending V

        ###################    Storing the values for each iteration    ################

        
        ################################################################################
 
        LLM_loss_on_U_V = LLM_loss + LLM_loss_on_V
        W_V_val = np.concatenate((W_val, V_val), axis=0)

        print("\n LLM_loss_on_U_V_len:",len(LLM_loss_on_U_V))
        print("\n LLM_loss_on_U_V:",LLM_loss_on_U_V)
        print("\n W_V_val_shape:",W_V_val.shape)
        print("\n W_V_val:",W_V_val)

        alpha = np.linalg.lstsq(W_V_val, LLM_loss_on_U_V, rcond=None)[0]
        # Print alpha

        print("\nAlpha shape:",alpha.shape)
        print("\nAlpha:",alpha)



        ################################################################################
        # Calculate the worst subset S_worst ∈ U_t that maximizes the approximate loss
        mul1 = np.matmul(W_val, alpha)
        mul_new1 = np.reshape(mul1, (len(U),-1))
        S_worst_ind = np.argmax(np.sum(mul_new1, axis=1))
        S_worst = U[S_worst_ind]

        ##############################################################################
        # mul1_test = np.matmul(W_test, alpha)

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

        '''
        #********* APPROX VALUE ON U ON TEST DATA *********
        approx_value_on_U_on_test_for_each_subset = np.reshape(mul1_test, (-1,len(test_data)))
        approx_value_on_U_on_test_for_each_subset = np.average(approx_value_on_U_on_test_for_each_subset, axis=1)
        approx_value_on_U_on_test_data.append(approx_value_on_U_on_test_for_each_subset.tolist())
        '''

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
        print("\nLLM Loss:",LLM_loss) 
        print("\nApproximation:\n",mul1)
        error1 = np.abs(np.array(LLM_loss) - mul1)
        error1 = np.reshape(error1, (-1,len(val_data)))
        error1 = np.mean(error1, axis=1)
        approx_error_on_U.append(error1.tolist())
        print("\nApprox error on U on val data:",approx_error_on_U) 

        '''
        print("\n*************Approximation error of Test Data on U ************")
        print("\nLLM Loss:",LLM_loss_on_test_data) 
        print("\nApproximation:\n",mul1_test)
        error1 = np.abs(np.array(LLM_loss_on_test_data) - mul1_test)
        error1 = np.reshape(error1, (-1,len(test_data)))
        error1 = np.mean(error1, axis=1)
        approx_error_on_U_on_test_data.append(error1.tolist())
        print("\nApprox error on U on test data:",approx_error_on_U_on_test_data) 
        '''

        #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        print("\n*************Approximation error of Validation Data on L_minus_U ************")
        print("\nLLM Loss on L_minus_U:",LLM_loss_on_L_minus_U)
        print("\nApproximation:\n",mul2)
        error2 = np.abs(np.array(LLM_loss_on_L_minus_U) - mul2)
        error2 = np.reshape(error2, (-1,len(val_data)))
        error2 = np.mean(error2, axis=1)
        approx_error_on_L_minus_U.append(error2.tolist())
        print("\nApprox error on L_minus_U on Val data:",approx_error_on_L_minus_U) 

        
        print("\n*************Approximation error of Validation Data on V ************")
        print("\nLLM Loss on V:",LLM_loss_on_V)
        print("\nApproximation:\n",mul3)
        error3 = np.abs(np.array(LLM_loss_on_V) - mul3)
        error3 = np.reshape(error3, (-1,len(val_data)))
        error3 = np.mean(error3, axis=1)
        approx_error_on_V.append(error3.tolist())
        print("\nApprox error on V on Val data:",approx_error_on_V) 

        #################################################################################
        # Calculate the new U_{t+1} by removing worst subset from U_t and adding the best subset from L \ U_t
        U.pop(S_worst_ind)
        U.append(S_star)


        #################################################################################
        # Make new L_minus_U by removing best subset from it and adding worst subset of U_t to it
        L_minus_U.pop(S_star_ind)
        L_minus_U.append(S_worst) 

        print("*****New LLM Losses*****")
        #################################################################################
        # Calculate Loss(Y,Star) 
        LLM_loss = LLM_loss[0:S_worst_ind*len(val_data)] + LLM_loss[(S_worst_ind*len(val_data))+len(val_data):]
        S_star_list = []
        S_star_list.append(S_star)
        new_LLM_loss = LLM_error_indicator(S_star_list, val_data)
        LLM_loss.extend(new_LLM_loss)

        '''
        LLM_loss_on_test_data = LLM_loss_on_test_data[0:S_worst_ind*len(test_data)] + LLM_loss_on_test_data[(S_worst_ind*len(test_data))+len(test_data):]
        new_LLM_loss = LLM_error_indicator(S_star_list, test_data)
        LLM_loss_on_test_data.extend(new_LLM_loss)
        '''


        LLM_loss_on_L_minus_U = LLM_loss_on_L_minus_U[0:S_star_ind*len(val_data)] + LLM_loss_on_L_minus_U[(S_star_ind*len(val_data))+len(val_data):] 
        S_worst_list = []
        S_worst_list.append(S_worst)
        LLM_loss_of_worst_on_val_data = LLM_error_indicator(S_worst_list, val_data)
        LLM_loss_on_L_minus_U.extend(LLM_loss_of_worst_on_val_data)


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
                    # W_test[u*len(test_emb):(u*len(test_emb)+len(test_emb)),i] = E_test[i]


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
        # mul1_test = np.matmul(W_test, alpha)
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

        '''
        print("\nAll the error values:",sum_mul_new2)
        print("\nTop v highest error indices:",v_worst_ind)

        print("\nV_old_indices:",V_indices)
        print("V_new_indices:",V_new_indices)
        print("\nLLM_loss_on_old_V:",LLM_loss_on_V)

        set_difference = [item for item in V_new_indices if item not in V_indices]

        # #new_items_in_V = [train_data[i] for i in set_difference]
        new_items_in_V = []
        for pos, ind in enumerate(L_indices):
            if ind in set_difference:
                new_items_in_V.append(L[pos])


        LLM_loss_of_new_items_in_V = LLM_error_indicator(new_items_in_V, val_data)
        V_indices_overlap = []
        LLM_loss_on_V_overlap = []
        for pos, ind in enumerate(V_indices):
            if ind in V_new_indices:
                V_indices_overlap.append(ind)
                LLM_loss_on_V_overlap.extend(LLM_loss_on_V[pos*len(val_data):(pos*len(val_data)+len(val_data))])
    

        LLM_loss_on_V = LLM_loss_on_V_overlap + LLM_loss_of_new_items_in_V
        V_indices = V_indices_overlap + set_difference

        print("\nV_indices_overlap:",V_indices_overlap)
        print("V_new-V_old = set_difference:",set_difference)
        print("\nV_latest_UPDATED_indices:",V_indices)
        print("\nLLM_loss_on_V_overlap:",LLM_loss_on_V_overlap)
        print("LLM_loss_of_new_items_in_V:",LLM_loss_of_new_items_in_V)
        print("\nLLM_loss_on_latest_UPDATED_V:",LLM_loss_on_V)

        #V = [train_data[i] for i in V_indices]
        V = []
        for pos, ind in enumerate(L_indices):
            if ind in V_indices:
                V.append(L[pos])
        '''

        # New mod
        V = V_new
        V_indices = V_new_indices
        LLM_loss_on_V = LLM_error_indicator(V, val_data)
        #


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
        print("S_worst_ind:",S_worst_ind)
        print("\n********* LLM LOSS ON U ON VALIDATION DATA *********")
        print("\nLLM_loss_on_val:",LLM_loss)
        print("\nAVG_LLM_loss_on_VAL_data:",avg_LLM_loss_on_val)
        print("\nMIN_LLM_loss_on_VAL_data:",min_LLM_loss_on_val)
        print("\nMAX_LLM_loss_on_VAL_data:",max_LLM_loss_on_val)

        '''
        LLM_loss_for_each_subset = np.reshape(LLM_loss_on_test_data, (len(U),-1))
        LLM_loss_for_each_subset = np.average(LLM_loss_for_each_subset, axis=1)
        LLM_loss_avg = np.average(LLM_loss_for_each_subset)
        LLM_loss_min = np.min(LLM_loss_for_each_subset)
        LLM_loss_max = np.max(LLM_loss_for_each_subset)
        LLM_loss_on_test.append(LLM_loss_for_each_subset.tolist())
        avg_LLM_loss.append(LLM_loss_avg.tolist()) 
        min_LLM_loss.append(LLM_loss_min.tolist())
        max_LLM_loss.append(LLM_loss_max.tolist())
        print("\n***********************************")
        print("S_worst_ind:",S_worst_ind)
        print("\n********* LLM LOSS ON U FOR TEST DATA *********")
        print("\nLLM_loss_on_test:",LLM_loss_on_test)
        print("\nAVG_LLM_loss_on_TEST_data:",avg_LLM_loss)
        print("\nMIN_LLM_loss_on_TEST_data:",min_LLM_loss)
        print("\nMAX_LLM_loss_on_TEST_data:",max_LLM_loss)
        '''

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
        print("S_best_ind:",S_star_ind)
        print("\n********* LLM LOSS ON L_minus_U FOR VALIDATION DATA *********")
        print("\nLLM_loss_on_val ",LLM_loss_on_L_minus_U_on_val)
        print("\nAVG_LLM_loss_on_VAL_data:",avg_LLM_loss_on_L_minus_U_on_val)
        print("\nMIN_LLM_loss_on_VAL_data:",min_LLM_loss_on_L_minus_U_on_val)
        print("\nMAX_LLM_loss_on_VAL_data:",max_LLM_loss_on_L_minus_U_on_val)


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
        print("\nLLM_loss_on_val:",LLM_loss_on_V_on_val)
        print("\nAVG_LLM_loss_on_VAL_data:",avg_LLM_loss_on_V_on_val)
        print("\nMIN_LLM_loss_on_VAL_data:",min_LLM_loss_on_V_on_val)
        print("\nMAX_LLM_loss_on_VAL_data:",max_LLM_loss_on_V_on_val)
        #===============================================================================

        #********* APPROX VALUE ON U ON VALIDATION DATA after updating U *********
        approx_value_on_U_for_each_subset = np.reshape(mul1, (len(U),-1))
        approx_value_on_U_for_each_subset = np.average(approx_value_on_U_for_each_subset, axis=1)
        approx_value_on_U_after_update.append(approx_value_on_U_for_each_subset.tolist())

        '''
        #********* APPROX VALUE ON U ON TEST DATA after updating U *********
        approx_value_on_U_on_test_for_each_subset = np.reshape(mul1_test, (-1,len(test_data)))
        approx_value_on_U_on_test_for_each_subset = np.average(approx_value_on_U_on_test_for_each_subset, axis=1)
        approx_value_on_U_after_update_on_test_data.append(approx_value_on_U_on_test_for_each_subset.tolist())
        '''

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
        print("\nUpdated LLM Loss on U for Validation Data:",LLM_loss)
        print("\nApproximation:\n",mul1)
        # mape = MAPE(np.array(LLM_loss), mul1)
        # approx_error_on_U.append(mape.tolist())
        # print("\napprox error on U ",approx_error_on_U)
        error1 = np.abs(np.array(LLM_loss) - mul1)
        error1 = np.reshape(error1, (-1,len(val_data)))
        error1 = np.mean(error1, axis=1)
        approx_error_on_U_after_update.append(error1.tolist())
        print("\nApprox error on U for Validation Data after updating U:",approx_error_on_U_after_update) 
        '''
        print("\n*************Approximation error of Test Data on U after updating U************")
        print("\nUpdated LLM Loss on U for Test Data:",LLM_loss_on_test_data)
        print("\nApproximation:\n",mul1_test)
        error1 = np.abs(np.array(LLM_loss_on_test_data) - mul1_test)
        error1 = np.reshape(error1, (-1,len(test_data)))
        error1 = np.mean(error1, axis=1)
        approx_error_on_U_after_update_on_test_data.append(error1.tolist())
        print("\nApprox error on U for Test Data after updating U:",approx_error_on_U_after_update_on_test_data) 
        '''
        #calculate the approximate error = (LLM_loss_on_L_minus_U - X*alpha) on L_minus_U
        print("\n*************Approximation error of Validation Data on L_minus_U after updating L_minus_U************")
        print("\nUpdated LLM Loss on L_minus_U for Validation Data:",LLM_loss_on_L_minus_U)
        print("\nApproximation:\n",mul2)
        error2 = np.abs(np.array(LLM_loss_on_L_minus_U) - mul2)
        error2 = np.reshape(error2, (-1,len(val_data)))
        error2 = np.mean(error2, axis=1)
        approx_error_on_L_minus_U_after_update.append(error2.tolist())
        print("\nApprox error on L_minus_U for Validation Data after updating L_minus_U:",approx_error_on_L_minus_U_after_update) 

        #calculate the approximate error = (LLM_loss_on_V - V*alpha) on V
        print("\n*************Approximation error of Validation Data on V after updating V************")
        print("\nUpdated LLM Loss on V for Validation Data:",LLM_loss_on_V)
        print("\nApproximation:\n",mul3)
        error3 = np.abs(np.array(LLM_loss_on_V) - mul3)
        error3 = np.reshape(error3, (-1,len(val_data)))
        error3 = np.mean(error3, axis=1)
        approx_error_on_V_after_update.append(error3.tolist())
        print("\nApprox error on V for Validation Data after updating V:",approx_error_on_V_after_update) 
    

        ###########################################################################################
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
                

        print("\nOverlaps:",overlaps)
        print("Len of overlaps:",len(overlaps))


        overlap_for_each_subset = np.average(overlaps, axis=1)
        overlap_avg = np.average(overlap_for_each_subset)
        overlap_min = np.min(overlap_for_each_subset)
        overlap_max = np.max(overlap_for_each_subset)

        overlap_for_subset.append(overlap_for_each_subset.tolist())
        avg_overlap.append(overlap_avg.tolist()) 
        min_overlap.append(overlap_min.tolist())
        max_overlap.append(overlap_max.tolist())
        print("\n********* PAIRWISE OVERLAP *********")
        print("\noverlap_for_subset:",overlap_for_subset)
        print("\nAVG_overlap:",avg_overlap)
        print("MIN_overlap:",min_overlap)
        print("MAX_overlap:",max_overlap)

        folder1 = f"./loss_folder"
        np.savez(f'{folder1}', LLM_loss_on_val = LLM_loss_on_val, avg_LLM_loss_on_val = avg_LLM_loss_on_val, min_LLM_loss_on_val = min_LLM_loss_on_val, max_LLM_loss_on_val = max_LLM_loss_on_val, LLM_loss_on_L_minus_U_on_val = LLM_loss_on_L_minus_U_on_val, avg_LLM_loss_on_L_minus_U_on_val = avg_LLM_loss_on_L_minus_U_on_val, min_LLM_loss_on_L_minus_U_on_val = min_LLM_loss_on_L_minus_U_on_val, max_LLM_loss_on_L_minus_U_on_val = max_LLM_loss_on_L_minus_U_on_val, LLM_loss_on_V_on_val = LLM_loss_on_V_on_val, avg_LLM_loss_on_V_on_val = avg_LLM_loss_on_V_on_val, min_LLM_loss_on_V_on_val = min_LLM_loss_on_V_on_val, max_LLM_loss_on_V_on_val = max_LLM_loss_on_V_on_val, approx_error_on_U = approx_error_on_U, approx_error_on_L_minus_U = approx_error_on_L_minus_U, approx_error_on_V = approx_error_on_V, approx_error_on_U_after_update = approx_error_on_U_after_update, approx_error_on_L_minus_U_after_update = approx_error_on_L_minus_U_after_update, approx_error_on_V_after_update = approx_error_on_V_after_update, approx_value_on_U = approx_value_on_U, approx_value_on_U_after_update = approx_value_on_U_after_update, approx_value_on_L_minus_U = approx_value_on_L_minus_U, approx_value_on_L_minus_U_after_update = approx_value_on_L_minus_U_after_update, approx_value_on_V = approx_value_on_V, approx_value_on_V_after_update = approx_value_on_V_after_update, overlap_for_subset = overlap_for_subset , avg_overlap = avg_overlap, min_overlap = min_overlap, max_overlap = max_overlap)
        #==============================================================================================================

        # Increment t
        # t+=1

    return U
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_open_source_completions(test_data, data):

    # stop_signal = "\n\n"
    matches = 0
    mismatches = 0

    question_df = {"question":[],"answers":[]}
    
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

    print("*************Starting static subset selection*************")

    exemplars = static_subset_selection(val_data, train_data, 5, test_data)

    print("*************Finished static subset selection*************")

    merged_exemplars = pd.concat(exemplars)
    merged_exemplars.to_csv("output/static_subset_selection_mistral3.csv")
    # exemplars = pd.read_csv("output/static_subset_selection_llama2.csv")
    
    print("\n\n\n********************Take the exemplar with minimum validation loss and use it as the exemplar")
    val_data = val_data[:20]
    avg_err = LLM_avg_error(exemplars, val_data)
    print("\n\nAvg Error:",avg_err)
    ind = np.argmin(avg_err)
    print("\n\nMin index:",ind)
    exemplars = exemplars[ind]

    index=0
    acc_records = []
    exnum = 1

    for row in tqdm(test_data,total=len(test_data),desc="Generating"):

        prompt = prompt_for_manual_prediction(row, exemplars)
        tmp_list = llm_output(prompt)
        # print(len(tmp_list))
        n = self_con(tmp_list)
        print(n)
        
        ground_truth = int(clean_ans(row["answer"]))

        answer = ""
        maxf = 0
        if len(n)==0: answer=""
        else: maxf = n[0][1]

        for z in n:
            if z[1]==maxf:
                if ground_truth==z[0]:
                    answer = z[0]

        if answer=="": 
            mismatches += 1
            if len(n)>0: answer = n[0][0]
        else: matches += 1
        
        print("\nAnswer:", answer)
        print("Ground Truth:", ground_truth)

        question_df['question'].append(row["question"])
        question_df["answers"].append(answer)
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("output/static_mistral_question_answer.tsv",sep="\t",index=False)

        print("Accuracy:", matches/exnum)
        exnum += 1

    print("EM:", matches/(matches+mismatches))

    return final_questions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    ### Load data
    # with open("train.jsonl", 'r') as f:
    #     json_list = list(f)
    # train_set = [json.loads(x) for x in json_list]
    train_set = pd.read_csv("train.csv")

    with open("test.jsonl", 'r') as f:
        json_list = list(f)
    test_set = [json.loads(x) for x in json_list]

    final_df = get_open_source_completions(test_set, train_set)
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()
