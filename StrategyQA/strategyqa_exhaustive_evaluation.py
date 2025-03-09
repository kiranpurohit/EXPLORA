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

    return L

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
    merged_exemplars.to_csv("output/strategyqa_L_subsets.csv")

    merged_exemplars = pd.read_csv("output/strategyqa_L_subsets.csv")
    exemplars=np.array_split(merged_exemplars, 40)
    print(exemplars[0].shape)
    print(len(exemplars))
    exit(0)
    

    print("\n\n\n_____________Take the exemplar with minimum validation loss and use it as the exemplar")
    avg_err = LLM_avg_error(exemplars, val_data)
    print("\n\navg_err ",avg_err)
    ind = np.argmin(avg_err)
    print("\n\nmin ind ",ind)
    exemplars = exemplars[ind]


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