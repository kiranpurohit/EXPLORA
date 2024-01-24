import faiss
import numpy as np
import pickle 
import json
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

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

def mmr(doc_embeddings, query_embedding, lambda_param, top_k):
    # Normalize embeddings
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Number of documents
    num_docs = doc_embeddings.shape[0]

    # Create an index for Faiss
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings.astype(np.float32))

    # Query the index for similar documents
    _, similarity_indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)

    # Compute relevance scores
    relevance_scores = np.dot(doc_embeddings, query_embedding)

    # Initialize selected set
    selected_set = set()

    # Add the most relevant document to the selected set
    max_relevance_index = np.argmax(relevance_scores)
    selected_set.add(max_relevance_index)

    # Compute MMR scores and select documents iteratively
    while len(selected_set) < top_k:
        remaining_indices = list(set(similarity_indices[0]) - selected_set)
        remaining_embeddings = doc_embeddings[remaining_indices]

        # Compute similarity with the query for remaining documents
        similarity_scores = np.dot(remaining_embeddings, query_embedding)
        # print(similarity_scores)
        # Compute MMR scores
        mmr_scores = lambda_param * relevance_scores[remaining_indices] - (1 - lambda_param) * similarity_scores

        # Select document with maximum MMR score
        max_mmr_index = remaining_indices[np.argmax(mmr_scores)]
        selected_set.add(max_mmr_index)

    # Convert selected set to a list of indices
    selected_indices = list(selected_set)

    return selected_indices

def self_con(tmp_list):
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
    return n

# Example usage
if __name__ == "__main__":
    # Generate random embeddings for demonstration
    with open('pickle_test.pkl', 'rb') as f:
        l1 = pickle.load(f)
    
    with open('pickle_train.pkl', 'rb') as f1:
        l2 = pickle.load(f1)

    test_emb = np.array(l1)
    train_emb = np.array(l2)

    dev_set1 = open("problems_dev.json")
    dev_set1 = json.load(dev_set1)
    dev_set1 = dict(list(dev_set1.items())) #7686
    dev_set = []
    for i in dev_set1:
        dev_set.append(dev_set1[i])

    train_set1 = open("problems_train.json")
    train_set1 = json.load(train_set1)
    train_set1 = dict(list(train_set1.items())) #23059
    train_set = []
    for i in train_set1:
        train_set.append(train_set1[i])

    # Set the lambda parameter (trade-off between relevance and diversity)
    lambda_param = 0.7

    # Set the number of documents to return
    top_k = 5
    # Apply MMR function
    exnum = 0
    matches = 0
    mismatches = 0
    for ex in tqdm(dev_set,total=len(dev_set),desc="Generating"):
        user_query = """Follow the giving Examples each using its Table to find the answer for its Question with the reasoning and solve the Test Question in a similar manner.
        Examples:
        """
        selected_indices = mmr(train_emb, test_emb[exnum], lambda_param, top_k)

        print("\nSelected Indices:", selected_indices)

        for tr in selected_indices:
            user_query += "\nTable:\n" + train_set[tr]["table"] + "\nQuestion:" + train_set[tr]["question"] + "\Answer:" + train_set[tr]["solution"] + "\The answer is:" + train_set[tr]["answer"]

        user_query += "\nTable:\n" + ex["table"] + "\nQuestion:" + ex["question"]
        # print(user_query)
        tmp_list = compare_llm_outputs(user_query)
        # print(len(tmp_list))
        
        n = self_con(tmp_list)
        answer = n[0][0]
        if answer=="" and len(n)>1: answer = n[1][0]
        print("\nAnswer: ", answer)
        gt = ex["answer"]
        print("GT: ", gt)
        if gt.lower() in answer.lower() or answer.lower() in gt.lower():
            matches+=1
        else:
            mismatches+=1
        exnum += 1
        print("Hits:", matches)
        print("Accuracy:", matches/exnum)
    