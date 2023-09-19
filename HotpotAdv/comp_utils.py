# import os
# import openai
# from transformers import GPT2TokenizerFast

# _TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
# GPT3_LENGTH_LIMIT = 2049
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def gpt_style_tokenize(x):
#     return _TOKENIZER.tokenize(x)

# def length_of_prompt(prompt, max_tokens):
#     return len(_TOKENIZER.tokenize(prompt)) + max_tokens

# def safe_completion(engine, prompt, max_tokens, stop, temp=0.0, logprobs=5):
#     len_prompt_token = len(_TOKENIZER.tokenize(prompt))    
#     if max_tokens + len_prompt_token >= GPT3_LENGTH_LIMIT:
#         print("OVERFLOW", max_tokens + len_prompt_token)
#         return {
#             "text": "overflow"
#         }
#     resp = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, stop=stop,
#         temperature=0.0, logprobs=logprobs, echo=True)

#     pred = resp["choices"][0]
#     return pred        

def conditional_strip_prompt_prefix(x, p):
    if x.startswith(p):
        x = x[len(p):]
    return x.strip()




import os
import torch as torch
import openai
from transformers import GPT2TokenizerFast, BertTokenizer, BertModel, logging
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.set_verbosity_error()

_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
GPT3_LENGTH_LIMIT = 2049
openai.api_key = os.getenv("OPENAI_API_KEY")

flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def get_similarity(sent1,sent2):
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode both sentences separately
    inputs_sentence1 = tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
    inputs_sentence2 = tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

    # Get BERT embeddings for both sentences
    with torch.no_grad():
        outputs_sentence1 = model(**inputs_sentence1)
        outputs_sentence2 = model(**inputs_sentence2)

    # Extract the embeddings for sentence1 and sentence2
    embedding_sentence1 = outputs_sentence1.last_hidden_state.mean(dim=1).numpy()
    embedding_sentence2 = outputs_sentence2.last_hidden_state.mean(dim=1).numpy()

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding_sentence1, embedding_sentence2)[0][0]

    return similarity

def flan_generate(x):
    inputs = flan_tokenizer(x, return_tensors="pt")
    outputs = flan_model.generate(**inputs, max_new_tokens=144)
    return(flan_tokenizer.batch_decode(outputs, skip_special_tokens=True))

def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)

def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens

def safe_completion(engine, prompt, max_tokens, stop, temp=0.0, logprobs=5):
    # len_prompt_token = len(_TOKENIZER.tokenize(prompt))    
    # if max_tokens + len_prompt_token >= GPT3_LENGTH_LIMIT:
    #     # print("OVERFLOW", max_tokens + len_prompt_token)
    #     # return {
    #     #     "text": "overflow"
    #     # }
    #     max_tokens = GPT3_LENGTH_LIMIT - len_prompt_token - 5
    
    print("prompt: ", prompt)

    resp = flan_generate(prompt)
    #resp = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, stop=stop, temperature=0.0, logprobs=logprobs, echo=True)

    print("resp: ", resp)
    
    #pred = resp["choices"][0]

    #print("pred: ", pred)
    #print("-------------------------------------")

    return resp

    pred = resp["choices"][0]

    print("pred: ", pred)
    print("-------------------------------------")



###########################################################################


# import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM
# import os
# import openai
# from transformers import GPT2TokenizerFast

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# _TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
# GPT3_LENGTH_LIMIT = 2049
# openai.api_key = os.getenv("OPENAI_API_KEY")

# ## v2 models
# #model_path = 'openlm-research/open_llama_3b_v2'
# # model_path = 'openlm-research/open_llama_7b_v2'

# ## v1 models
# model_path = 'openlm-research/open_llama_3b'
# # model_path = 'openlm-research/open_llama_7b'
# # model_path = 'openlm-research/open_llama_13b'

# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.float32, device_map='auto',
# )
# def openLlama(prompt):
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     #intput_ids=input_ids.to('cuda')

#     generation_output = model.generate(
#         input_ids=input_ids, max_new_tokens=32
#     )
#     return(tokenizer.decode(generation_output[0]))

# def gpt_style_tokenize(x):
#     return _TOKENIZER.tokenize(x)

# def length_of_prompt(prompt, max_tokens):
#     return len(_TOKENIZER.tokenize(prompt)) + max_tokens
# def safe_completion(engine, prompt, max_tokens, stop, temp=0.0, logprobs=5):
#     # len_prompt_token = len(_TOKENIZER.tokenize(prompt))    
#     # if max_tokens + len_prompt_token >= GPT3_LENGTH_LIMIT:
#     #     # print("OVERFLOW", max_tokens + len_prompt_token)
#     #     # return {
#     #     #     "text": "overflow"
#     #     # }
#     #     max_tokens = GPT3_LENGTH_LIMIT - len_prompt_token - 5
    
#     print("prompt: ", prompt)

#     resp = openLlama(prompt)
#     #resp = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, stop=stop, temperature=0.0, logprobs=logprobs, echo=True)

#     print("resp: ", resp)

#     #pred = resp["choices"][0]

#     #print("pred: ", pred)
#     #print("-------------------------------------")

#     return resp
