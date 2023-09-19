import os
import torch as torch
import openai
from transformers import GPT2TokenizerFast, BertTokenizer, BertModel, logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

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
    outputs = flan_model.generate(**inputs)
    return(flan_tokenizer.batch_decode(outputs, skip_special_tokens=True))

def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)

def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens

def safe_completion(engine, prompt, max_tokens, stop, temp=0.0, logprobs=5):

    print("prompt: ", prompt)

    resp = flan_generate(prompt)

    print("resp: ", resp)

    return resp
