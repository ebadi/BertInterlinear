from collections import defaultdict
from typing import Dict, List, Tuple, Union, overload

import torch
from functional import pseq, seq
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)

def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
    
model=None
tokenizer=None
model_name="bert-large-uncased"
mask_token="[MASK]"
disable_gpu=False
device = torch.device("cpu")
print("device:", device)
if not model:
    print("using model:", model_name)
    if "distilbert" in model_name:
        bert = DistilBertForMaskedLM.from_pretrained(model_name)
    else:
        bert = BertForMaskedLM.from_pretrained(model_name)
    bert.to(device)
else:
    print("using custom model:", model.config.architectures)
    bert = model
    bert.to(device)
    
if not tokenizer:
    if "distilbert" in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
else:
    tokenizer = tokenizer

bert.eval()

sentence = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
print("String::")
print(sentence)
strlist = sentence.split()
arr = []
for i in range(len(strlist)):
    compound_word = strlist[i]
    toklist = tokenizer.tokenize(compound_word)
    toklist_id = tokenizer.convert_tokens_to_ids(toklist)
    if i % 6 == 0 :
    	arr.append((i, compound_word, (toklist, toklist_id), ["brown", strlist[i], "dog", "lazy"]))
    elif i % 7 == 0 :
        arr.append((i, compound_word, (toklist, toklist_id), [strlist[i], "lazy", "jump"]))
    else:
    	arr.append((i, compound_word, (toklist, toklist_id), [strlist[i]]))
    	

print("\n\n========Array::")
print(arr)

bertlist = []
for i in range(len(arr)):
    (indx, word, tokz , trans_options) = arr[i]
    (toklist,toklist_id) = tokz
    if len(trans_options) == 1:
        bertlist.extend(toklist)
    else:
        bertlist.append(mask_token)

print("\n\n========Bertlist::")
print(bertlist)
tens = torch.tensor(toklist_id).unsqueeze(0)
tens = tens.to(device)
torch.no_grad()
preds = bert(tens)[0]
probs = softmax(preds)
print(type(probs), probs.dtype)
for i in range(len(arr)):
    (indx, word, tokz , trans_options) = arr[i]
    (toklist,toklist_id) = tokz
    trans_options_ids = (
        seq(trans_options)
        .map(lambda x: tokenizer.tokenize(x))
        .map(lambda x: tokenizer.convert_tokens_to_ids(x)[0])
    )
    if bertlist[i] == mask_token :
        print("Mask index",  i)
        ranked_pairs = (
            seq(trans_options_ids)
            .map(lambda x: float(probs[0][i][x].item()))
            .zip(trans_options)
            .sorted(key=lambda x: x[0], reverse=True)
        )
        #ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
        print(ranked_pairs)
        #print(ranked_options)
