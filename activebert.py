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

mask_token="[MASK]"
mask_token_id=tokenizer.convert_tokens_to_ids("[MASK]")

sentence = "The quickest brown fox jumping over the lazies dog. The quick brown fox jumps over the lazy dog . "
print("String::")
print(sentence)
strlist = sentence.split()
arr = []
bertlist = []
bertlist_ids = []

for i in range(len(strlist)):
    compound_word = strlist[i]
    toklist = tokenizer.tokenize(compound_word)
    toklist_id = tokenizer.convert_tokens_to_ids(toklist)
    
    if i % 5 == 1 :
    	trans_options =  ["brown", strlist[i], "dog", "lazy"]
    else:
    	trans_options = [strlist[i]]	
    arr.append((i, compound_word, (toklist, toklist_id), trans_options ))
    
    if len(trans_options) == 1:
        bertlist.extend(toklist)
        bertlist_ids.extend(toklist_id)
    else:
        bertlist.append(mask_token)
        bertlist_ids.append(mask_token_id)


print("\n\n========Array::")
for i in range(len(arr)):
    print(arr[i])


for i in range(len(arr)):
    (indx, word, tokz , trans_options) = arr[i]
    (toklist,toklist_id) = tokz
    if len(trans_options) == 1:
        bertlist.extend(toklist)
        bertlist_ids.extend(toklist_id)
    else:
        bertlist.append(mask_token)
        bertlist_ids.append(mask_token_id)

print("\n\n========Bertlist::")
print(len(bertlist), bertlist)

print(len(bertlist_ids), bertlist_ids)

tens = torch.tensor(bertlist_ids).unsqueeze(0)
tens = tens.to(device)
torch.no_grad()
preds = bert(tens)[0]
probs = softmax(preds)


for i in range(len(bertlist_ids)):
    (indx, word, tokz , trans_options) = arr[i]
    (toklist,toklist_id) = tokz
    trans_options_ids = (
        seq(trans_options)
        .map(lambda x: tokenizer.tokenize(x))
        .map(lambda x: tokenizer.convert_tokens_to_ids(x)[0])
    )
    if bertlist_ids[i] == mask_token_id :
        print("\nMask index",  i)
        ranked_pairs = (
            seq(trans_options_ids)
            .map(lambda x: float(probs[0][i][x].item()))
            .zip(trans_options)
            .sorted(key=lambda x: x[0], reverse=True)
        )
        print(i, "ranked_pairs: " , ranked_pairs)
        ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
        print(i, "ranked_options: " , ranked_options)

