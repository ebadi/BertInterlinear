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

sentence = "As the use of typewriters grew in the late 19th century, the phrase began appearing in typing lesson books as a practice sentence. Early examples include How to Become Expert in Typewriting: A Complete Instructor Designed Especially for the Remington Typewriter (1890),[5] and Typewriting Instructor and Stenographer's Hand-book (1892). By the turn of the 20th century, the phrase had become widely known. In the January 10, 1903, issue of Pitman's Phonetic Journal, it is referred to as  the well known memorized typing line embracing all the letters of the alphabet Robert Baden-Powell's book Scouting for Boys (1908) uses the phrase as a practice sentence for signaling"
print("String::")
print(sentence)
tokenized_sentence = sentence.split()
sentecelist = []
bertlist = []
bertlist_ids = []
masked_sentencelist = []

print("\n\n======== Sentecelist::")
for i in range(len(tokenized_sentence)):
    compound_word = tokenized_sentence[i]
    toklist = tokenizer.tokenize(compound_word)
    toklist_id = tokenizer.convert_tokens_to_ids(toklist)
    
    if i % 5 == 1 :
    	trans_options =  [("brown",0), (tokenized_sentence[i],0), ("dog",0), ("lazy",0)]
    else:
    	trans_options = [(tokenized_sentence[i],1)]

    sentecelist.append((i, compound_word, (toklist, toklist_id), trans_options ))
    print(sentecelist[i])



for i in range(len(sentecelist)):
    (indx, word, tokz , trans_options) = sentecelist[i]
    if len(trans_options) == 1:
        masked_sentencelist.append((indx, word, tokz , trans_options))
    else:
        masked_sentencelist.append((indx, "[MASK]", ([mask_token], [mask_token_id] ) , trans_options))

print("\n\n======== masked_sentencelist::")
for i in range(len(masked_sentencelist)):
    (indx, word, tokz , trans_options) = masked_sentencelist[i]
    (toklist, toklist_id) = tokz
    bertlist.extend(toklist)
    bertlist_ids.extend(toklist_id)
    print(masked_sentencelist[i])


print("\n\n============= BERT")    
print(len(bertlist), bertlist)
print(len(bertlist_ids), bertlist_ids)
tens = torch.tensor(bertlist_ids).unsqueeze(0)
tens = tens.to(device)
torch.no_grad()
preds = bert(tens)[0]
probs = softmax(preds)
bertIndx = 0


sorted_sentecelist = []
ranked_pairs = []

for i in range(len(masked_sentencelist)):
    (indx, word, tokz, trans_options) = masked_sentencelist[i]
    (toklist,toklist_id) = tokz
    trans_words = (
        seq(trans_options)
        .map(lambda x: x[0])
    )
    trans_options_ids = (
        seq(trans_words)
        .map(lambda x: tokenizer.tokenize(x))
        .map(lambda x: tokenizer.convert_tokens_to_ids(x)[0])
    )
    #print("masked_sentencelist index",i,tokz, " Bert index", bertIndx, bertlist[indx] )
    if bertlist_ids[bertIndx] == mask_token_id :
        ranked_pairs = (
            seq(trans_options_ids)
            .map(lambda x: float(probs[0][i][x].item()))
            .zip(trans_words)
            .map (lambda x: (x[1], x[0]))
            .sorted(key=lambda x: x[0], reverse=True)
        )
        print("[" + ranked_pairs[0][0] + "]",  end=" ")
    else:
        ranked_pairs = trans_options
        print(word, end=" ")
    bertIndx = bertIndx + len(toklist_id)
    sorted_sentecelist.append((indx, word, tokz , ranked_pairs))



print("\n\n======== sorted_sentecelist::")
for i in range(len(sorted_sentecelist)):
    print(sorted_sentecelist[i])
    



 

