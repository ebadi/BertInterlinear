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

def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

def rank_single(masked_sent: str, words: List[str]):
    print("MSG1 :: ", masked_sent, words )


    tokenized_text = tokenizer.tokenize(masked_sent)
    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 7
    tokenized_text[masked_index] = '[MASK]'
    
    masked_index_extra = 9
    tokenized_text[masked_index_extra] = '[MASK]'
    
    print("MSG2 :: ", masked_sent)
    print("MSG3 :: ", tokenized_text)


    words_ids = (
        seq(words)
        .map(lambda x: tokenizer.tokenize(x))
        .map(lambda x: tokenizer.convert_tokens_to_ids(x)[0])
    )
    print("MSG4 :: ", words_ids)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    print("MSG5 :: ", input_ids)
    tens = torch.tensor(input_ids).unsqueeze(0)
    tens = tens.to(device)
    print("MSG6 :: ", tens)
    with torch.no_grad():
        preds = bert(tens)[0]
        probs = softmax(preds)

        ranked_pairs = (
            seq(words_ids)
            .map(lambda x: float(probs[0][masked_index][x].item()))
            .zip(words)
            .sorted(key=lambda x: x[0], reverse=True)
        )
        ranked_pairs_extra = (
            seq(words_ids)
            .map(lambda x: float(probs[0][masked_index_extra][x].item()))
            .zip(words)
            .sorted(key=lambda x: x[0], reverse=True)
        )
        

        ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
        ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()

        print("MSG6 :: ", ranked_options)
        print("MSG7 :: ", ranked_options_prob)
        
        ranked_options_extra = (seq(ranked_pairs_extra).map(lambda x: x[1])).list()
        ranked_options_prob_extra = (seq(ranked_pairs_extra).map(lambda x: x[0])).list()
        
        print("MSG8 :: ", ranked_options_extra)
        print("MSG9 :: ", ranked_options_prob_extra)
        

        del tens, preds, probs, tokenized_text, words_ids, input_ids
        if device == "cuda":
            torch.cuda.empty_cache()
        return ranked_options, ranked_options_prob


sent = "Why Bert, you're looking test today!"
options = ['buff', 'handsome', 'strong']

ranked, prob = rank_single(sent, options)

ranked = (
        seq(ranked)
        .map(lambda x: seq(x).make_string(" ").strip())
        .list()
    )
    

print(ranked, prob)
