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
mask_token="***mask***"
disable_gpu=False
mask_token = mask_token
device = torch.device( "cpu" )
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
    print("MSG4 :: ", masked_sent, words )
    pre, post = masked_sent.split(mask_token)
    print("MSG5 :: ", pre, post )
    tokens = ["[CLS]"] + tokenizer.tokenize(pre)
    target_idx = len(tokens)
    tokens += ["[MASK]"]
    tokens += tokenizer.tokenize(post) + ["[SEP]"]
    
    print("MSG2 :: ", tokens)

    words_ids = (
        seq(words)
        .map(lambda x: tokenizer.tokenize(x))
        .map(lambda x: tokenizer.convert_tokens_to_ids(x)[0])
    )
    print("MSG6 :: ", words_ids)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("MSG3 :: ", input_ids)
    tens = torch.tensor(input_ids).unsqueeze(0)
    tens = tens.to(device)
    print("MSG7 :: ", tens)
    with torch.no_grad():
        preds = bert(tens)[0]
        probs = softmax(preds)

        ranked_pairs = (
            seq(words_ids)
            .map(lambda x: float(probs[0][target_idx][x].item()))
            .zip(words)
            .sorted(key=lambda x: x[0], reverse=True)
        )
        

        ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
        ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()

        print("MSG8 :: ", ranked_options)
        print("MSG9 :: ", ranked_options_prob)

        del tens, preds, probs, tokens, words_ids, input_ids
        if device == "cuda":
            torch.cuda.empty_cache()
        return ranked_options, ranked_options_prob



def rank(
    sent: str,
    options: List[str],
    with_prob: bool = False,
    ):
    """
    Rank a list of candidates

    returns: Either a List of strings,
    or if `with_prob` is True, a Tuple of List[str], List[float]

    """

    options = ['buff', 'handsome', 'strong']
    sent = "Why Bert, you're looking ***mask*** today!"
    
    print("MSG1 ::", options, sent)

    ranked, prob = rank_single(sent, options)

    ranked = (
        seq(ranked)
        .map(lambda x: seq(x).make_string(" ").strip())
        .list()
    )
    if with_prob:
        return ranked, prob
    else:
        return ranked



masked_string = "Why Bert, you're looking ***mask*** today!"
options = ['buff', 'handsome', 'strong']

ranked_options = rank(masked_string, options=options, with_prob= True)
# >>> ['handsome', 'strong', 'buff']

print(ranked_options)
