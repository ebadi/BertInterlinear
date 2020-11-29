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

        
class FitBert:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name="bert-large-uncased",
        mask_token="***mask***",
        disable_gpu=False,
    ):
        self.mask_token = mask_token
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        print("device:", self.device)

        if not model:
            print("using model:", model_name)
            if "distilbert" in model_name:
                self.bert = DistilBertForMaskedLM.from_pretrained(model_name)
            else:
                self.bert = BertForMaskedLM.from_pretrained(model_name)
            self.bert.to(self.device)
        else:
            print("using custom model:", model.config.architectures)
            self.bert = model
            self.bert.to(self.device)
            
        if not tokenizer:
            if "distilbert" in model_name:
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        self.bert.eval()

    @staticmethod
    def softmax(x):
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

    def rank_single(self, masked_sent: str, words: List[str]):
        print("MSG4 :: ", masked_sent, words )
        pre, post = masked_sent.split(self.mask_token)
        print("MSG5 :: ", pre, post )
        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post) + ["[SEP]"]
        
        print("MSG2 :: ", tokens)

        words_ids = (
            seq(words)
            .map(lambda x: self.tokenizer.tokenize(x))
            .map(lambda x: self.tokenizer.convert_tokens_to_ids(x)[0])
        )
        print("MSG6 :: ", words_ids)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        print("MSG3 :: ", input_ids)
        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.to(self.device)
        print("MSG7 :: ", tens)
        with torch.no_grad():
            preds = self.bert(tens)[0]
            probs = self.softmax(preds)

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
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return ranked_options, ranked_options_prob


    def _simplify_options(self, sent: str, options: List[str]):

        options_split = seq(options).map(lambda x: x.split())

        trans_start = list(zip(*options_split))

        start = (
            seq(trans_start)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        options_split_reversed = seq(options_split).map(
            lambda x: seq(x[len(start) :]).reverse()
        )

        trans_end = list(zip(*options_split_reversed))

        end = (
            seq(trans_end)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        start_words = seq(start).make_string(" ")
        end_words = seq(end).reverse().make_string(" ")

        options = (
            seq(options_split)
            .map(lambda x: x[len(start) : len(x) - len(end)])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )

        sub = seq([start_words, self.mask_token, end_words]).make_string(" ").strip()
        sent = sent.replace(self.mask_token, sub)

        return options, sent, start_words, end_words

    def rank(
        self,
        sent: str,
        options: List[str],
        with_prob: bool = False,
    ):
        """
        Rank a list of candidates

        returns: Either a List of strings,
        or if `with_prob` is True, a Tuple of List[str], List[float]

        """

        options = seq(options).distinct().list()

        options, sent, start_words, end_words = self._simplify_options(sent, options)
        
        print("MSG1 ::", options, sent, start_words, end_words)

        ranked, prob = self.rank_single(sent, options)

        ranked = (
            seq(ranked)
            .map(lambda x: [start_words, x, end_words])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )
        if with_prob:
            return ranked, prob
        else:
            return ranked



fb = FitBert()

masked_string = "Why Bert, you're looking ***mask*** today!"
options = ['buff', 'handsome', 'strong']

ranked_options = fb.rank(masked_string, options=options, with_prob= True)
# >>> ['handsome', 'strong', 'buff']

print(ranked_options)
