from transformers import BertTokenizer, BertForMaskedLM
import torch

# init model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
# init softmax to get probabilities later on
sm = torch.nn.Softmax(dim=0)
torch.set_grad_enabled(False)

# set sentence with MASK token, convert to token_ids
sentence = f"I {tokenizer.mask_token} you"
token_ids = tokenizer.encode(sentence, return_tensors='pt')
print(token_ids)
# tensor([[ 101, 1045,  103, 2017,  102]])
# get the position of the masked token
masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

# forward
output = model(token_ids)
last_hidden_state = output[0].squeeze(0)
# only get output for masked token
# output is the size of the vocabulary
mask_hidden_state = last_hidden_state[masked_position]
# convert to probabilities (softmax)
# giving a probability for each item in the vocabulary
probs = sm(mask_hidden_state)

# get probability of token 'hate'
hate_id = tokenizer.convert_tokens_to_ids('hate')
print('hate probability', probs[hate_id].item())
# hate probability 0.008057191967964172

# get probability of token 'love'
love_id = tokenizer.convert_tokens_to_ids('love')
print('love probability', probs[love_id].item())
# love probability 0.6704086065292358

# get probability of token 'reprimand' (?)
reprimand_id = tokenizer.convert_tokens_to_ids('reprimand')
# reprimand is not in the vocabulary, so it needs to be split into subword units
print(tokenizer.convert_ids_to_tokens(reprimand_id))
# [UNK]

reprimand_id = tokenizer.encode('reprimand', add_special_tokens=False)
print(tokenizer.convert_ids_to_tokens(reprimand_id))
# ['rep', '##rim', '##and']
# but how do we now get the probability of a multi-token word in a single-token position?
