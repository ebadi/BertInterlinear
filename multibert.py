# https://github.com/gdario/learning_transformers/blob/master/src/bert_example.py

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index1 = 8
tokenized_text[masked_index1] = '[MASK]'

masked_index2 = 11
tokenized_text[masked_index2] = '[MASK]'


# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Define sentence A and B indices associated to 1st and 2nd sentences (see
# paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Predict the hidden states
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()



with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs[0]

# Predict the masked word
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

predicted_index1 = torch.argmax(predictions[0, masked_index1]).item()
predicted_token1 = tokenizer.convert_ids_to_tokens([predicted_index1])[0]
print('The predicted token is: {}'.format(predicted_token1))

predicted_index2 = torch.argmax(predictions[0, masked_index2]).item()
predicted_token2 = tokenizer.convert_ids_to_tokens([predicted_index2])[0]
print('The predicted token is: {}'.format(predicted_token2))

