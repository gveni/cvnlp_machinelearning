import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# instantiate bert-tokenizier
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

input_text_seq1 = "Using a transformer network is simple"
input_text_seq2 = "I’ve been waiting for a HuggingFace course my whole life."
input_text_seq3 = "I hate this so much!"
tokenizer(input_text_seq1)

# elaborative tokenization approach (two steps: (1) convert to tokens, (2) extract token-IDs from model vocabulary)
token1 = tokenizer.tokenize(input_text_seq1)
token1_id = tokenizer.convert_token_to_ids(token1)
print("Number of tokens in token1", len(token1_id))
token2 = tokenizer.tokenize(input_text_seq2)
token2_id = tokenizer.convert_token_to_ids(token2)
print("Number of tokens in token2", len(token2_id))
token3 = tokenizer.tokenize(input_text_seq3)
token3_id = tokenizer.convert_token_to_ids(token3)
print("Number of tokens in token3", len(token3_id))

#input_ids = torch.tensor([token_id])
#print("Input IDs:", token_ids)

# decoding (converting token-IDs back to input text sequence)
#output_text_seq = bert_tokenizer.decode(token_ids)
#print(output_text_seq)
#model_output = model(input_ids)
#print("Logits:", model_output.logits)
