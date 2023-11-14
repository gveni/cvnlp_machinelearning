# Load model directly
import re
from string import punctuation
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

text = "(John3,"
text_nonum = re.sub(r'[0-9]', '', text)
text_nopunc = text_nonum.strip(punctuation)
print(text_nopunc)
tokenizer = AutoTokenizer.from_pretrained("padmajabfrl/Gender-Classification")
inputs = tokenizer(text_nopunc, return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained("padmajabfrl/Gender-Classification")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_gender = logits.argmax().item()
predicted_gender_score = torch.softmax(logits, dim=1).numpy().max()
print(str(text_nopunc), "is", model.config.id2label[predicted_gender], "and its score is", predicted_gender_score)