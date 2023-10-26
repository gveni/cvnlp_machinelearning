# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


text = "Jemmimah"
tokenizer = AutoTokenizer.from_pretrained("padmajabfrl/Gender-Classification")
inputs = tokenizer(text, return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained("padmajabfrl/Gender-Classification")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_gender = logits.argmax().item()
print(str(text), "is", model.config.id2label[predicted_gender])