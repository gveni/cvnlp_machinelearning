# Preprocessing with tokenizer
from torch.nn import functional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setting up tokenizer (preprocessing), model specific for sentence classification task
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Setting up raw inputs
raw_text = ["This is an economic crunch time",
            "Try to stash cash, otherwise you will be doomed",
            "Hope for a silver lining soon",
            "Your future will be bright"]
inputs = tokenizer(raw_text, padding=True, truncation=True, return_tensors="pt")
print(inputs)
outputs = model(**inputs)
print(outputs.logits)

# Postprocessing the model output
predictions = functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)
