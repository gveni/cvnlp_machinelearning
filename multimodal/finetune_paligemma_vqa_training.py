"""
Fine-tune PaliGemma with QLoRA for Visual Question Answering
Tutorial URL; https://pyimagesearch.com/2024/12/02/fine-tune-paligemma-with-qlora-for-visual-question-answering/
"""

from huggingface_hub import login
login()

import torch
import requests
from PIL import Image
from datasets import load_dataset
from peft import get_peft_model, LoraConfig  # create a model with parameter-efficient fine-tuning; low-rank adaptation method for fine-tuning
# Simplify training and evaluation of models by using the following classes and set the training configuration
from transformers import Trainer   
from transformers import TrainingArguments
from transformers import PaliGemmaProcessor  # Prepare inputs for PaliGemma model and manage preprocessing tasks
from transformers import BitsAndBytesConfig  # Optimize memory usage and computational efficiency during LLM training
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

device = "cuda"
pretrained_model_id = "google/paligemma-3b-pt-224"

# Load and inspect dataset
ds = load_dataset("merve/vqav2-small")
ds_dataframe = ds["validation"].to_pandas()

# Split the dataset
split_ds = ds["validation"].train_test_split(test_size=.05)
train_ds = split_ds["train"]
test_ds= split_ds["test"]

# Preprocess the dataset
processor = PaliGemmaProcessor.from_pretrained(pretrained_model_id)
def collate_fn(examples):
    texts = [f"<image> <bos> answer {example['question']}" for example in examples]
    labels = [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

# Fine-tune only the decoder (dataset closely resembles paligemma's pretrined dataset)
model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained_model_id, torch_dtype=torch.bfloat16).to(device)
# Freezing vision model parameter weights
for param in model.vision_tower.parameters():
    param.requires_grad = False
# Freeze multi-modal projector parameter weights
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False
    
# Optimize memory by using 4-but quantization 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Improve model efficiency by setting up QLoRa configuration
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained_model_id, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define training arguments using HuggingFace's Trainer API
args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir="finetuned_paligemma_vqav2_small",
    bf16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to=["tensorboard"]
)

# Start training
trainer = Trainer(
    model = model,
    train_dataset=train_ds,
    data_collator = collate_fn,
    args=args
)
trainer.train()

# Push the fine-tuned PaliGemma model to the Hub
# trainer.push_to_hub()  # currently not working due to write permission issues

#prompt = "What is behind cat?"
#image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"
#raw_img = Image.open(requests.get(image_file, stream=True).raw)
#inputs = processor(raw_img.convert("RGB"), prompt, return_tensors="pt")
#output = pretrained_model.generate(**inputs, max_new_tokens=20)
#print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])