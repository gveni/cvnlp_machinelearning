import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

pretrained_model_id = "google/paligemma-3b-pt-224"
finetuned_model_id = "pyimagesearch/finetuned_paligemma_vqav2_small"
processor = AutoProcessor.from_pretrained(pretrained_model_id) 
finetuned_model = PaliGemmaForConditionalGeneration.from_pretrained(finetuned_model_id)

prompt = "What's behind the cat"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(raw_image.convert("RGB"), prompt, return_tensors="pt")
output = finetuned_model.generate(**inputs, max_new_tokens=20)
print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])

