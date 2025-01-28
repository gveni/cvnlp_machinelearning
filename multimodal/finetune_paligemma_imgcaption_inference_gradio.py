"""
Image captioning and description generator using mix PaliGemma model
"""
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import gradio as gr

# Define model IDs to be used
pretrained_model_id = "google/paligemma-3b-pt-224"
finetuned_model_id = "pyimagesearch/finetuned_paligemma_vqav2_small"
mix_model_id = "google/paligemma-3b-mix-224"

# Load models and processors based on above model IDs
mix_model = PaliGemmaForConditionalGeneration.from_pretrained(mix_model_id)
mix_processor = AutoProcessor.from_pretrained(mix_model_id) 

# Image captioning and description generator
def process_image(image, prompt):
    inputs = mix_processor(image.convert("RGB"), prompt, return_tensors="pt")
    try:
        output = mix_model.generate(**inputs, max_new_tokens=20)
        decoded_output = mix_processor.decode(output[0], skip_special_tokens=True)
        return decoded_output[len(prompt):]
    except IndexError as e:
        print(f"IndexedError: {e}")
        return "An error occured during processing"

# Build an user interface with Gradio
inputs = [
    gr.Image(type="pil"),
    gr.Textbox(label="Prompt", placeholder="Ask Your question here")
]
outputs = gr.Textbox(label="Answer")

demo = gr.Interface(fn=process_image, inputs=inputs, outputs=outputs, title="Image captioning using mix PaliGemma model", description="Upload an image and get caption that describes it")

demo.launch(debug=True)