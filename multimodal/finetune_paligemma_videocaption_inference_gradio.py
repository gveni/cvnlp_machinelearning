"""
Video captioning using Google's mix PaliGemma model 
"""
import requests
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import gradio as gr

# Define model IDs to be used
pretrained_model_id = "google/paligemma-3b-pt-224"
finetuned_model_id = "pyimagesearch/finetuned_paligemma_vqav2_small"
mix_model_id = "google/paligemma-3b-mix-224"

# Load models and processors based on above model IDs
finetuned_model = PaliGemmaForConditionalGeneration.from_pretrained(finetuned_model_id)
mix_model = PaliGemmaForConditionalGeneration.from_pretrained(mix_model_id)
pretrained_processor = AutoProcessor.from_pretrained(pretrained_model_id) 
mix_processor = AutoProcessor.from_pretrained(mix_model_id) 

# Frame extraction for video caption
def extract_frames(video_filepath, frame_interval=1):
    video_capture = cv2.VideoCapture(video_filepath)
    frames = []
    success, image = video_capture.read()
    count = 0
    
    while success:
        if count % frame_interval == 0:  # Only grab frames at specific interval to reduce computational load
            frames.append(image)
        success, image = video_capture.read()
        count += 1
    video_capture.release()
    return frames    

def process_video(video, prompt):
    frames = extract_frames(video, frame_interval=10)
    
    captions = []
    for frame in frames:
        # Convert frame to RGB format and transform to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = mix_processor(image.convert("RGB"), prompt, return_tensors="pt")
        try:
            output = mix_model.generate(**inputs, max_new_tokens=20)
            decoded_output = mix_processor.decode(output[0], skip_special_tokens=True)
            captions.append(decoded_output[len(prompt):])
        except IndexError as e:
            print(f"IndexError: {e}")
            captions.append("Error processing frame")

    return " ".join(captions)

# Build an user interface with Gradio
inputs = [
    gr.Video(label="Upload Video"),
    gr.Textbox(label="Prompt", placeholder="Ask Your question here")
]
outputs = gr.Textbox(label="Answer")

demo = gr.Interface(fn=process_video, inputs=inputs, outputs=outputs, title="Video caption using mix PaliGemma model", description="Upload an image and get captions that describes it")

demo.launch(debug=True)