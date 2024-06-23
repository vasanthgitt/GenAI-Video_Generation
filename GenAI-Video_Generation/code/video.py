import gradio as gr
from transformers import pipeline
import google.generativeai as genai
import re
import os
import requests
import io
from PIL import Image
from gtts import gTTS
from moviepy.editor import *
from textwrap import wrap

# Generative model setup
API_KEY = ''
genai.configure(api_key=API_KEY)  # Replace with your actual API key
generation_config = {"temperature": 0.9, "max_output_tokens": 2048, "top_k": 1, "top_p": 1}

# Use the appropriate generative model, e.g., "gemini-pro" (replace with the actual model name)
model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

# Summarization model setup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": " "}

# Function to remove asterisks from text
def remove_asterisks(text):
    return text.replace('*', '')

# Function to wrap text
def wrap_text(text, max_width, font_size):
    lines = wrap(text, width=int(max_width / font_size))
    return '\n'.join(lines)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def generate_video(prompt):
    # Generate detailed content based on the prompt using the generative model
    response_content = model.generate_content(prompt)

    # Extract the response text
    response_text = ''.join([chunk.text for chunk in response_content])

    # Use the summarization pipeline to generate a summary of the response text
    response_summary = summarizer(response_text, max_length=1000, min_length=200, do_sample=False)

    # Extract the generated summary text
    summary_text = response_summary[0]['summary_text']

    with open("summary_text.txt", "w") as file:
        file.write(summary_text.strip())

    # Read the text file
    with open("summary_text.txt", "r") as file:
        text = file.read()

    # Split the text by , and .
    paragraphs = re.split(r"[,.]", text)

    # Create Necessary Folders
    os.makedirs("audio", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    # Loop through each paragraph and generate an image for each
    i = 1
    for para in paragraphs[:-1]:
        # Call the Hugging Face model to generate an image based on the paragraph
        image_bytes = query({
            "inputs": para.strip(),
        })

        # Save the image to the "images" folder
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f"images/image{i}.jpg")

        # Create gTTS instance and save to a file
        tts = gTTS(text=para, lang='en', slow=False)
        tts.save(f"audio/voiceover{i}.mp3")

        # Load the audio file using moviepy
        audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
        audio_duration = audio_clip.duration

        # Load the image file using moviepy
        image_clip = ImageClip(f"images/image{i}.jpg").set_duration(audio_duration)

        # Wrap text into multiple lines
        wrapped_text = wrap_text(para, image_clip.w, 30)

        # Calculate the position dynamically based on the length of the text
        text_height = TextClip(wrapped_text, fontsize=20, color="white").h
        bottom_margin = 50
        text_clip = TextClip(wrapped_text, fontsize=15, color="white", bg_color="black")
        text_clip = text_clip.set_position(('center', image_clip.h - text_height - bottom_margin)).set_duration(audio_duration)
        text_clip = text_clip.crossfadein(1).crossfadeout(1)

        # Use moviepy to create a final video by concatenating
        clip = image_clip.set_audio(audio_clip)
        video = CompositeVideoClip([clip, text_clip])

        # Save the final video to a file
        video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
        i += 1

    # Concatenate all the clips to create a final video
    clips = []
    l_files = os.listdir("videos")
    for file in l_files:
        clip = VideoFileClip(f"videos/{file}")
        clips.append(clip)

    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.write_videofile("final_video.mp4")

    return "final_video.mp4", summary_text


# Interface
iface = gr.Interface(
    fn=generate_video,
    inputs="text",
    outputs= ["video","text"],
    title="VideoGen Model",
    description="Generate a video based on a prompt.",
    examples=[["Explain Photosynthesis"]],
    theme="ParityError/Interstellar"
)
iface.launch()