# app.py
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image

# Title of the Streamlit app
st.title("Text-to-Image Generator using Stable Diffusion")

# Input from the user for the text prompt
prompt = st.text_input("Enter a text prompt:", "A fantasy landscape with mountains and a castle in the background")

# Generate the image when the button is clicked
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Load the Stable Diffusion model
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Use GPU if available

        # Generate the image from the prompt
        image = pipe(prompt).images[0]

        # Display the generated image
        st.image(image, caption="Generated image from the prompt", use_column_width=True)

        # Optionally save the image
        image.save("generated_image.png")
        st.success("Image generated successfully and saved as 'generated_image.png'")
