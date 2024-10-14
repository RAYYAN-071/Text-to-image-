# Step 1: Install necessary libraries
!pip install streamlit diffusers transformers accelerate
!pip install torch torchvision

# Step 2: Import libraries
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image

# Step 3: Define the Streamlit app
def main():
    st.title("Text-to-Image Generator using Stable Diffusion")
    
    # Prompt input from user
    prompt = st.text_input("Enter a text prompt:", "A fantasy landscape with mountains and a castle in the background")
    
    # Button to generate image
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            # Load the model (if not already loaded)
            model_id = "CompVis/stable-diffusion-v1-4"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")  # Ensure the model is using GPU
            
            # Generate the image from the prompt
            image = pipe(prompt).images[0]
            
            # Display the image
            st.image(image, caption="Generated image from the prompt", use_column_width=True)
            
            # Option to save the image
            image.save("generated_image.png")
            st.success("Image generated successfully and saved as 'generated_image.png'")

# Step 4: Run the Streamlit app
if __name__ == '__main__':
    main()
