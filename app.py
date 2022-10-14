from ipywidgets.widgets.interaction import Dropdown
import gradio as gr
import torch
import os
from diffusers import StableDiffusionPipeline
import base64
import io

auth_token = os.environ.get("auth_token")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token)
pipe = pipe.to(device) 

device = "cuda"
generator = torch.Generator(device=device)
seed = generator.seed()
print(f"The seed for this generator is: {seed}")

latents1 = torch.randn(1,4,64,64)

def convert_image_2string(image):
  out_buffer  = io.BytesIO()
  image.save(out_buffer, format="PNG")
  out_buffer .seek(0)
  base64_bytes = base64.b64encode(out_buffer .read())
  base64_str = base64_bytes.decode("ascii")
  return base64_str

def improve_image(img):
  url = 'https://hf.space/embed/abidlabs/GFPGAN/+/api/predict'
  request_objt = {
      "data":[convert_image_2string(img),'v1.3',20]}
  return requests.post(url, json=request_objt).json()

def generate(celebrity, setting):
  prompt = 'A movie potrait of' + celebrity + 'sterring in' + setting
  image = pipe(prompt,
              guidance_scale=2, 
              num_inference_steps=50,
              latents=latents1).images[0]
  image = improve_image(image)
  image = gr.processing_utils.decode_base64_to_image(image['data'][0])
  return image

title="🖼️Movie poster Generator (Diffusion Model) Demo"
description = "Upload similar/different images to compare Image similarity for face-id demo"
article = """
            - Select an image from the examples provided as demo image
            - Click submit button to make Image classification
            - Click clear button to try new Image for classification
          """

gr.Interface(
  fn=generate,
  inputs=[gr.Textbox(),
          gr.Dropdown(['terminator', 
                     'matrix',
                     'Gladiator',
                     'The Godfather',
                     'The Dark Knight',
                     'The Lord of the Rings',
                     'Star Wars'])
          ],
  outputs='image',
  title=title,
  description=description,
  article=article
).launch()
