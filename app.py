from ipywidgets.widgets.interaction import Dropdown
import gradio as gr
import torch
import os
from diffusers import StableDiffusionPipeline
import base64
import io
import requests


auth_token = os.environ.get("auth_token")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token)
pipe = pipe.to(device) 

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

def improve_image(image):
  url1 = 'https://hf.space/embed/NotFungibleIO/GFPGAN/+/api/predict'
  url2 = 'https://hf.space/embed/abidlabs/GFPGAN/+/api/predict'
  request_objt = {
      "data":[f'image/jpeg;base64,{convert_image_2string(image)}',2]}
  return requests.post(url2, json=request_objt).json()

def generate(celebrity, setting):
  prompt = f'A movie poster of {celebrity} in {setting},photorealistic, 4k High Definition by magali villeneuve, jeremy lipkin and michael garmash style' 
  #'A movie potrait of' + celebrity + 'sterring in' + setting
  image = pipe(prompt,
              guidance_scale=20, 
              num_inference_steps=100,
              latents=latents1).images[0]
  image = improve_image(image)
  image = gr.processing_utils.decode_base64_to_image(image['data'][0])
  return image

title="🖼️Movie poster Generator (Diffusion Model) Demo"
description = "Generate amazing photo realistic images of your favourite movie\
 characters starring in movies that did not exist"
article = """
            - Enter the name of your preffered movie character
            - Also select a movie from the posible list of options.
          """

gr.Interface(
  fn=generate,
  inputs=[gr.Textbox(value='Will Smith'),
          gr.Dropdown(['matrix',
                     'Gladiator',
                     'The Godfather',
                     'The Dark Knight',
                     'The Lord of the Rings',
                     'Star Wars'], value='The Godfather')
          ],
  outputs='image',
  title=title,
  description=description,
  article=article
).launch(debug=True)
