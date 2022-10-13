from ipywidgets.widgets.interaction import Dropdown
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda") 

device = "cuda"
generator = torch.Generator(device=device)
seed = generator.seed()
print(f"The seed for this generator is: {seed}")

latents1 = torch.randn(1,4,64,64)

def improve_image(img):
  url = 'https://hf.space/embed/NotFungibleIO/GFPGAN/+/api/predict'
  request_objt = {
      "data":[gr.processing_utils.encode_pil_to_base64(img),'v1.3',20]}
  return requests.post(url, json=request_objt).json()

def generate(celebrity, setting):
  prompt = 'A movie potrait of' + celebrity + 'sterring in' + setting
  image = pipe(prompt,
              guidance_scale=20, 
              num_inference_steps=100,
              latents=latents1).images[0]
  image = improve_image(image)
  image = gr.processing_utils.decode_base64_to_image(image['data'][0])
  return image

title="Face-id Application Demo"
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
                       'Star Wars'])],
  outputs='image',
  title=title,
  description=description,
  article=article
).launch()
