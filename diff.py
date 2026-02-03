import os
from PIL import Image,ImageDraw
import numpy as np 

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from huggingface_hub import notebook_login

device = 'cpu'
# The absolute path to your local model snapshot
model_path = "/home/xcode/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"# change this to your cache path

# vae = AutoencoderKL()
# unet = UNet2DConditionModel()
# tokenizer = CLIPTokenizer()
# scheduler = LMSDiscreteScheduler()
# text_encoder = CLIPTextModel()


# 1. Load VAE
vae = AutoencoderKL.from_pretrained(
    model_path, 
    subfolder="vae", 
    local_files_only=True
).to(device)

# 2. Load UNet
unet = UNet2DConditionModel.from_pretrained(
    model_path, 
    subfolder="unet", 
    local_files_only=True
).to(device)

# 3. Load Tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    model_path, 
    subfolder="tokenizer", 
    local_files_only=True
)

# 4. Load Text Encoder
text_encoder = CLIPTextModel.from_pretrained(
    model_path, 
    subfolder="text_encoder", 
    local_files_only=True
).to(device)

# 5. Load Scheduler (Configuration only)
scheduler = LMSDiscreteScheduler.from_pretrained(
    model_path, 
    subfolder="scheduler", 
    local_files_only=True
)

def get_text_embeddings(prompt):
    # 1. Tokenize (Notice the .model_max_length)
    text_input = tokenizer(
        prompt, 
        padding='max_length', 
        max_length=tokenizer.model_max_length,
        truncation=True, 
        return_tensors='pt'
    )

    with torch.no_grad():
        # Move inputs to CPU (explicitly)
        text_embeddings = text_encoder(text_input.input_ids.to('cpu'))[0]

    # 2. Create unconditional (negative) embeddings
    uncond_input = tokenizer(
        [''] * len(prompt), 
        padding='max_length',
        max_length=tokenizer.model_max_length, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to('cpu'))[0]

    # 3. Combine both for Classifier-Free Guidance
    # We stack [Negative, Positive]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    return text_embeddings

def produce_latents(text_embeddings,height=512,width=512,num_inference_steps=50,guidance_scale=7.5,latents=None):
    if latents is None:
        latents= torch.randn(text_embeddings.shape[0]//2,unet.in_channels,height//8,width//8)
        # The Dimensions: (Batch, Channels, Height, Width)
        # The numbers inside the parentheses define the shape of this noise tensor. This is where the core logic of Stable Diffusion’s efficiency lies:
        # text_embeddings.shape[0] // 2 (Batch Size): Earlier in your code, you combined your prompt and the "negative prompt" (empty string) into one list. This doubled the batch size. We divide by 2 here to ensure we only generate the actual number of images requested, not double that amount.
        # unet.in_channels (Channels): For Stable Diffusion v1.4/1.5, this value is 4. While a standard image has 3 channels (Red, Green, Blue), the VAE (Variational Autoencoder) compresses those into 4 "latent" channels that contain mathematical patterns rather than raw colors.
        # height // 8 and width // 8 (Compressed Resolution): This is the most important part. The Stable Diffusion VAE has a downsampling factor of 8.
        # If you want a 512x512 image, the latent noise is only 64x64.

        latents=latents.to(device)
        scheduler.set_timesteps(num_inference_steps)
        latents=latents*scheduler.sigmas[0]

        for i,t in tqdm(enumerate(scheduler.timesteps)):
            latent_model_input=torch.cat([latents]*2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input/((sigma**2 +1)*0.5)

            #predict noise from the unet
            noise_prediction= unet(latent_model_input,t,encoder_hidden_states=text_embeddings)['sample']
            #perform guidance
            noise_pred_uncond, noise_pred_text = noise_prediction.chunk(2)
            noise_prediction = noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_prediction, t, latents)['prev_sample']
        return latents
def decode_img_latents(latents):
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        imgs = vae.decode(latents)
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images

def prompt_to_image(prompts,height=512,width=512,num_inference_steps=50,guidance_scale=7.5,latents=None):
    if isinstance(prompts,str):
        prompts=[prompts]

    text_embeddings = get_text_embeddings(prompts)
    latents=produce_latents(text_embeddings,height=height,width=width,latents=latents,
                            num_inference_steps=num_inference_steps,guidance_scale=guidance_scale)
    imgs=decode_img_latents(latents)
    return imgs

if __name__ == "__main__" :
    torch.set_num_threads(5)
    img = prompt_to_image('ghost in the middle of the road', 512, 512, 20)[0]
    plt.imshow(img)
    plt.axis('off') # Hide the x and y numbers
    plt.show()