from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils
import io
# Make a 4×4 grid of the combined (image + chart) pairs
def make_image_grid(images, rows, cols):
    assert len(images) == rows * cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid



def evaluate(config, epoch, pipeline, full_model=None,vae=None):
    """if full_model is not None:
        full_model.eval()"""
    
    
    with torch.no_grad():
        generator = torch.Generator(device='cuda').manual_seed(config.seed)
        output = pipeline(
            batch_size=config.eval_batch_size,
            generator=generator,
            return_dict = False,
            output_type="latent" if vae is not None else "pil"
        )
        images = output[0]
        images = images /vae.config.scaling_factor 
        concepts = output[1].cpu().numpy()
        #mu, logvar = vae.encode(clean_images)
        #z = vae.reparameterize(mu, logvar)
        print("EVAL ", images.shape)

        recon = vae.decode(images).sample
        print("HERE")
        vutils.save_image(recon.cpu().data,os.path.join(config.output_dir, f"samples_{epoch:04d}.png"),normalize=True,nrow=12)
