
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def evaluate(config, epoch, pipeline, full_model=None):
    if full_model is not None:
        full_model.eval()
    
    with torch.no_grad():
        generator = torch.Generator(device='cuda').manual_seed(config.seed)
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=generator,
        ).images

    if not isinstance(images[0], Image.Image):
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]

    image_grid = make_image_grid(images, rows=4, cols=4)
    plt.figure(figsize=(12, 12))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")



"""from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
import matplotlib.pyplot as plt

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    #SHOW IMAGE
    plt.figure(figsize=(10,10))
    plt.imshow(image_grid)
    plt.show()

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")"""