from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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



def evaluate(config, epoch, pipeline, full_model=None):
    if full_model is not None:
        full_model.eval()
    
    with torch.no_grad():
        generator = torch.Generator(device='cuda').manual_seed(config.seed)
        output = pipeline(
            batch_size=config.eval_batch_size,
            generator=generator,
            return_dict = False
        )
        images = output[0]
        concepts = output[1].cpu().numpy()

    if not isinstance(images[0], Image.Image):
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]

            # Combine each image with its corresponding concept distribution
    combined_images = []

    for img, concept in zip(images, concepts):
        # Convert concept probs to NumPy if it's a tensor
        if torch.is_tensor(concept):
            concept = concept.cpu().numpy()

        # Create distribution plot
        plt.figure(figsize=(2, 2))
        plt.bar(range(len(concept)), concept)
        plt.ylim(0, 1)
        plt.xticks(range(len(concept)))
        plt.title("Concepts")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        buf.seek(0)
        dist_img = Image.open(buf).convert("RGB")

        # Resize both to same height
        img = img.convert("RGB")
        #dist_img = dist_img.resize((img.height, img.height))
        img = img.resize((dist_img.height, dist_img.height))  # keep square

        # Concatenate side by side
        combined = Image.new("RGB", (img.width + dist_img.width, img.height))
        combined.paste(img, (0, 0))
        combined.paste(dist_img, (img.width, 0))
        combined_images.append(combined)

    
    # Create and show/save the final grid
    image_grid = make_image_grid(combined_images, rows=4, cols=4)

    plt.figure(figsize=(12, 12))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save the grid
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png") 