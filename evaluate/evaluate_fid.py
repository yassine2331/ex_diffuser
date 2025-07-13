import os
import torch
from cleanfid import fid
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
import shutil

def prepare_real_images(path="./real_mnist", num_images=1000):
    """
    downloads mnist and saves a subset of images to a directory.
    this helps avoid re-downloading and processing images every time you evaluate.
    """
    if os.path.exists(path) and len(os.listdir(path)) >= num_images:
        print(f"real images found at {path}, skipping preparation.")
        return
    
    print(f"preparing real images at {path}...")
    dataset = MNIST(root=".", train=True, download=True)
    os.makedirs(path, exist_ok=True)
    for i in range(num_images):
        img, _ = dataset[i]
        # fid expects rgb images, and cleanfid by default uses inceptionv3 which needs 299x299 images.
        img.convert("RGB").resize((299, 299)).save(f"{path}/{i:05d}.png")
    print("done preparing real images.")

def generate_fake_images(pipeline, path, num_images, batch_size, device="cuda"):
    """
    generates images using your diffusion pipeline and saves them.
    this creates the "fake" images to compare against the real ones.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    print(f"generating {num_images} fake images...")
    
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_images:
            current_batch_size = min(batch_size, num_images - generated_count)
            if current_batch_size <= 0:
                break
                
            # we don't set a seed here to get a diverse set of images each time.
            generator = torch.Generator(device=device)
            images = pipeline(
                batch_size=current_batch_size,
                generator=generator,
            ).images

            # some pipelines might return numpy arrays, so we convert to pil images if needed.
            if not isinstance(images[0], Image.Image):
                # denormalize from [0,1] to [0,255] and convert to uint8
                images = [Image.fromarray((img.squeeze() * 255).astype(np.uint8)) for img in images]

            for img in images:
                if generated_count >= num_images:
                    break
                
                # resize and convert to rgb for fid computation.
                img.convert("RGB").resize((299, 299)).save(f"{path}/{generated_count:05d}.png")
                generated_count += 1
    print("done generating fake images.")

def compute_fid_score(pipeline, config):
    """
    this function ties it all together. it prepares the real images,
    generates fakes ones with your model, and then calculates the fid score.
    """
    real_images_path = "./real_mnist"
    fake_images_path = "./fake_mnist"
    
    # you can control these through your config object.
    num_images = config.num_fid_images
    batch_size = config.eval_batch_size
    device = config.device
    
    # first, we need a set of real images to compare against.
    prepare_real_images(path=real_images_path, num_images=num_images)
    
    # next, we generate images from your model.
    """generate_fake_images(
        pipeline, 
        path=fake_images_path, 
        num_images=num_images,
        batch_size=batch_size,
        device=device
    )"""
    
    # finally, we compute the fid score.
    print("computing fid score...")
    score = fid.compute_fid(real_images_path, fake_images_path,dataset_res=299)
    print(f"fid score: {score}")
    return score

if __name__ == '__main__':
    import os
    import sys
    import torch
    from diffusers import DDPMScheduler

    # Set up path to import project modules
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    
    from models.UNet2DWithCBM_new import UNet2DWithCBM
    from models.CBM import CBM_new
    from configs.diffusion_config import AttentionConfig
    from models.UNet2DWithCBM import DDPMPipelineCBM


    class MockConfig:
        num_fid_images = 1000
        eval_batch_size = 64
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_ckpt_path = "/teamspace/studios/this_studio/project/experiments/MNIST/concept_model_crossAttention_4.pt"

    config = MockConfig()
    print(f"Using device: {config.device}")

    # Load model config
    model_config = AttentionConfig()

    # Build model and scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = UNet2DWithCBM(model_config, CBM_new).to(config.device)

    # Load model checkpoint
    print(f"Loading checkpoint from: {config.model_ckpt_path}")
    checkpoint = torch.load(config.model_ckpt_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Wrap model into a diffusion pipeline
    pipeline = DDPMPipelineCBM(unet=model, scheduler=noise_scheduler).to(config.device)


    # Evaluate FID
    compute_fid_score(pipeline, config)
