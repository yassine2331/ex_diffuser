import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.diffusion_config import TrainingConfig
from models.diffusion_model import create_diffusion_model
from data.dataloader import get_dataloader
from train.train_diffusion import train_loop
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch

def run():
    config = TrainingConfig()
    model = create_diffusion_model(config)
    dataloader = get_dataloader(config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)

if __name__ == "__main__":
    run()
