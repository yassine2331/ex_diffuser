import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from configs.diffusion_config import TrainingConfig
#from models.UNet2DWithCBM import UNet2DWithCBM
from models.UNet2DWithCBM_new import UNet2DWithCBM
from models.CBM import CBM_new

from data.dataloader import get_dataloader
from train.trainCBM import train_loop

from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch




def run():
    model_name = sys.argv[1]

    config = TrainingConfig()
    model = UNet2DWithCBM(config,CBM_new)
    dataloader = get_dataloader(config,concept=config.concept)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler,model_name=model_name)

if __name__ == "__main__":
    run()