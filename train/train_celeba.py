import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from accelerate import Accelerator
from diffusers import DDPMPipeline
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import torch
import os
import torch.nn.functional as F
from evaluate.evaluate_celeba import evaluate
from models.UNet2DWithCBM_new import DDPMPipelineCBM
import wandb



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,model_name="",vae=None):
    # Using wandb to log training metrics 

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="yassine-oueslati7726-usi-universit-della-svizzera-italiana",
        # Set the wandb project where this run will be logged.
        project="DDPM-CBM",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "DDPM-CBM",
            "dataset": "CelebA",
            "epochs": 150,
        },
    )

    
    
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,4
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("concept_diffusion")

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler , vae= accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler,vae
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        #for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(train_dataloader):
            
             # Get the clean images and concepts
            
            clean_images = batch["images"]
            concepts = batch["attributes"]
            #concepts = batch.get("concepts", None)  # Now getting concept vectors
            vae.eval()
            with torch.no_grad():
                latent_dist= vae.encode(clean_images).latent_dist
                z = latent_dist.sample() 
                z = z * vae.config.scaling_factor
                #recon = vae.decode(z)
            #print("vae. ", z.shape)
            # Sample noise to add to the images
            noise = torch.randn(z.shape, device=z.device)
            bs = z.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=z.device,
                dtype=torch.int64
            )

            # Add noise to the clean images
            noisy_images = noise_scheduler.add_noise(z, noise, timesteps)

            with accelerator.accumulate(model):
                # Forward pass with concept return
                outputs = model(noisy_images, timesteps,interventions = concepts, return_dict=False)
                noise_pred = outputs[0]
                concept_pred = outputs[1]
                
                # Diffusion loss
                diffusion_loss = F.mse_loss(noise_pred, noise)
                
                # Concept supervision loss if concepts are provided
                concept_loss = 0
                if concepts is not None:
                    concept_loss = F.binary_cross_entropy(concept_pred, concepts.float())
                
                # Combined loss
                total_loss = diffusion_loss +  0.1*concept_loss
                
                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": total_loss.detach().item(),
                "diffusion_loss": diffusion_loss.detach().item(),
                "concept_loss": concept_loss.detach().item() if concepts is not None else 0,
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            run.log({"total_loss": total_loss.detach().item(),
                      "diffusion_loss": diffusion_loss.detach().item(),
                      "concept_loss": concept_loss.detach().item() if concepts is not None else 0,
                      "lr": lr_scheduler.get_last_lr()[0],})
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipelineCBM(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler
            )
            

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, accelerator.unwrap_model(model),accelerator.unwrap_model(vae))

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
                    torch.save({
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(config.output_dir, f"concept_model_{model_name}_{epoch}.pt"))
    
    run.finish()
