import sys
import os
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import DDPMPipeline
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path

# sys path append if strictly necessary for your module structure
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Ensure these imports exist in your project structure
# from evaluate.evaluate_celeba import evaluate
# from models.UNet2DWithCBM_new import DDPMPipelineCBM

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader,test_dataloader, lr_scheduler, model_name="", vae=None):
    # --- 1. Initialize WandB ---
    # It is good practice to check if a run is already active
    if wandb.run is None:
        run = wandb.init(
            entity="yassine-oueslati7726-usi-universit-della-svizzera-italiana",
            project="DDPM-CBM",
            config={
                "architecture": "classfier-celeba",
                "dataset": "CelebA",
                "epochs": config.num_epochs,
            },
        )

    # --- 2. Initialize Accelerator ---
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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

    # --- 3. Prepare System ---
    # Note: VAE is usually kept frozen and not optimized, so we don't always need to 'prepare' it 
    # unless you are fine-tuning it. If it's frozen, move it to device manually.
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    
    # Move VAE to device manually if not prepared (assuming inference only for VAE)
    if vae is not None:
        vae.to(accelerator.device)
        vae.eval() 

    global_step = 0

    best_accuracy = 0.0
    # --- 4. Training Loop ---
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in progress_bar:
            with accelerator.accumulate(model):
                # Get data
                images = batch["images"].to(accelerator.device)
                concepts = batch["attributes"].to(accelerator.device).float() # Ensure float for BCE Loss

                # Encode images (VAE Frozen)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Add Noise (Classifier Robustness)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Forward Pass
                outputs = model(noisy_latents)

                # --- FIX: Loss Calculation ---
                # Use Binary Cross Entropy for Multi-Label (CelebA)
                # This treats each concept output as an independent probability
                loss = F.binary_cross_entropy_with_logits(outputs, concepts)

                # Backward
                accelerator.backward(loss)

                # Optimizer
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # loging the loss to wandb
            run.log({"train/loss": loss.detach().item(), "train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

        # --- 5. Evaluation Loop ---
        model.eval()
        total_attributes = 0
        correct_attributes = 0
        
        print(f"Running evaluation for Epoch {epoch+1}...")
        
        with torch.no_grad():
            for batch in test_dataloader: # Note: Usually you should evaluate on a separate validation set
                images = batch["images"].to(accelerator.device)
                concepts = batch["attributes"].to(accelerator.device).float()

                # Encode
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Forward (Clean latents usually for eval, or noisy if testing robustness)
                outputs = model(latents)

                # --- FIX: Extract Labels ---
                # 1. Sigmoid to get probabilities (0.0 to 1.0)
                probs = torch.sigmoid(outputs)
                # 2. Threshold > 0.5 to extract labels (0 or 1)
                predicted = (probs > 0.5).float()

                # Calculate Accuracy (Element-wise)
                # total elements = batch_size * num_concepts
                total_attributes += concepts.numel() 
                correct_attributes += (predicted == concepts).sum().item()

        # Calculate percentage of correctly predicted attributes
        accuracy = 100 * correct_attributes / total_attributes
        print(f"Epoch {epoch+1} Attribute Accuracy: {accuracy:.2f}%")
        
        accelerator.log({"val_accuracy": accuracy}, step=global_step)

        # --- 6. Save the best model 
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if accelerator.is_main_process:
                save_path = os.path.join(config.output_dir, f"{model_name}_best.pt")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f"New best model saved at {save_path} with accuracy: {accuracy:.2f}%")
                run.log({"val_accuracy": accuracy}, step=global_step)
    run.finish()
