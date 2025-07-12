import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from accelerate import Accelerator

# add project root to path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import get_dataloader_test
from models.UNet2DWithCBM_new import UNet2DWithCBM
from models.CBM import CBM_new
from configs.diffusion_config import AttentionConfig
from diffusers import DDPMScheduler

def get_class_separated_data(dataloader, num_classes):
    """
    Organizes images from the dataloader into a dictionary keyed by class label.
    This helps in easily accessing all images of a specific class for intervention tests.
    """
    print("separating test set by class...")
    class_data = {i: [] for i in range(num_classes)}
    for batch in tqdm(dataloader, desc="organizing data by class"):
        images = batch['images']
        concepts = batch.get("concepts")
        if concepts is None:
            raise ValueError("dataloader must provide 'concepts' for class separation.")
        
        labels = torch.argmax(concepts, dim=1)
        for i in range(len(images)):
            class_label = labels[i].item()
            class_data[class_label].append(images[i])

    for i in range(num_classes):
        if class_data[i]:
            class_data[i] = torch.stack(class_data[i])
        else:
            class_data[i] = torch.tensor([])
    
    print("done separating data.")
    return class_data

def plot_results(results, config, noise_levels):
    """
    Plots and saves the separability evaluation results.
    This helps visualize how well interventions work at different noise levels.
    """
    print(f"\nplotting results...")
    for to_class in range(config.num_classes):
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        
        total_intervention_scores = {n: [] for n in noise_levels}
        
        for from_class in range(config.num_classes):
            if from_class == to_class:
                continue
            
            if from_class in results[to_class] and results[to_class][from_class]:
                accuracies = [results[to_class][from_class].get(n, 0) for n in noise_levels]
                ax.plot(noise_levels, accuracies, marker='o', linestyle='-', label=f'from class {from_class}')
                
                for n_idx, n in enumerate(noise_levels):
                    total_intervention_scores[n].append(accuracies[n_idx])

        avg_scores = [np.mean(total_intervention_scores[n]) if total_intervention_scores[n] else 0 for n in noise_levels]
        ax.plot(noise_levels, avg_scores, marker='x', linestyle='--', color='black', linewidth=2, label='average intervention score')

        ax.set_title(f'intervention separability score to class {to_class}')
        ax.set_xlabel('noise level (t)')
        ax.set_ylabel('intervention accuracy (%)')
        ax.set_ylim(0, 101)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        
        plot_filename = f'separability_to_{to_class}.png'
        plt.savefig(plot_filename)
        print(f"saved plot: {plot_filename}")
        plt.close()

def evaluate_separability(model, scheduler, test_dataloader, config):
    """
    Evaluates the model's ability to be steered from a source class to a target class.
    It systematically tests every combination of source/target classes across different
    noise levels, providing a comprehensive measure of controllabilty.
    """
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    model = accelerator.prepare(model)
    device = accelerator.device

    model.eval()
    
    class_data = get_class_separated_data(test_dataloader, config.num_classes)
    
    noise_levels = list(range(1, 1000, 100))
    
    results = {to_c: {from_c: {n: 0 for n in noise_levels} for from_c in range(config.num_classes)} for to_c in range(config.num_classes)}
    
    num_samples_per_class = 32

    for to_class in tqdm(range(config.num_classes), desc="target class"):
        intervention_concept = F.one_hot(torch.tensor(to_class), num_classes=config.num_classes).float().unsqueeze(0).to(device)

        for from_class in tqdm(range(config.num_classes), desc="source class", leave=False):
            if from_class == to_class:
                continue

            source_images = class_data[from_class]
            if len(source_images) == 0:
                continue

            test_images = source_images[:num_samples_per_class].to(device)
            if len(test_images) == 0:
                continue
            
            for n in tqdm(noise_levels, desc="noise level", leave=False):
                clean_images = test_images
                
                noise = torch.randn_like(clean_images)
                timesteps = torch.full((clean_images.shape[0],), n, device=device, dtype=torch.int64)
                
                noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                
                images_to_denoise = noisy_images
                scheduler.set_timesteps(1000)
                timesteps_to_run = scheduler.timesteps[scheduler.timesteps < n]

                batch_intervention_concept = intervention_concept.repeat(images_to_denoise.shape[0], 1)

                for t in timesteps_to_run:
                    t_tensor = torch.full((images_to_denoise.shape[0],), t.item(), device=device, dtype=torch.int64)
                    with torch.no_grad():
                        noise_pred = model(images_to_denoise, t_tensor, interventions=batch_intervention_concept, return_dict=False)[0]
                    images_to_denoise = scheduler.step(noise_pred, t, images_to_denoise).prev_sample
                
                denoised_images = images_to_denoise

                with torch.no_grad():
                    # Classify the denoised image using the model's concept head at t=0
                    t_zero = torch.zeros(denoised_images.shape[0], device=device, dtype=torch.int64)
                    _, pred_concepts = model(denoised_images, t_zero, interventions=None, return_dict=False)
                    final_classes = torch.argmax(pred_concepts, dim=1)
                    correct_interventions = (final_classes == to_class).sum().item()
                
                accuracy = (correct_interventions / len(test_images)) * 100 if len(test_images) > 0 else 0
                results[to_class][from_class][n] = accuracy
            
            # Print accuracies after each noise loop
            print(f"to->{to_class}, from->{from_class}:")
            acc_str = [f"n={n}: {results[to_class][from_class][n]:.2f}%" for n in noise_levels]
            print(f"  concept model: {', '.join(acc_str)}")

    # plotting the results
    plot_results(results, config, noise_levels)

if __name__ == "__main__":
    config = AttentionConfig()
    
    # load data
    test_dataloader = get_dataloader_test(config, concept=config.concept)
    
    # load models
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = UNet2DWithCBM(config, CBM_new)

    # load pre-trained weights
    model.load_state_dict(torch.load("/teamspace/studios/this_studio/project/experiments/MNIST/concept_model_crossAttention_4.pt")['model_state_dict'])

    evaluate_separability(model, noise_scheduler, test_dataloader, config)