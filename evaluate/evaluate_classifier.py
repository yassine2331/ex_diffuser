import sys
import os

# define project_root based on the script's location to make paths robust.
# this ensures that imports and data paths work correctly regardless of
# where you run the script from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.dataloader import get_dataloader_test

from accelerate import Accelerator
from diffusers import DDPMPipeline
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from evaluate.evaluateCBM import evaluate
from models.UNet2DWithCBM import DDPMPipelineCBM

def evaluate_accuracy(config, model, noise_scheduler, test_dataloader, classifier=None):
    # Initialize accelerator and tensorboard logging
    if model is not None:
        model.eval()
    if classifier is not None:
        classifier.eval()
        
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

    # Prepare everything, being careful about None values
    to_prepare = []
    if model is not None:
        to_prepare.append(model)
    if classifier is not None:
        to_prepare.append(classifier)
    to_prepare.append(test_dataloader)
    
    prepared_objects = accelerator.prepare(*to_prepare)
    
    prepared_iter = iter(prepared_objects)
    if model is not None:
        model = next(prepared_iter)
    if classifier is not None:
        classifier = next(prepared_iter)
    test_dataloader = next(prepared_iter)


    global_step = 0
    accuracies = []
    classifier_accuracies = []
    
    progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for n in range(1, 1000, 100):
            correct = 0
            total = 0
            classifier_correct = 0
            classifier_total = 0
            
            for step, batch in enumerate(test_dataloader):
                clean_images = batch["images"]
                concepts = batch.get("concepts", None)  # Now getting concept vectors
                
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                timesteps = torch.randint(
                    n - 1, n, (bs,), device=clean_images.device,
                    dtype=torch.int64
                )
               
                # Add noise to the clean images
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                total_loss = 0
                diffusion_loss = 0
                concept_loss = 0
                
                if model is not None:
                    with accelerator.accumulate(model):
                        # Forward pass with concept return
                        outputs = model(noisy_images, timesteps, interventions=concepts, encoder_hidden_states=clean_images, return_dict=False)
                        noise_pred = outputs[0]
                        concept_pred = outputs[1]
                        
                        diffusion_loss = F.mse_loss(noise_pred, noise)
                        
                        if concepts is not None:
                            concept_loss = F.binary_cross_entropy(concept_pred, concepts.float())
                            _, predictions = concept_pred.max(1)
                            correct += (predictions == torch.argmax(concepts.float(), dim=1)).sum().item()
                            total += concepts.size(0)
                            
                        total_loss = diffusion_loss + 0.45 * concept_loss
                
                if classifier is not None:
                    concept_pred = classifier(noisy_images)
                    if concepts is not None:
                        _, predictions = concept_pred.max(1)
                        classifier_correct += (predictions == torch.argmax(concepts.float(), dim=1)).sum().item()
                        classifier_total += concepts.size(0)

                progress_bar.update(1)
                logs = {}
                if model is not None:
                    logs.update({
                        "loss": total_loss.detach().item() if isinstance(total_loss, torch.Tensor) else total_loss,
                        "diffusion_loss": diffusion_loss.detach().item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss,
                        "concept_loss": concept_loss.detach().item() if isinstance(concept_loss, torch.Tensor) else concept_loss,
                        "step": global_step,
                    })
                    if total > 0:
                        logs["accuracy"] = 100.0 * correct / total
                
                if classifier is not None and classifier_total > 0:
                    logs["classifier_accuracy"] = 100.0 * classifier_correct / classifier_total
                
                if logs:
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                global_step += 1
                
            if total > 0:
                accuracies.append(100.0 * correct / total)
                print(f"noise n= {n} : {100.0 * correct / total:.2f}%")

            if classifier_total > 0:
                classifier_accuracies.append(100.0 * classifier_correct / classifier_total)
                print(f"classifier noise n= {n} : {100.0 * classifier_correct / classifier_total:.2f}%")

    return accuracies, classifier_accuracies
    

if __name__ == "__main__":
    from models.CBM import CBM_new
    from models.UNet2DWithCBM_new import UNet2DWithCBM
    from configs.diffusion_config import AttentionConfig
    from diffusers import DDPMScheduler
    from classifier.classifier import LeNet
    from PIL import Image
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    #creating the model
    config = AttentionConfig()
    # by setting an absolute path for data_dir, we avoid 'Permission denied' errors
    # that can happen when using relative paths from different working directories.
    config.data_dir = os.path.join(project_root, 'data')
    
    test_dataloader = get_dataloader_test(config,concept=config.concept)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    classifier = LeNet()
    model = UNet2DWithCBM(config,CBM_new) #CBM,Intervention
    #loading he model 

    classifier.load_state_dict(torch.load("/teamspace/studios/this_studio/project/classifier/best_model.pth", weights_only=True)) 
    model.load_state_dict(torch.load("/teamspace/studios/this_studio/project/experiments/MNIST/concept_model_crossAttention_4.pt", weights_only=True)['model_state_dict'])
    # make sure to pass the classifier if you have one
    accuracy, classifier_accuracy = evaluate_accuracy(config, model, noise_scheduler, test_dataloader,classifier)
    # Loss Plot
    print(accuracy)
    print(classifier_accuracy)
    # Data
    epochs_range = range(1, 1001, 100)
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(epochs_range, accuracy, label='Model Accuracy per noise')
    plt.plot(epochs_range, classifier_accuracy, label='Classifier Accuracy per noise')

    # Adding values at each point
    for i, acc in enumerate(accuracy):
        plt.text(epochs_range[i], acc, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    for i, acc in enumerate(classifier_accuracy):
        plt.text(epochs_range[i], acc, f'{acc:.2f}', ha='center', va='top', fontsize=8)

    # Formatting plot
    plt.gca().invert_xaxis()
    plt.xlabel('Noise')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy per noise')

    # Save the figure
    plt.savefig("Accuracy_per_noise.png")