import sys
import os
import torch
import matplotlib.pyplot as plt

# Add project directory to path to handle relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate.evaluate_classifier import evaluate_accuracy
from data.dataloader import get_dataloader_test
from models.CBM import CBM_new
from models.UNet2DWithCBM_new import UNet2DWithCBM
from configs.diffusion_config import AttentionConfig
from diffusers import DDPMScheduler
from classifier.classifier import LeNet

def main():
    """
    Main function to run the evaluation and plot the results.
    """
    # --- Configuration and Data Loading ---
    config = AttentionConfig()
    test_dataloader = get_dataloader_test(config, concept=config.concept)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # --- Model Initialization ---
    classifier = LeNet()
    cbm_model = UNet2DWithCBM(config, CBM_new)
    
    # --- Model Loading ---
    print("loading model weights...")
    classifier.load_state_dict(torch.load("/teamspace/studios/this_studio/project/classifier/best_model.pth", weights_only=True)) 
    cbm_model.load_state_dict(torch.load("/teamspace/studios/this_studio/project/experiments/MNIST/concept_model_crossAttention_4.pt", weights_only=True)['model_state_dict'])
    print("models loaded successfully.")

    # --- Evaluation ---
    print("starting evaluation...")
    model_accuracy, classifier_accuracy = evaluate_accuracy(config, cbm_model, noise_scheduler, test_dataloader, classifier)
    print("evaluation finished.")
    
    print("\n--- results ---")
    print(f"cbm model accuracy: {[f'{acc:.2f}' for acc in model_accuracy]}")
    print(f"standalone classifier accuracy: {[f'{acc:.2f}' for acc in classifier_accuracy]}")
    
    # --- Plotting ---
    print("\ngenerating plot...")
    noise_range = range(1, 1000, 100)
    
    plt.figure(figsize=(12, 7))
    
    # plot cbm model accuracy
    plt.plot(noise_range, model_accuracy, 'o-', label='CBM Model Accuracy')
    for i, acc in enumerate(model_accuracy):
        plt.text(noise_range[i], acc + 0.5, f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
        
    # plot standalone classifier accuracy
    plt.plot(noise_range, classifier_accuracy, 's--', label='Standalone Classifier Accuracy')
    for i, acc in enumerate(classifier_accuracy):
        plt.text(noise_range[i], acc - 1.5, f'{acc:.2f}', ha='center', va='top', fontsize=9)

    # formatting plot
    plt.gca().invert_xaxis()
    plt.xlabel('Noise Timestep (n)')
    plt.ylabel('Accuracy (%)')
    plt.title('CBM Model vs. Standalone Classifier Accuracy at Different Noise Levels')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(min(min(model_accuracy, default=0), min(classifier_accuracy, default=0)) - 10, 105)
    
    # save the figure
    output_filename = "accuracy_comparison_per_noise.png"
    plt.savefig(output_filename)
    print(f"plot saved to {output_filename}")

if __name__ == "__main__":
    main()