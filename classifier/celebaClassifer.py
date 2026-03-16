import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import DownBlock2D, ResnetBlock2D

class DiffusersLatentClassifier(nn.Module):
    def __init__(self, 
                 input_channels=16, 
                 selected_attributes= [], # Standard CelebA has 40 binary attributes
                 block_out_channels=(64, 128, 256), # Increasing depth
                 layers_per_block=2):
        super().__init__()

        self.input_channels = input_channels
        num_classes = len(selected_attributes) if selected_attributes else 40
        # 1. Project Input Channels
        # We need to project the 16 latent channels to the size expected by the first block
        self.conv_in = nn.Conv2d(input_channels, block_out_channels[0], kernel_size=3, padding=1)

        # 2. Build the Downscaling Backbone using Diffusers
        # This gives you the "Conv and Skip Connections" you asked for.
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        
        for i, out_channels in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = out_channels
            
            is_final_block = (i == len(block_out_channels) - 1)
            
            # DownBlock2D consists of ResNet blocks (skip connections) + Downsample
            block = DownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None, # We are not doing diffusion, so no time embedding needed
                dropout=0.0,
                num_layers=layers_per_block,
                resnet_eps=1e-6,
                resnet_act_fn="swish",
                resnet_groups=8,     # GroupNorm is standard in Diffusers
                add_downsample=not is_final_block, # Downsample on all but the last block? Adjust as needed.
            )
            self.down_blocks.append(block)

        # 3. Classification Head
        # Global Average Pooling to handle whatever spatial size remains
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()
        
        # Final Linear Layer for CelebA Classification
        self.classifier = nn.Linear(block_out_channels[-1], num_classes)

    def forward(self, x):
        # x shape: (Batch, 16, 16, 16)
        
        # Initial projection
        x = self.conv_in(x)
        
        # Pass through diffusers DownBlocks
        # Note: diffusers blocks usually expect `temb` (time embeddings). 
        # Since this is a classifier, we pass None.
        for block in self.down_blocks:
            x, _ = block(x, temb=None) 

        # x shape is now likely (Batch, 256, 4, 4) or similar depending on depth
        
        # Pooling and Flattening
        x = self.global_pool(x)
        x = self.flatten(x)
        #print("Flattened feature shape:", x.shape)  # Debug: Check feature shape before classifier
        # Prediction
        logits = self.classifier(x)
        
        return logits

# --- Example Usage ---

# 1. Instantiate the model
# model = DiffusersLatentClassifier(input_channels=16)

# # 2. Create dummy data representing your VAE Latents
# # Shape: (Batch_Size, Channels, Height, Width) -> (8, 16, 16, 16)
# dummy_latents = torch.randn(8, 16, 16, 16)

# # 3. Forward pass
# output = model(dummy_latents)

# print(f"Input Latent Shape: {dummy_latents.shape}")
# print(f"Classifier Output Shape: {output.shape}") # Should be [8, 40]
# #print(f"Classifier Output: {output}")
