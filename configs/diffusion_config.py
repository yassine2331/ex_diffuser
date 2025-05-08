from dataclasses import dataclass

@dataclass
class TrainingConfig:
    root_dir = ".."
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 5
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "MNIST"
    push_to_hub = False
    hub_model_id = "<your-username>/<my-awesome-model>"
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
    concept= True 
    #model
    image_size = 32
    context_dim = 8 # dimention of centext in the CBM paper 
    skip_context = False
    num_concepts = 10
    in_channels=1
    out_channels=1
    layers_per_block=2
    block_out_channels=(128, 128, 256, 256, 512)
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
