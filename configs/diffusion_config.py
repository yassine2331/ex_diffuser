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
    skip_concept = False
    num_concepts = 10
    in_channels=1
    out_channels=1
    layers_per_block=2
    block_out_channels=(128, 128, 256, 256, 512)
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")



@dataclass
class AttentionConfig:
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
    skip_concept = False
    num_concepts = 10
    num_classes = 10
    in_channels=1
    out_channels=1
    layers_per_block=2
    block_out_channels=(128, 128, 256, 256, 512)
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    cross_attention_dim = 256
    num_attention_heads = 4




@dataclass
class AttentionConfigCIFAR10:
    root_dir = ".."
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 30
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "CIFAR10"
    push_to_hub = False
    hub_model_id = "<your-username>/<my-awesome-model>"
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
    concept= True 
    #model
    image_size = 32
    context_dim = 8 # dimention of centext in the CBM paper 
    skip_concept = True
    num_concepts = 10
    in_channels=3
    out_channels=3
    layers_per_block=2
    block_out_channels=(128, 128, 256, 256, 512)
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    cross_attention_dim = 256
    num_attention_heads = 4





@dataclass
class celebA:
    root_dir = ".."
    train_batch_size = 64
    eval_batch_size = 72 
    num_epochs = 300
    gradient_accumulation_steps = 1
    learning_rate = 6e-5
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 50
    mixed_precision = "fp16"
    output_dir = "CelebA"
    push_to_hub = False
    hub_model_id = "<your-username>/<my-awesome-model>"
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 1265
    concept= True 
    #model
    image_size = 16
    context_dim = 16 # dimention of centext in the CBM paper 
    skip_concept = False
    num_concepts = 17
    in_channels=16
    out_channels=16
    layers_per_block=4
    block_out_channels=(128, 256,256,512)
    down_block_types=( "DownBlock2D", "AttnDownBlock2D","DownBlock2D","AttnDownBlock2D")
    up_block_types=( "AttnUpBlock2D","UpBlock2D",  "AttnUpBlock2D","UpBlock2D")
    cross_attention_dim = None
    num_attention_heads = 4
    selected_attrs = ["Male", "Young", "Smiling", "Eyeglasses", "Big_Lips", "Big_Nose", "Narrow_Eyes", "Black_Hair", "Blond_Hair", "Bangs", "Bald", "No_Beard","Mustache","Mouth_Slightly_Open", "High_Cheekbones", "Oval_Face", "Chubby"]



@dataclass
class celebA_old:
    root_dir = ".."
    train_batch_size = 64
    eval_batch_size = 72 
    num_epochs = 20
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 10
    mixed_precision = "fp16"
    output_dir = "CelebA"
    push_to_hub = False
    hub_model_id = "<your-username>/<my-awesome-model>"
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 1265
    concept= True 
    #model
    image_size = 16
    context_dim = 8 # dimention of centext in the CBM paper 
    skip_concept = False
    num_concepts = 17
    in_channels=16
    out_channels=16
    layers_per_block=2
    block_out_channels=(128, 256,256,512)
    down_block_types=( "DownBlock2D", "AttnDownBlock2D","DownBlock2D","AttnDownBlock2D")
    up_block_types=( "AttnUpBlock2D","UpBlock2D",  "AttnUpBlock2D","UpBlock2D")
    cross_attention_dim = None
    num_attention_heads = 2
    selected_attrs = ["Male", "Young", "Smiling", "Eyeglasses", "Big_Lips", "Big_Nose", "Narrow_Eyes", "Black_Hair", "Blond_Hair", "Bangs", "Bald", "No_Beard","Mustache","Mouth_Slightly_Open", "High_Cheekbones", "Oval_Face", "Chubby"]

#                       0.       1         2             3          4           5             6             7             8           9        10      11          12           13                   14               15             16