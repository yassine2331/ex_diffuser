from dataclasses import dataclass

@dataclass
class TrainingConfig:
    root_dir = ".."
    image_size = 32
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 1
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
