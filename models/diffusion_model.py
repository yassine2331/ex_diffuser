from diffusers import UNet2DModel

def create_diffusion_model(config):
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128,128,256,512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    )
