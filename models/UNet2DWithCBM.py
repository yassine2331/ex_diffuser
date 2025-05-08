import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


from diffusers import UNet2DModel , DDPMPipeline
from typing import Optional, Tuple, Union, List
from torch import nn 
from diffusers.utils import BaseOutput 
from diffusers.utils.import_utils import is_torch_xla_available
from diffusers.pipelines import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from dataclasses import dataclass 
import torch
import warnings


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False




@dataclass
class UNet2DCBMOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor
    concept: torch.Tensor
    
        



class DDPMPipelineCBM(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)

    
    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,     
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        
        if isinstance(self.unet.config.sample_size, int):
                    image_shape = (
                        batch_size,
                        self.unet.config.in_channels,
                        self.unet.config.sample_size,
                        self.unet.config.sample_size,
                    )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            # 1. predict noise model_output
            output = self.unet(image, t,return_dict=False)
            model_output = output[0]
            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample


            if XLA_AVAILABLE:
                xm.mark_step()
        concept = output[1]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,concept)

        return ImagePipelineOutput(images=image)



class UNet2DWithCBM(UNet2DModel):
    def __init__(self,config, CBM, *args, **kwargs):
        super().__init__(
            sample_size = config.image_size,
            in_channels = config.in_channels,
            out_channels= config.out_channels,
            layers_per_block= config.layers_per_block,
            block_out_channels = config.block_out_channels,
            down_block_types = config.down_block_types,
            up_block_types = config.up_block_types
            )
        self.input_size = int(((config.image_size /( 2**(len(config.block_out_channels)-1)) )**2) * config.block_out_channels[-1])
        
        self.cbm = CBM(self.input_size, config.num_concepts )

        self.linear =  nn.Linear( config.num_concepts, self.input_size )
        self.linear_2 =  nn.Linear( self.input_size, self.input_size )
        
        self.activation_concept = nn.SiLU()
        self.activation_2 = nn.SiLU()
        #self.m = nn.Softmax(dim=1)

        #self.cbm_model = cbm_model  # your concept bottleneck model

    def forward(
        self,
        sample: torch.Tensor,
        
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DCBMOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        pre_concepts = torch.clone(sample)

        pre_concepts = pre_concepts.flatten(start_dim=1,end_dim=-1)


        

        concepts  = self.cbm(pre_concepts)
        
        post_concepts = self.linear(concepts)
        post_concepts = self.activation_concept(post_concepts)
        post_concepts = self.linear_2(post_concepts)
        post_concepts = self.activation_2(post_concepts)
        post_concepts = torch.reshape(post_concepts,sample.shape) 
        
        

        #sample = torch.add(sample,post_concepts)
        sample = post_concepts


        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)
               

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,concepts)

        return UNet2DCBMOutput(sample=sample,concept=concepts)
