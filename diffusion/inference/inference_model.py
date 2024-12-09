# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Inference endpoint for Stable Diffusion."""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

import diffusion.models
from diffusion.models import stable_diffusion_2, stable_diffusion_xl

# Local checkpoint params
LOCAL_CHECKPOINT_PATH = '/tmp/model.pt'


class StableDiffusionInference():
    """Inference endpoint class for Stable Diffusion 2.

    Args:
        model_name (str, optional): Name of the model to load. Default: 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        local_checkpoint_path (str): Path to the local checkpoint. Default: '/tmp/model.pt'.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(self,
                 model_name: str = 'stabilityai/stable-diffusion-2-base',
                 pretrained: bool = False,
                 prediction_type: str = 'epsilon',
                 local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 **kwargs):
        self.device = torch.cuda.current_device()

        model = stable_diffusion_2(
            model_name=model_name,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(local_checkpoint_path)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(self.device)
        self.model = model.eval()

    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = self.model.generate(prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images


class StableDiffusionXLInference():
    """Inference endpoint class for Stable Diffusion XL.

    Args:
        tokenizer_names (str, Tuple[str, ...]): HuggingFace name(s) of the tokenizer(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/tokenizer',
            'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2')``.
        text_encoder_names (str, Tuple[str, ...]): HuggingFace name(s) of the text encoder(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/text_encoder',
            'stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2')``.
        unet_model_name (str): Name of the UNet model to load. Default: 'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to 6.0. Improves stability
            of training.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(self,
                 tokenizer_names: Union[str, Tuple[str,
                                                   ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/tokenizer',
                                                            'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2'),
                 text_encoder_names: Union[str,
                                           Tuple[str,
                                                 ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/text_encoder',
                                                          'stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2'),
                 unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
                 vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
                 clip_qkv: Optional[float] = None,
                 pretrained: bool = False,
                 prediction_type: str = 'epsilon',
                 local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 device:Optional[str] = None,
                 **kwargs):
        #self.device = torch.cuda.current_device()

        model = stable_diffusion_xl(
            tokenizer_names=tokenizer_names,
            text_encoder_names=text_encoder_names,
            unet_model_name=unet_model_name,
            vae_model_name=vae_model_name,
            clip_qkv=clip_qkv,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            device=device,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(local_checkpoint_path)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(device)
        self.model = model.eval()

    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = self.model.generate(prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images


class ModelInference():
    """Generic inference endpoint class for diffusion models with a model.generate() method.

    Args:
        model_name (str): Name of the model from `diffusion.models` to load. Ex: for stable diffusion xl, use 'stable_diffusion_xl'.
        local_checkpoint_path (str): Path to the local checkpoint. Default: '/tmp/model.pt'.
        strict (bool): Whether to load the model weights strictly. Default: False.
        dtype: The data type to use. One of [`float32`, `float16`, `bfloat16`]. Default: `bfloat16`.
        **model_kwargs: Keyword arguments to pass to the model initialization.
    """

    def __init__(self,
                 model_name,
                 local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 strict=False,
                 dtype='bfloat16',
                 **model_kwargs):
        self.device = torch.cuda.current_device()
        model_factory = getattr(diffusion.models, model_name)
        model = model_factory(**model_kwargs)
        dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f'Invalid dtype: {dtype}. Must be one of {list(dtype_map.keys())}')
        self.dtype = dtype_map[dtype]

        if 'pretrained' in model_kwargs and model_kwargs['pretrained']:
            pass
        else:
            state_dict = torch.load(local_checkpoint_path)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=strict)
        model.to(self.device)
        self.model = model.eval()

    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True, dtype=self.dtype):
            imgs = self.model.generate(prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images
