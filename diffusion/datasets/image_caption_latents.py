# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image-Caption Dataset for use with Pre-computed Text Latents."""

import logging
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion.datasets.laion.transforms import (LargestCenterSquare, RandomCropAspectRatioTransform,
                                                 RandomCropBucketedAspectRatioTransform, RandomCropSquare)
from diffusion.datasets.utils import make_streams

log = logging.getLogger(__name__)


class StreamingImageCaptionLatentsDataset(StreamingDataset):
    """Streaming dataset for image-caption datasets with pre-computed text latents.

    Args:
        streams (Sequence[Stream]): One or more Streams to stream/cache samples from.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        microcond_drop_prob (float): The probability of dropping microconditioning. Only relevant for SDXL. Default: ``0.0``.
        crop (Callable, optional): The crop transform to apply to the image before ``transform``. Default: ``None``
        transform (Callable, optional): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_keys (Tuple[str, ...]): Key(s) associated with captions in the streaming dataset. Default: ``('caption',)``.
        caption_selection_probs (Tuple[float, ...]): The probability of selecting each caption key. Default: ``(1.0,)``.
        aspect_ratio_bucket_key (str, optional): Key associated with the aspect ratio bucket in the streaming dataset. Default: ``None``.
        text_latent_keys (Tuple[str, ...]): Key(s) associated with text latents in the streaming dataset.
            Default: ``('T5_LATENTS', 'CLIP_LATENTS')``.
        text_latent_shapes (Tuple[Tuple[int, int], ...]): The shape(s) of the text latents in the streaming dataset.
            Each shape is a 2-tuple where the first dim is the sequence length and the second dim is the feature size.
            Default: ``((512, 4096), (77, 768))``.
        attention_mask_keys (Tuple[str, ...]): Key(s) associated with attention masks in the streaming dataset.
            Default: ``('T5_ATTENTION_MASK', 'CLIP_ATTENTION_MASK')``.
        latent_dtype (torch.dtype): The dtype to cast the text latents to. Default: ``torch.bfloat16``.
        drop_nans (bool): Whether to treat samples with NaN latents as dropped captions. Default: ``True``.
        **streaming_kwargs: Additional arguments to pass in the construction of the StreamingDataloader
    """

    def __init__(
        self,
        streams: Sequence[Stream],
        caption_drop_prob: float = 0.0,
        microcond_drop_prob: float = 0.0,
        crop: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        image_key: str = 'image',
        caption_keys: Tuple[str, ...] = ('caption',),
        caption_selection_probs: Tuple[float, ...] = (1.0,),
        aspect_ratio_bucket_key: Optional[str] = None,
        text_latent_keys: Tuple[str, ...] = ('T5_LATENTS', 'CLIP_LATENTS'),
        text_latent_shapes: Tuple[Tuple[int, int], ...] = ((512, 4096), (77, 768)),
        attention_mask_keys: Tuple[str, ...] = ('T5_ATTENTION_MASK', 'CLIP_ATTENTION_MASK'),
        latent_dtype: torch.dtype = torch.bfloat16,
        drop_nans: bool = True,
        **streaming_kwargs,
    ):

        # Set defaults for vision-friendly streaming args.
        streaming_kwargs.setdefault('shuffle_block_size', 1 << 18)
        streaming_kwargs.setdefault('shuffle_algo', 'py1s')
        super().__init__(streams=streams, **streaming_kwargs)

        self.crop = crop
        self.transform = transform
        self.caption_drop_prob = caption_drop_prob
        self.microcond_drop_prob = microcond_drop_prob
        self.image_key = image_key
        self.caption_keys = caption_keys
        self.caption_selection_probs = caption_selection_probs
        self.aspect_ratio_bucket_key = aspect_ratio_bucket_key
        if isinstance(self.crop, RandomCropBucketedAspectRatioTransform):
            assert self.aspect_ratio_bucket_key is not None, 'aspect_ratio_bucket_key must be provided when using RandomCropBucketedAspectRatioTransform'
        self.text_latent_keys = text_latent_keys
        self.text_latent_shapes = text_latent_shapes
        self.attention_mask_keys = attention_mask_keys
        self.latent_dtype = latent_dtype
        self.drop_nans = drop_nans

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        out = {}

        # Image
        img = sample[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.open(BytesIO(sample[self.image_key]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        out['cond_original_size'] = torch.tensor(img.size)

        # Image transforms
        if isinstance(self.crop, RandomCropBucketedAspectRatioTransform):
            img, crop_top, crop_left = self.crop(img, sample[self.aspect_ratio_bucket_key])
        elif self.crop is not None:
            img, crop_top, crop_left = self.crop(img)
        else:
            crop_top, crop_left = 0, 0
        if self.transform is not None:
            img = self.transform(img)
        out['image'] = img
        out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left])

        # Get the new height and width
        if isinstance(img, torch.Tensor):
            img_h, img_w = img.shape[-2], img.shape[-1]
        elif isinstance(img, Image.Image):
            img_w, img_h = img.size
        else:
            raise ValueError('Image after transformations must either be a PIL Image or Torch Tensor')
        out['cond_target_size'] = torch.tensor([img_w, img_h])

        # Microconditioning dropout as in Stability repo
        # https://github.com/Stability-AI/generative-models/blob/477d8b9a7730d9b2e92b326a770c0420d00308c9/sgm/modules/encoders/modules.py#L151-L160
        if torch.rand(1) < self.microcond_drop_prob:
            out['cond_crops_coords_top_left'] = out['cond_crops_coords_top_left'] * 0
        if torch.rand(1) < self.microcond_drop_prob:
            out['cond_original_size'] = out['cond_original_size'] * 0
        if torch.rand(1) < self.microcond_drop_prob:
            out['cond_target_size'] = out['cond_target_size'] * 0

        # Randomly select a caption according to the selection probabilities
        caption_key = np.random.choice(self.caption_keys, p=self.caption_selection_probs)
        # Load text latents, attention masks, and clip pooled embeddings
        for i in range(len(self.text_latent_keys)):
            latent_key = f'{caption_key}_{self.text_latent_keys[i]}'
            latent_shape = self.text_latent_shapes[i]
            attention_key = f'{caption_key}_{self.attention_mask_keys[i]}'

            if torch.rand(1) < self.caption_drop_prob:
                out[self.text_latent_keys[i]] = torch.zeros(latent_shape, dtype=self.latent_dtype)
                out[self.attention_mask_keys[i]] = torch.zeros(latent_shape[0])
                if 'CLIP_LATENTS' in latent_key:
                    out['CLIP_POOLED'] = torch.zeros(latent_shape[1])
            else:
                text_latent = np.frombuffer(sample[latent_key], dtype=np.float32).copy()
                out[self.text_latent_keys[i]] = torch.from_numpy(text_latent).to(
                    self.latent_dtype).reshape(latent_shape)
                attention_mask = np.frombuffer(sample[attention_key], dtype=np.bool_).copy()
                out[self.attention_mask_keys[i]] = torch.from_numpy(attention_mask).to(dtype=torch.float).reshape(-1)  #.reshape(latent_shape[0])
                if 'CLIP_LATENTS' in latent_key:
                    clip_pooled = np.frombuffer(sample[f'{caption_key}_CLIP_POOLED_TEXT'], dtype=np.float32).copy()
                    out['CLIP_POOLED'] = torch.from_numpy(clip_pooled).to(self.latent_dtype).reshape(latent_shape[1])
        if self.drop_nans:
            for latent_key, attn_key in zip(self.text_latent_keys, self.attention_mask_keys):
                if out[latent_key].isnan().any():
                    out[latent_key] = torch.zeros_like(out[latent_key])
                    out[attn_key] = torch.zeros_like(out[attn_key])
                if 'CLIP_LATENTS' in latent_key and out['CLIP_POOLED'].isnan().any():
                    out['CLIP_POOLED'] = torch.zeros_like(out['CLIP_POOLED'])
        return out


def build_streaming_image_caption_latents_dataloader(
    remote: Union[str, List],
    batch_size: int,
    local: Optional[Union[str, List]] = None,
    caption_drop_prob: float = 0.0,
    microcond_drop_prob: float = 0.0,
    resize_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]] = 256,
    ar_bucket_boundaries: Optional[Tuple[float, ...]] = None,
    transform: Optional[List[Callable]] = None,
    crop_type: Optional[str] = 'square',
    image_key: str = 'image',
    caption_keys: Tuple[str, ...] = ('caption',),
    caption_selection_probs: Tuple[float, ...] = (1.0,),
    text_latent_keys: Tuple[str, ...] = ('T5_LATENTS', 'CLIP_LATENTS'),
    text_latent_shapes: Tuple[Tuple, ...] = ((512, 4096), (77, 768)),
    attention_mask_keys: Tuple[str, ...] = ('T5_ATTENTION_MASK', 'CLIP_ATTENTION_MASK'),
    latent_dtype: str = 'torch.bfloat16',
    aspect_ratio_bucket_key: Optional[str] = None,
    proportion: Optional[list] = None,
    repeat: Optional[list] = None,
    choose: Optional[list] = None,
    streaming_kwargs: Optional[Dict] = None,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a streaming dataloader for image-caption pairs with pre-computed text latents.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        batch_size (int): The batch size to use for both the ``StreamingDataset`` and ``DataLoader``.
        local (str, Sequence[str], optional): One or more local filesystem directories where dataset is cached during operation.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        microcond_drop_prob (float): The probability of dropping microconditioning. Default:``0.0``.
        resize_size (int, Tuple[int, int], Tuple[Tuple[int, int], ...]): The size to resize the image to. Specify a
            tuple of tuples if using 'aspect_ratio' crop_type. Default: ``256``.
        ar_bucket_boundaries (Tuple[float, ...], optional): When using ``crop_type='aspect_ratio'``, specifies the
            boundary points for bucket assignment. This tuple should be of length len(resize_size) - 1. If set to
            ``None``, the bucket with the smallest distance to the current sample's aspect ratio is selected.
            Default: ``None``.
        transform (Callable, optional): The transforms to apply to the image. Default: ``None``.
        crop_type (str, optional): Type of crop to perform, either ['square', 'random', 'aspect_ratio', 'bucketed_aspect_ratio'].
            Default: ``'square'``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_keys (Tuple[str, ...]): Key(s) associated with captions in the streaming dataset. Default: ``('caption',)``.
        caption_selection_probs (Tuple[float, ...]): The probability of selecting each caption key. Default: ``(1.0,)``.
        aspect_ratio_bucket_key (str, optional): Key associated with the aspect ratio bucket in the streaming dataset. Default: ``None``.
        text_latent_keys (Tuple[str, ...]): Key(s) associated with text latents in the streaming dataset.
            Default: ``('T5_LATENTS', 'CLIP_LATENTS')``.
        text_latent_shapes (Tuple[Tuple[int, int], ...]): The shape(s) of the text latents in the streaming dataset.
            Each shape is a 2-tuple where the first dim is the sequence length and the second dim is the feature size.
            Default: ``((512, 4096), (77, 768))``.
        attention_mask_keys (Tuple[str, ...]): Key(s) associated with attention masks in the streaming dataset.
            Default: ``('T5_ATTENTION_MASK', 'CLIP_ATTENTION_MASK')``.
        latent_dtype (str): The torch dtype to cast the text latents to. One of 'torch.float16', 'torch.float32',
            or 'torch.bfloat16'. Default: ``'torch.bfloat16'``.
        aspect_ratio_bucket_key (str, optional): Key associated with the aspect ratio bucket in the streaming dataset.
            Needed if using ``crop_type='bucketed_aspect_ratio'``. Default: ``None``.
        proportion (list, optional): Specifies how to sample this Stream relative to other Streams. Default: ``None``.
        repeat (list, optional): Specifies the degree to which a Stream is upsampled or downsampled. Default: ``None``.
        choose (list, optional): Specifies the number of samples to choose from a Stream. Default: ``None``.
        streaming_kwargs (dict, optional): Additional arguments to pass to the ``StreamingDataset``. Default: ``None``.
        dataloader_kwargs (dict, optional): Additional arguments to pass to the ``DataLoader``. Default: ``None``.
    """
    # Check crop type
    if crop_type is not None:
        crop_type = crop_type.lower()
        if crop_type not in ['square', 'random', 'aspect_ratio', 'bucketed_aspect_ratio']:
            raise ValueError(
                f'Invalid crop_type: {crop_type}. Must be ["square", "random", "aspect_ratio", "bucketed_aspect_ratio", None]'
            )
        if crop_type in ['aspect_ratio', 'bucketed_aspect_ratio'] and (isinstance(resize_size, int) or
                                                                       isinstance(resize_size[0], int)):
            raise ValueError(
                'If using aspect ratio bucketing, specify aspect ratio buckets in resize_size as a tuple of tuples.')
    # Check latent dtype
    dtypes = {'torch.float16': torch.float16, 'torch.float32': torch.float32, 'torch.bfloat16': torch.bfloat16}
    assert latent_dtype in dtypes, f'Invalid latent_dtype: {latent_dtype}. Must be one of {list(dtypes.keys())}'
    dtype = dtypes[latent_dtype]

    # Handle ``None`` kwargs
    if streaming_kwargs is None:
        streaming_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # Make streams
    streams = make_streams(remote, local=local, proportion=proportion, repeat=repeat, choose=choose)

    # Set the crop to apply
    if crop_type == 'square':
        crop = LargestCenterSquare(resize_size)
    elif crop_type == 'random':
        crop = RandomCropSquare(resize_size)
    elif crop_type == 'aspect_ratio':
        crop = RandomCropAspectRatioTransform(resize_size, ar_bucket_boundaries)  # type: ignore
    elif crop_type == 'bucketed_aspect_ratio':
        assert aspect_ratio_bucket_key is not None, 'aspect_ratio_bucket_key must be provided when using bucketed_aspect_ratio crop type'
        crop = RandomCropBucketedAspectRatioTransform(resize_size)  # type: ignore
    else:
        crop = None

    if transform is None:
        transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform)
    assert isinstance(transform, Callable)

    dataset = StreamingImageCaptionLatentsDataset(
        streams=streams,
        caption_drop_prob=caption_drop_prob,
        microcond_drop_prob=microcond_drop_prob,
        crop=crop,
        transform=transform,
        image_key=image_key,
        caption_keys=caption_keys,
        caption_selection_probs=caption_selection_probs,
        aspect_ratio_bucket_key=aspect_ratio_bucket_key,
        text_latent_keys=text_latent_keys,
        text_latent_shapes=text_latent_shapes,
        attention_mask_keys=attention_mask_keys,
        latent_dtype=dtype,
        batch_size=batch_size,
        **streaming_kwargs,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )

    return dataloader
