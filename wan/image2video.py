# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.lora_utils import apply_lora


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        low_noise_model=None,
        high_noise_model=None,
        lora_path=None,
        lora_path_high=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            low_noise_model (torch.nn.Module, *optional*, defaults to None):
                Pre-trained low noise model instance.
            high_noise_model (torch.nn.Module, *optional*, defaults to None):
                Pre-trained high noise model instance.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if low_noise_model is None:
            self.low_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.low_noise_checkpoint)
            self.low_noise_model = self._configure_model(
                model=self.low_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
                lora_path=lora_path)
        else:
            self.low_noise_model = low_noise_model

        if high_noise_model is None:
            self.high_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.high_noise_checkpoint)
            self.high_noise_model = self._configure_model(
                model=self.high_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
                lora_path=lora_path_high)
        else:
            self.high_noise_model = high_noise_model

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, lora_path=None):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()
        
        if lora_path is not None:
             apply_lora(model, lora_path)

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 y=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            y (torch.Tensor, *optional*, defaults to None):
                Pre-computed conditional video inputs. If provided, skips the default VAE encoding of `img`.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale

        F = frame_num
        
        if y is None:
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            h, w = img.shape[1:]
            aspect_ratio = h / w
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                self.patch_size[1] * self.patch_size[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                self.patch_size[2] * self.patch_size[2])
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
        else:
            # infer dimensions from y
            # y shape: [C, F_latent, lat_h, lat_w]
            lat_h = y.shape[-2]
            lat_w = y.shape[-1]
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
            # Verify F matches? 
            # latent_timesteps = (F - 1) // stride + 1
            # We trust F is consistent or we rely on y's shape for seq len

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        if y is None:
            msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                               dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w)
                ],
                             dim=1).to(self.device)
            ])[0]
            y = torch.concat([msk, y])
        else:
            y = y.to(self.device)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def multi_generate(self,
                       input_prompt,
                       frame_num=81,
                       width=832,
                       height=480,
                       previous_video=None,
                       start_image=None,
                       end_image=None,
                       motion_amplitude=1.15,
                       motion_frames=5,
                       initial_reference_image=None,
                       **kwargs):
        r"""
        Advanced generation method supporting continuation, inpainting, and motion control,
        ported from ComfyUI-PainterLongVideo.

        Args:
            input_prompt (`str`):
                Text prompt.
            frame_num (`int`):
                Total frames to generate.
            width (`int`):
                Target width (must be divisible by 16).
            height (`int`):
                Target height (must be divisible by 16).
            previous_video (`torch.Tensor`, optional):
                Previous video tensor [C, N, H, W] or [N, H, W, C] (auto-detected).
                Used for continuation if start_image is None.
            start_image (`torch.Tensor`, optional):
                Start frame [C, H, W]. Takes precedence over previous_video.
            end_image (`torch.Tensor`, optional):
                End frame [C, H, W]. Used for loop/target constraint.
            motion_amplitude (`float`):
                Enhance motion in the conditioning latent. >1.0 increases motion.
            motion_frames (`int`):
                Number of frames from previous_video to use for continuity.
            initial_reference_image (`torch.Tensor`, optional):
                Reference image [C, H, W]. (Currently used for visual continuity if needed, 
                logic primarily uses start/end images).
            **kwargs:
                Passed to generate (e.g. shift, sampling_steps, seed, offload_model).

        Returns:
            torch.Tensor: Generated video [C, frame_num, H, W].
        """
        device = self.device
        
        # Helper for resizing
        def _resize(tensor, target_h, target_w):
            # Tensor expected to be [..., C, H, W] or [C, H, W]
            # interpolate expects [N, C, H, W] or [C, H, W] (if 3D) -> wait, interpolate 3D?
            # 2D interpolation: input [N, C, H, W]
            
            orig_dim = tensor.dim()
            if orig_dim == 3: # [C, H, W]
                tensor = tensor.unsqueeze(0)
            
            resized = torch.nn.functional.interpolate(
                tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
            )
            
            if orig_dim == 3:
                resized = resized.squeeze(0)
            return resized

        # Helper to normalize input to [C, ..., H, W] from likely [..., H, W, C] or [..., C, H, W]
        # and ensure [C, F, H, W] for video, [C, H, W] for image
        def _normalize_input(t, is_video=False):
            if t is None: return None
            # Check last dim vs first dim for Channel=3
            if t.shape[-1] == 3: # likely [..., H, W, C]
                if is_video: # [F, H, W, 3] -> [3, F, H, W]
                     t = t.permute(3, 0, 1, 2)
                else: # [H, W, 3] -> [3, H, W] (or [1, H, W, 3] -> [3, 1, H, W])
                     if t.dim() == 4: t = t.permute(0, 3, 1, 2).squeeze(0) # [1,H,W,3]->[3,H,W]
                     else: t = t.permute(2, 0, 1) # [H,W,3]->[3,H,W]
            elif t.shape[1] == 3 and is_video and t.dim()==4: # [F, C, H, W] -> [C, F, H, W]
                 t = t.transpose(0, 1)
            # If already [C, F, H, W] or [C, H, W], keep as is.
            # Assuming C=3 is distinct from F (usually F > 3 or F is small but H,W large)
            return t.to(device)

        prev_vid = _normalize_input(previous_video, is_video=True)
        start_img = _normalize_input(start_image, is_video=False)
        end_img = _normalize_input(end_image, is_video=False)
        init_ref = _normalize_input(initial_reference_image, is_video=False)

        # 1. Logic Switches
        has_prev = prev_vid is not None
        has_start = start_img is not None
        has_end = end_img is not None

        if not has_prev and not has_start and not has_end:
             # Fallback to standard generation if provided img in kwargs, else error
             if 'img' in kwargs: 
                  return self.generate(input_prompt, frame_num=frame_num, **kwargs)
             raise RuntimeError("multi_generate: Provide previous_video, start_image, or end_image")
             
        # 2. Dimensions
        # Wan requires dimensions to be divisible by vae_stride * patch_size
        # Usually 16? vae_stride=(1,8,8), patch=(1,2,2) -> 8*2=16
        # Latents are downscaled by 8.
        
        # 3. Canvas Construction
        # Create base image buffer [C, F, H, W]
        image_seq = torch.full((3, frame_num, height, width), 0.5, device=device, dtype=torch.float32)
        
        # Mask: 1=Cond, 0=Gen. Init to 0. (Wan convention: 1 is kept)
        # Latent dimensions
        lat_h = height // 8
        lat_w = width // 8
        lat_f = (frame_num - 1) // 4 + 1
        
        # Wan mask is [1, LatF, LatH, LatW] but with interleaved repetition for channel 0 used in VAE?
        # Actually generate() constructs `msk` as [1, F, lat_h, lat_w] then transforms it.
        # We should construct `msk` in signal domain [1, F, lat_h, lat_w] first (matching generate's logic pre-transform)
        # or better: construct `msk` in latent domain directly if we know how?
        # `generate` logic:
        # msk = torch.ones(1, F, lat_h, lat_w) ...
        # msk = msk.view(1, msk.shape[1]//4, 4 ...).transpose ...
        # This implies msk is defined on T (frame_num).
        
        msk = torch.zeros(1, frame_num, lat_h, lat_w, device=device) # 0 = Generate
        
        # 4. Fill Content
        if has_start or has_end:
            if has_start:
                # Resize start_img
                s_img = _resize(start_img, height, width) # [3, H, W]
                actual_len = min(frame_num, 1) # Start implies first frame usually, maybe more? 
                # PainterLongVideo uses `start_image[:length]` if it's a batch.
                # Assuming start_image is single frame [3, H, W].
                image_seq[:, 0] = s_img
                msk[:, 0] = 1.0 # Keep frame 0
                # Protect buffer? Painter protects +3 frames. Wan's mask is on latent t?
                # `msk` here is length `frame_num`.
                # If we want to condition frame 0, msk[:, 0] = 1.
                # If we want to condition more, set more.
                # Painter sets mask 0.0 (Condition) for `actual_len + 3`.
                # We set 1.0 (Condition) for `actual_len + 3`?
                # Wan VAE might encode temporal chunks. 
                # Safer to just mask the exact frames we provide.
                # But Wan `generate` sets msk[:, 0:1] = 1, rest 0.
            else:
                 # prev video last frame as start
                 if has_prev:
                     last_frame = prev_vid[:, -1] # [C, H, W]
                     s_img = _resize(last_frame, height, width)
                     image_seq[:, 0] = s_img
                     msk[:, 0] = 1.0
            
            if has_end:
                e_img = _resize(end_img, height, width)
                image_seq[:, -1] = e_img
                msk[:, -1] = 1.0
                # Maybe protect 3 frames equivalent?
                # msk[:, -4:] = 1.0 ?
                # Let's stick to strict 1 frame unless we observe instability.
        else:
             # Only previous video (continuation)
             # Wan logic for standard continuation is usually frame 0 conditioning.
             last_frame = prev_vid[:, -1]
             s_img = _resize(last_frame, height, width)
             image_seq[:, 0] = s_img
             msk[:, 0] = 1.0

        # Motion Amplitude & Latent construction
        # We need to encode `image_seq`.
        # Wan VAE expects [B, C, F, H, W]? 
        # vae.encode argument: `[torch.concat([interpolated_img, zeros], dim=1)]` in generate() is weird.
        # It stacks img (frame 0) and zeros.
        # HERE we have a full `image_seq` which has content at 0 and -1 (if set), and gray in between.
        # We should encode the WHOLE `image_seq`.
        
        # `generate` logic lines:
        # y = self.vae.encode([ torch.concat([img_interpolated_frame0, zeros], dim=1) ])
        # This implies it encodes a tensor where other frames are zero/gray?
        # If we provide full video buffer, we should just encode it.
        
        # Prepare for VAE: [B, C, F, H, W]
        vae_input = image_seq.unsqueeze(0) # [1, 3, F, H, W]
        y = self.vae.encode([vae_input])[0] # [C_out, LatF, LatH, LatW]
        
        # Prepare Mask for concatenation
        # msk was [1, F, lat_h, lat_w]
        # We need to transform it like in `generate`
        # msk[:, 1:] = 0 # (Done by init zeros and setting ones) -> OK.
        
        # Transform msk to VAE format?
        # generate:
        # msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        # msk = msk.view(1, msk.shape[1]//4, 4, lat_h, lat_w).transpose(1, 2)[0]
        # This handles the temporal downsample/stride?
        # WanVAE stride is 4 on temporal? (4n+1)
        # See `frame_num` arg defaults to 81 = 20*4 + 1.
        
        # We need to apply same transform to our custom msk.
        
        # Padding for mask transform?
        # If F=81.
        # msk shape [1, 81, h, w].
        # msk[:, 0:1] repeat 4 -> [1, 4, h, w].
        # msk[:, 1:] -> [1, 80, h, w].    (80 divisible by 4)
        # concat -> [1, 84, h, w].
        # view -> [1, 21, 4, h, w].
        # transpose -> [1, 4, 21, h, w]. -> [4, 21, h, w] (taking [0]).
        # y from vae.encode(F=81) -> Latent F=21? (80/4 + 1 = 21).
        # matches.
        
        msk_t = torch.concat([
             torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk_t = msk_t.view(1, msk_t.shape[1] // 4, 4, lat_h, lat_w)
        msk_t = msk_t.transpose(1, 2)[0]
        
        # Motion Amplitude Implementation
        if motion_amplitude > 1.0:
             # Apply to y (latent)
             # y is [C, LatF, LatH, LatW]
             # Painter: diff = gray - base. 
             # Here y is whole sequence.
             # Base (frame 0) vs others?
             # Wan latents: channel-first.
             # Painter logic:
             # base = latent[:, :, 0:1] (Time 0)
             # gray = latent[:, :, 1:]
             # diff = gray - base
             # ...
             
             base_latent = y[:, 0:1]
             rest_latent = y[:, 1:]
             diff = rest_latent - base_latent
             diff_mean = diff.mean(dim=(0, 2, 3), keepdim=True) # Mean over C, H, W? 
             # Painter: mean(dim=(1,3,4)) -> channels, h, w. (Batch is 0, Time is 2?)
             # Painter layout: [B, C, T, H, W]
             # Wan layout: [C, T, H, W] (No batch dimension here, strictly one video)
             
             # Painter: mean(dim=(1,3,4)) means Channel, H, W. Preserves T.
             # Here C is dim 0, H is 2, W is 3.
             diff_mean = diff.mean(dim=(0, 2, 3), keepdim=True)
             
             diff_centered = diff - diff_mean
             scaled_rest = base_latent + diff_centered * motion_amplitude + diff_mean
             
             # Clamp?
             scaled_rest = torch.clamp(scaled_rest, -6.0, 6.0) # Painter clamps -6,6
             
             y = torch.concat([base_latent, scaled_rest], dim=1)
        
        # Concatenate mask
        y_final = torch.concat([msk_t, y], dim=0)
        
        # Call generate
        # Provide img=None since we provide y.
        # Need pass width/height? generate calculates from img or y.
        # "img" arg is required by generate signature but we can pass a dummy PIL or Tensor if needed,
        # or rely on our refactor that checks if y is None.
        # We need to pass valid tensor to `img` to pass type checks if any, or just None if we updated signature to allow None?
        # I updated signature `generate(..., img, ...` -> `img` is positional arg.
        # I should pass a dummy image to satisfy the argument parser if it's strict, or pass None if I allow it.
        # In my refactor: `if y is None: img = TF.to_tensor(img)...`
        # So if y is NOT None, img can be anything that doesn't crash before that check.
        # But `generate` signature expects `img`.
        # I will pass a dummy tensor.
        
        dummy_img = torch.zeros(3, height, width, device=device)
        
        return self.generate(
            input_prompt,
            dummy_img,
            frame_num=frame_num,
            y=y_final,
            offload_model=kwargs.get('offload_model', True),
            **kwargs
        )
