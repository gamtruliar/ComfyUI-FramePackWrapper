import os
import torch
import torch.nn.functional as F
import gc
import numpy as np
import math
from tqdm import tqdm
import re

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.cli_args import args, LatentPreviewMethod

script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .diffusers_helper.bucket_tools import find_nearest_bucket

class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = mm.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

class FramePackCascadeSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The size of the latent window to use for sampling."}),
                "total_second_length": ("FLOAT", {"default": 5, "min": 1, "max": 120, "step": 0.1, "tooltip": "The total length of the video in seconds."}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"],
                    {
                        "default": 'unipc_bh1'
                    }),
            },
            "optional": {
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2video"} ),
                "end_latent": ("LATENT", {"tooltip": "end Latents to use for last frame"} ),
                "keyframes": ("LATENT", {"tooltip": "init Lantents to use for image2video keyframes"} ),
                "keyframe_indices": ("LIST", {"tooltip": "section index for each keyframe (e.g. [0, 3, 5])"}),
                "initial_samples": ("LATENT", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positive_keyframes": ("LIST", {"tooltip": "List of positive CONDITIONING for keyframes"}),
                "positive_keyframe_indices": ("LIST", {"tooltip": "Section indices for each positive_keyframe"}),
                "keyframe_weight": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Keyframe multiplier: How much to emphasize the latent at keyframe positions."}),
                "section_start": ("INT", {"default": 0, "min": 0}),
                "section_count": ("INT", {"default": -1, "min": -1, "tooltip": "-1または未指定で全セクションを処理"}),
                "history_latents": ("LATENT", ),
                "total_generated_latent_frames": ("INT", {"default": 0, "min": 0, "tooltip": "total generated frames"}),
            }
        }

    RETURN_TYPES = ("LATENT", 
                    "FramePackMODEL", 
                    "CONDITIONING", 
                    "CONDITIONING", 
                    "CLIP_VISION_OUTPUT", 
                    "LATENT", 
                    "LATENT", 
                    "FLOAT", 
                    "INT", 
                    "INT")
    RETURN_NAMES = ("samples", 
                    "model", 
                    "positive", 
                    "negative", 
                    "image_embeds", 
                    "start_latent", 
                    "history_latents", 
                    "total_second_length", 
                    "next_section_start", 
                    "total_generated_latent_frames")
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, total_second_length, teacache_rel_l1_thresh, image_embeds, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation,
                start_latent=None, initial_samples=None, keyframes=None, end_latent=None, denoise_strength=1.0, keyframe_indices=None,
                positive_keyframes=None, positive_keyframe_indices=None, keyframe_weight=2.0, force_keyframe=False,
                section_start=0, section_count=-1, history_latents=None, total_generated_latent_frames=None):
        

        # process start
        total_latent_sections = total_second_length * 30 / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print("total_latent_sections: ", total_latent_sections)
        force_keyframe = False

        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        original_start_latent = start_latent
        start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        if keyframes is not None:
            keyframes = keyframes["samples"] * vae_scaling_factor
            print(f"keyframes shape: {keyframes.shape}")
        if end_latent is not None:
            end_latent = end_latent["samples"] * vae_scaling_factor
        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        print(f"[FramePackSampler] device: {device}")
        print(f"[FramePackSampler] start_latent device: {start_latent.device}")
        if keyframes is not None:
            print(f"[FramePackSampler] keyframes device: {keyframes.device}")
        if end_latent is not None:
            print(f"[FramePackSampler] end_latent device: {end_latent.device}")
        print(f"[FramePackSampler] positive[0][0] device: {positive[0][0].device}")
        print(f"[FramePackSampler] negative[0][0] device: {negative[0][0].device}")

        image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(base_dtype).to(device)

        llama_vec = positive[0][0].to(base_dtype).to(device)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        clip_l_pooler = positive[0][1]["pooled_output"].to(base_dtype).to(device)
        cached_keyframe_vecs = []
        cached_keyframe_masks = []
        cached_keyframe_poolers = []
        if positive_keyframes is not None:
            for kf in positive_keyframes:
                v = kf[0][0].to(base_dtype).to(device)
                v, m = crop_or_pad_yield_mask(v, length=512)
                p = kf[0][1]["pooled_output"].to(base_dtype).to(device)
                cached_keyframe_vecs.append(v)
                cached_keyframe_masks.append(m)
                cached_keyframe_poolers.append(p)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(base_dtype).to(device)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            

        # Sampling

        rnd = torch.Generator("cpu").manual_seed(seed)
        
        num_frames = latent_window_size * 4 - 3

        if history_latents is None:
            print("[FramePackSampler] initializing new history_latents")
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32).cpu()
        else:
            print("[FramePackSampler] using previous history_latents")
            history_latents = history_latents["samples"]  # 必ず1+2+16フレームを含む
            print(f"[FramePackSampler] previous history shape: {history_latents.shape}")
            print(f"[FramePackSampler] previous history means: {history_latents[:,:,:3,:,:].mean().item():.4f}")
       
        # nodes.py準拠: inで受け取った値を使う
        # 初回は0、2回目以降は前段から受け継ぐ
        if total_generated_latent_frames is None:
            total_generated_latent_frames = 0

        latent_paddings_list = list(reversed(range(total_latent_sections)))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model = HyVideoModel(
                HyVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device,
            )
      
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            latent_paddings_list = latent_paddings.copy()

        print(f"[FramePackSampler] initial section_start: {section_start}, section_count: {section_count}")
        # セクション範囲の制御
        if section_count == -1:
            section_count = total_latent_sections - section_start
        section_end = min(section_start + section_count, total_latent_sections)
        next_section = section_start + section_count
        print(f"[FramePackSampler] calculated section_count: {section_count}, section_end: {section_end}, next_section: {next_section}")
        for section_no in range(section_start, section_end):
            latent_padding = latent_paddings[section_no]
            print(f"latent_padding: {latent_padding}")
            print(f"section no: {section_no}")
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            total_size = sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
            print(f"[FramePackSampler] section {section_no}: indices components: pre=1, padding={latent_padding_size}, window={latent_window_size}, post=1, 2x=2, 4x=16")
            print(f"[FramePackSampler] section {section_no}: total indices size: {total_size}")
            
            indices = torch.arange(0, total_size).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            
            print(f"[FramePackSampler] section {section_no}: latent_indices: {latent_indices.shape}, values={latent_indices.tolist()}")
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # clean_latents_pre を keyframes からセクションごとに取得。なければ start_latent
            total_sections = len(latent_paddings)
            forward_section_no = total_sections - 1 - section_no
            current_keyframe = start_latent.to(history_latents)
            idx_current = 0
            next_idx = None
            if keyframes is not None and keyframes.shape[2] > 0 and keyframe_indices is not None and len(keyframe_indices) > 0:
                if forward_section_no < keyframe_indices[0]:
                    # 先頭より前の区間: start_latent→最初のキーフレーム
                    current_keyframe = start_latent.to(history_latents)
                    idx_current = 0
                    next_idx = keyframe_indices[0]
                elif forward_section_no >= keyframe_indices[-1]:
                    # 最後のキーフレーム以降: 最後のキーフレーム→末尾
                    current_keyframe = keyframes[:, :, -1:, :, :].to(history_latents)
                    idx_current = keyframe_indices[-1]
                    next_idx = total_sections - 1
                else:
                    for i in range(1, len(keyframe_indices)):
                        if keyframe_indices[i-1] <= forward_section_no < keyframe_indices[i]:
                            current_keyframe = keyframes[:, :, i-1:i, :, :].to(history_latents)
                            idx_current = keyframe_indices[i-1]
                            next_idx = keyframe_indices[i]
                            break
                # t計算: 分母が0（同じキーフレームindexが複数回設定など）の場合はt=1.0で回避
                width = next_idx - idx_current if next_idx is not None else 1
                if width == 0:
                    t = 1.0
                else:
                    t = (next_idx - forward_section_no) / width
                weight = 1.0 + (keyframe_weight - 1.0) * t
                clean_latents_pre = current_keyframe * weight
                print(f"[FramePackSampler] forward_section_no={forward_section_no}: use keyframe {idx_current} (next {next_idx}), weight={weight:.2f}, t={t:.2f}")
            else:
                clean_latents_pre = start_latent.to(history_latents)
                print(f"keyframes is None: uses start_latent")
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            print(f"[FramePackSampler] section {section_no}: clean_latents sizes: post={clean_latents_post.shape}, 2x={clean_latents_2x.shape}, 4x={clean_latents_4x.shape}")
            # end_latent対応: 最初のセクションでclean_latents_postをend_latentで差し替え
            if section_no == 0 and end_latent is not None:
                print(f"[FramePackSampler] end_latent is set. Overwriting clean_latents_post. old shape: {clean_latents_post.shape}, new shape: {end_latent.shape}")
                clean_latents_post = end_latent.to(clean_latents_post)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            print(f"[FramePackSampler] section {section_no}: final clean_latents shape: {clean_latents.shape}")

            #vid2vid
            
            if initial_samples is not None:
                total_length = initial_samples.shape[2]
                
                # Get the max padding value for normalization
                max_padding = max(latent_paddings_list)
                
                if is_last_section:
                    # Last section should capture the end of the sequence
                    start_idx = max(0, total_length - latent_window_size)
                else:
                    # Calculate windows that distribute more evenly across the sequence
                    # This normalizes the padding values to create appropriate spacing
                    if max_padding > 0:  # Avoid division by zero
                        progress = (max_padding - latent_padding) / max_padding
                        start_idx = int(progress * max(0, total_length - latent_window_size))
                    else:
                        start_idx = 0
                
                end_idx = min(start_idx + latent_window_size, total_length)
                print(f"start_idx: {start_idx}, end_idx: {end_idx}, total_length: {total_length}")
                input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(device)
          

            # セクションごとのpositiveを選択
            section_positive = positive
            use_keyframe_positive = False
            current_llama_vec = llama_vec
            current_llama_attention_mask = llama_attention_mask
            current_clip_l_pooler = clip_l_pooler
            if positive_keyframes is not None and positive_keyframe_indices is not None and len(positive_keyframes) > 0:
                total_sections = len(latent_paddings)
                forward_section_no = total_sections - 1 - section_no
                kf_idx = None
                for i, idx in enumerate(positive_keyframe_indices):
                    if forward_section_no <= idx:
                        kf_idx = i
                        break
                if kf_idx is not None:
                    section_positive = positive_keyframes[kf_idx]
                    use_keyframe_positive = True
                    current_llama_vec = cached_keyframe_vecs[kf_idx]
                    current_llama_attention_mask = cached_keyframe_masks[kf_idx]
                    current_clip_l_pooler = cached_keyframe_poolers[kf_idx]
                    print(f"[FramePackSampler] section {section_no} (forward {forward_section_no}): use positive_keyframe {kf_idx} (user index {positive_keyframe_indices[kf_idx]})")
                else:
                    # forward_section_no が最後のキーフレームindexより大きい場合は最終キーフレームを使う
                    section_positive = positive_keyframes[-1]
                    use_keyframe_positive = True
                    current_llama_vec = cached_keyframe_vecs[-1]
                    current_llama_attention_mask = cached_keyframe_masks[-1]
                    current_clip_l_pooler = cached_keyframe_poolers[-1]
                    print(f"[FramePackSampler] section {section_no} (forward {forward_section_no}): use last positive_keyframe (user index {positive_keyframe_indices[-1]})")
            print(f"[FramePackSampler] section {section_no}: section_positive[0][0].shape = {section_positive[0][0].shape}")

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            print(f"[FramePackSampler] section {section_no}: starting generation...")
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents if initial_samples is not None else None,
                    strength=denoise_strength,
                    width=W * 8,
                    height=H * 8,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0,
                    shift=shift if shift != 0 else None,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=current_llama_vec,
                    prompt_embeds_mask=current_llama_attention_mask,
                    prompt_poolers=current_clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=base_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
            print(f"[FramePackSampler] section {section_no}: generated shape: {generated_latents.shape}")
            if section_no > 0:
                print(f"[FramePackSampler] section {section_no}: first frame values: {generated_latents[:,:,0,:,:].mean().item():.4f}")

            # キーフレーム強制オプションが有効な場合、current_keyframe（weightなし）で上書き
            # うまく前後がつながらないので蓋してます
            if force_keyframe and (keyframes is not None and keyframe_indices is not None):
                if section_no in keyframe_indices:
                    print(f"[FramePackSampler] section {section_no}: blend first frame with keyframe (no weight, 50%)")
                    generated_latents[:, :, 0:1, :, :] = current_keyframe.to(generated_latents)

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            print(f"[FramePackSampler] section {section_no}: final output shape: {generated_latents.shape}, frames: {generated_latents.shape[2]}")

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]            

            print(f"[FramePackSampler] section {section_no}: history after cat: {history_latents.shape}")
            print(f"[FramePackSampler] section {section_no}: section output shape: {generated_latents.shape}")

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        # 次のセクション番号を計算
        next_section_start = section_start + section_count
        print(f"[FramePackSampler] returning next_section_start: {next_section_start} (current: {section_start}, count: {section_count})")

        return (
            {"samples": real_history_latents / vae_scaling_factor},
            model,  # MODEL
            positive,  # CONDITIONING
            negative,  # CONDITIONING
            image_embeds,  # CLIP_VISION_OUTPUT
            original_start_latent,  # LATENT（オリジナルをそのまま返す）
            {"samples": history_latents},  # LATENT（必要なバッファを含むhistory）
            total_second_length,  # FLOAT
            next_section_start,  # next_section_start (INT)
            total_generated_latent_frames,  # INT（累積生成フレーム数）
        )
