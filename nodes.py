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

from .utils import log

script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModel
from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .diffusers_helper.bucket_tools import find_nearest_bucket
from .cascade_node import FramePackCascadeSampler

from .diffusers_helper.lora import create_arch_network_from_weights

from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers

def calcTotalLatentSections(total_second_length, latent_window_size):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    return total_latent_sections

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

class FramePackTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable single block compilation"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable double block compilation"}),
            },
        }
    RETURN_TYPES = ("FRAMEPACKCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks
        }

        return (compile_args, )

#region Model loading
class DownloadAndLoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["lllyasviel/FramePackI2V_HY"],),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa"):

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()

        model_path = os.path.join(folder_paths.models_dir, "diffusers", "lllyasviel", "FramePackI2V_HY")
        if not os.path.exists(model_path):
            print(f"Downloading clip model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_path, torch_dtype=base_dtype, attention_mode=attention_mode).cpu()
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == 'fp8_e4m3fn' or quantization == 'fp8_e4m3fn_fast':
            transformer = transformer.to(torch.float8_e4m3fn)
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
        elif quantization == 'fp8_e5m2':
            transformer = transformer.to(torch.float8_e5m2)
        else:
            transformer = transformer.to(base_dtype)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

            #transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class FramePackLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora": ("BOOLEAN", {"default": True, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
            },
            "optional": {
                "prev_lora":("FPLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            }
        }

    RETURN_TYPES = ("FPLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, prev_lora=None, fuse_lora=True):
        loras_list = []

        lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora,
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)

class LoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "cuda", "tooltip": "Initialize the model on the main device or offload device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
                "lora": ("FPLORA", {"default": None, "tooltip": "LORA model to load"}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa", lora=None, load_device="main_device"):

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if load_device == "main_device":
            transformer_load_device = device
        else:
            transformer_load_device = offload_device

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json
        from safetensors.torch import load_file as load_safetensors
        with open(model_config_path, "r") as f:
            config = json.load(f)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        model_weight_dtype = sd['single_transformer_blocks.0.attn.to_k.weight'].dtype

        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModel(**config, attention_mode=attention_mode)

        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype

        if lora is not None:
            after_lora_dtype = dtype
            dtype = base_dtype

        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(),
                desc=f"Loading transformer parameters to {transformer_load_device}",
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype

            set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

        if lora is not None:
            adapter_list = []
            adapter_weights = []

            for l in lora:
                fuse = True if l["fuse_lora"] else False
                lora_sd = load_torch_file(l["path"])

                if "lora_unet_single_transformer_blocks_0_attn_to_k.lora_up.weight" in lora_sd:
                    from .utils import convert_to_diffusers
                    lora_sd = convert_to_diffusers("lora_unet_", lora_sd)

                if not "transformer.single_transformer_blocks.0.attn_to.k.lora_A.weight" in lora_sd:
                    log.info(f"Converting LoRA weights from {l['path']} to diffusers format...")
                    lora_sd = _convert_hunyuan_video_lora_to_diffusers(lora_sd)

                lora_rank = None
                for key, val in lora_sd.items():
                    if "lora_B" in key or "lora_up" in key:
                        lora_rank = val.shape[1]
                        break
                if lora_rank is not None:
                    log.info(f"Merging rank {lora_rank} LoRA weights from {l['path']} with strength {l['strength']}")
                    adapter_name = l['path'].split("/")[-1].split(".")[0]
                    adapter_weight = l['strength']
                    transformer.load_lora_adapter(lora_sd, weight_name=l['path'].split("/")[-1], lora_rank=lora_rank, adapter_name=adapter_name)

                    adapter_list.append(adapter_name)
                    adapter_weights.append(adapter_weight)

                del lora_sd
                mm.soft_empty_cache()
            if adapter_list:
                transformer.set_adapters(adapter_list, weights=adapter_weights)
                if fuse:
                    if model_weight_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        raise ValueError("Fusing LoRA doesn't work well with fp8 model weights. Please use a bf16 model file, or disable LoRA fusing.")
                    lora_scale = 1
                    transformer.fuse_lora(lora_scale=lora_scale)
                    transformer.delete_adapters(adapter_list)

            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_e5m2":
                params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
                for name, param in transformer.named_parameters():
                    # Make sure to not cast the LoRA weights to fp8.
                    if not any(keyword in name for keyword in params_to_keep) and not 'lora' in name:
                        param.data = param.data.to(after_lora_dtype)

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)


        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

            #transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class FramePackFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "base_resolution": ("INT", {"default": 640, "min": 64, "max": 2048, "step": 16, "tooltip": "Width of the image to encode"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("width","height",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Finds the closes resolution bucket as defined in the orignal code"

    def process(self, image, base_resolution):

        H, W = image.shape[1], image.shape[2]

        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)

        return (new_width, new_height, )


class CreateKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "image_embeds_a": ("CLIP_VISION_OUTPUT",),
                "embed_strength_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "Weighted average constant for image embed interpolation."}),
                "seconds_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1, "tooltip": "timing of the keyframe in seconds"}),
            },
            "optional": {
                "latent_b": ("LATENT",),
                "image_embeds_b": ("CLIP_VISION_OUTPUT",),
                "embed_strength_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "Weighted average constant for image embed interpolation."}),
                "seconds_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1,"tooltip": "timing of the keyframe in seconds"}),
                "latent_c": ("LATENT",),
                "image_embeds_c": ("CLIP_VISION_OUTPUT",),
                "embed_strength_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "Weighted average constant for image embed interpolation."}),
                "seconds_c": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1,"tooltip": "timing of the keyframe in seconds"}),
                "prev_keyframes": ("LATENT", {"default": None}),
                "prev_keyframe_image_embeds": ("LIST", {"default": []}),
                "prev_keyframe_image_embed_strengths": ("LIST", {"default": []}),
                "prev_keyframe_seconds": ("LIST", {"default": []}),

            }
        }
    RETURN_TYPES = ("LATENT", "LIST", "LIST", "LIST")
    RETURN_NAMES = ("keyframes","keyframe_embeds","keyframe_embed_strengths", "keyframe_seconds")
    FUNCTION = "create_keyframes"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Create keyframes latents and section timing."

    def create_keyframes(self,
                         latent_a,image_embeds_a,embed_strength_a, seconds_a,
                         latent_b=None,image_embeds_b=None,embed_strength_b=None, seconds_b=None,
                         latent_c=None,image_embeds_c=None,embed_strength_c=None, seconds_c=None,
                         prev_keyframes=None,prev_keyframe_image_embeds=None,prev_keyframe_image_embed_strengths=None, prev_keyframe_seconds=None):
        tensors = []
        indices = []
        image_embeds=[]
        image_embed_strengths = []
        if prev_keyframes is not None and prev_keyframe_seconds is not None and prev_keyframe_image_embeds is not None and prev_keyframe_image_embed_strengths is not None:
            tensors.append(prev_keyframes["samples"])
            indices += list(prev_keyframe_seconds)
            image_embeds+= list(prev_keyframe_image_embeds)
            image_embed_strengths+= list(prev_keyframe_image_embed_strengths)
        tensors.append(latent_a["samples"])
        indices.append(seconds_a)
        image_embeds.append(image_embeds_a)
        image_embed_strengths.append(embed_strength_a)
        if latent_b is not None and seconds_b is not None and image_embeds_b is not None and embed_strength_b is not None:
            tensors.append(latent_b["samples"])
            indices.append(seconds_b)
            image_embeds.append(image_embeds_b)
            image_embed_strengths.append(embed_strength_b)
        if latent_c is not None and seconds_c is not None and image_embeds_c is not None and embed_strength_c is not None:
            tensors.append(latent_c["samples"])
            indices.append(seconds_c)
            image_embeds.append(image_embeds_c)
            image_embed_strengths.append(embed_strength_c)
        zipped = list(zip(indices, tensors,image_embeds, image_embed_strengths))
        zipped.sort(key=lambda x: x[0])
        sorted_indices = [z[0] for z in zipped]
        sorted_tensors = [z[1] for z in zipped]
        sorted_image_embeds = [z[2] for z in zipped]
        sorted_image_embed_strengths = [z[3] for z in zipped]
        keyframes = torch.cat(sorted_tensors, dim=2) if len(sorted_tensors) > 1 else sorted_tensors[0]
        print(f"keyframes shape: {keyframes.shape}")
        print(f"keyframe_indices: {sorted_indices}")
        return ({"samples": keyframes}, sorted_image_embeds,sorted_image_embed_strengths,sorted_indices,)

class CreatePositiveKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_a": ("CONDITIONING",),
                "index_a": ("INT", {"tooltip": "section index for positive_a"}),
            },
            "optional": {
                "positive_b": ("CONDITIONING",),
                "index_b": ("INT", {"tooltip": "section index for positive_b"}),
                "positive_c": ("CONDITIONING",),
                "index_c": ("INT", {"tooltip": "section index for positive_c"}),
                "prev_keyframes": ("LIST", {"default": []}),
                "prev_keyframe_indices": ("LIST", {"default": []}),
            }
        }
    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("positive_keyframes", "positive_keyframe_indices")
    FUNCTION = "create_positive_keyframes"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Create positive conditioning keyframes and section indices. All CONDITIONING shapes are padded/cropped to match. index_*: section index for each positive. Can be cascaded."

    def create_positive_keyframes(self, positive_a, index_a, positive_b=None, index_b=None, positive_c=None, index_c=None, prev_keyframes=None, prev_keyframe_indices=None):
        keyframes = []
        indices = []
        if prev_keyframes is not None and prev_keyframe_indices is not None:
            keyframes += list(prev_keyframes)
            indices += list(prev_keyframe_indices)
        keyframes.append(positive_a)
        indices.append(index_a)
        if positive_b is not None and index_b is not None:
            keyframes.append(positive_b)
            indices.append(index_b)
        if positive_c is not None and index_c is not None:
            keyframes.append(positive_c)
            indices.append(index_c)
        zipped = list(zip(indices, keyframes))
        zipped.sort(key=lambda x: x[0])
        sorted_indices = [z[0] for z in zipped]
        sorted_keyframes = [z[1] for z in zipped]
        for i, kf in enumerate(sorted_keyframes):
            print(f"[CreatePositiveKeyframes] keyframe {i} shape: {kf[0][0].shape}, device = {kf[0][0].device}, index: {sorted_indices[i]}")
        return sorted_keyframes, sorted_indices

class FramePackSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2video"} ),     
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
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "end_latent": ("LATENT", {"tooltip": "end Latents to use for last frame"} ),
                "end_image_embeds": ("CLIP_VISION_OUTPUT",),
                "start_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "Weighted average constant for image embed interpolation."}),
                "end_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "Weighted average constant for image embed interpolation."}),
                "keyframes": ("LATENT", {"tooltip": "init Lantents to use for image2video keyframes"} ),
                "keyframes_embeds": ("LIST", {"tooltip":"keyframe CLIP_VISION_OUTPUT"}),
                "keyframes_embed_strengths": ("LIST", {"tooltip":"keyframe Weighted average constant for image embed interpolation"}),
                "keyframe_seconds": ("LIST", {"tooltip": "section seconds for each keyframe (e.g. [0.5, 1.2, 5])"}),
				"initial_samples": ("LATENT", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positive_keyframes": ("LIST", {"tooltip": "List of positive CONDITIONING for keyframes"}),
                "positive_keyframe_indices": ("LIST", {"tooltip": "Section indices for each positive_keyframe"}),
                "mix_latent": ("BOOLEAN", {"default": False, "tooltip": "interpolate latents between keyframes"}),
            }
        }
    #end_image_embeds,embed_interpolation,start_embed_strength conflict with keyframes
    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def calcInterpolationList(self, start_latent, start_embed_strength,start_image_encoder_last_hidden_state,
                          end_indices,end_latent,end_embed_strength,end_image_encoder_last_hidden_state,
                          keyframe_indices,keyframes,keyframes_embed_strengths,keyframes_image_encoder_last_hidden_state,
                              mix_latent):
        #from max strength to mix
        orderedStuffs=[]
        orderedStuffs.append((0, start_latent, start_embed_strength, start_image_encoder_last_hidden_state))
        if keyframes is not None:
            for i in range(len(keyframe_indices)):
                orderedStuffs.append((keyframe_indices[i],  keyframes[:, :, i:i+1, :, :], keyframes_embed_strengths[i], keyframes_image_encoder_last_hidden_state[i]))
        orderedStuffs.append((end_indices,end_latent, end_embed_strength,end_image_encoder_last_hidden_state))
        keys=[]
        maxStrength=0
        maxStrengthIndex=0
        for i in range(len(orderedStuffs)):
            stuff=orderedStuffs[i]
            if stuff[2]>maxStrength:
                maxStrength=stuff[2]
                maxStrengthIndex=i
        if maxStrength==0:
            for stuff in orderedStuffs:
                keys.append((stuff[0], stuff[1], stuff[3],stuff[2]))
        else:
            preKey = (orderedStuffs[maxStrengthIndex][0], orderedStuffs[maxStrengthIndex][1], orderedStuffs[maxStrengthIndex][3], orderedStuffs[maxStrengthIndex][2])
            keys.append(preKey)
            #forward
            for i in range(maxStrengthIndex+1,len(orderedStuffs)):
                stuff=orderedStuffs[i]
                nStrength=stuff[2]
                p=nStrength/maxStrength
                hidden_state=None
                if stuff[3] is not None:
                    hidden_state=stuff[3]*(p)+preKey[2]*(1-p)
                latent=stuff[1]
                if mix_latent:
                    latent=latent*(p)+preKey[1]*(1-p)
                nkey=(stuff[0], latent,hidden_state,stuff[2])
                keys.append(nkey)
                preKey=nkey
            #backward
            preKey = (orderedStuffs[maxStrengthIndex][0], orderedStuffs[maxStrengthIndex][1], orderedStuffs[maxStrengthIndex][3])
            for i in range(maxStrengthIndex-1,0,-1):
                stuff=orderedStuffs[i]
                nStrength=stuff[2]
                p=nStrength/maxStrength
                hidden_state = None
                if stuff[3] is not None:
                    hidden_state = stuff[3] * (p) + preKey[2] * (1 - p)
                latent = stuff[1]
                if mix_latent:
                    latent = latent * (p) + preKey[1] * (1 - p)
                nkey=(stuff[0], latent,hidden_state,stuff[2])
                keys.append(nkey)
                preKey=nkey
        sortedkeys=list(sorted(keys, key=lambda x: x[0]))
        if mix_latent:
            #never mix start and end
            sortedkeys[0]=(sortedkeys[0][0],orderedStuffs[0][1],sortedkeys[0][2],sortedkeys[0][3])
            sortedkeys[-1]=(sortedkeys[-1][0],orderedStuffs[-1][1],sortedkeys[-1][2],sortedkeys[-1][3])
        return sortedkeys
    def getInterpolation(self, interpolationList, currentIndex,mix_latent):
        #find the two closest keyframes
        preKey = None
        postKey = None
        preIndex=0
        postIndex=0
        #the start frame
        currentIndex=currentIndex/(interpolationList[-1][0]+1)*(interpolationList[-1][0])
        if currentIndex<interpolationList[0][0]:
            return interpolationList[0][1], interpolationList[0][2],0,0
        if currentIndex>interpolationList[-1][0]:
            return interpolationList[-1][1], interpolationList[-1][2],len(interpolationList)-1,len(interpolationList)-1
        for i in range(len(interpolationList)):
            if interpolationList[i][0] == currentIndex:
                return interpolationList[i][1], interpolationList[i][2], i,i
            if interpolationList[i][0] > currentIndex:
                postKey = interpolationList[i]
                postIndex=i
                break
            preKey = interpolationList[i]
            preIndex=i

        if preKey is None or postKey is None:
            raise ValueError("preKey or postKey is None")
        #interpolate
        p=(postKey[0]-currentIndex)/(postKey[0]-preKey[0])
        index=postIndex*(1-p)+preIndex*(p)
        latentIdx=index
        if mix_latent:
            nkey = postKey[1] * (1-p) + preKey[1] * (p)
        else:
            dp=postKey[3]/(postKey[3]+preKey[3])
            intp=0
            latentIdx=postIndex
            #never use end latent as start
            if p>dp or postIndex==len(interpolationList)-1:
                intp=1
                latentIdx=preIndex
            nkey=postKey[1]*(1-intp)+preKey[1]*(intp)
        if postKey[2] is None or preKey[2] is None:
            nkey2=None
        else:
            nkey2=postKey[2]*(1-p)+preKey[2]*(p)
        return nkey, nkey2,index,latentIdx






    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, total_second_length, teacache_rel_l1_thresh,
                image_embeds, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation, 
                start_latent=None, initial_samples=None,
                end_image_embeds=None, start_embed_strength=1.0, end_embed_strength=1.0,
                keyframes=None, end_latent=None, denoise_strength=1.0,
                keyframe_seconds=None,keyframes_embeds=None,keyframes_embed_strengths=None,
                positive_keyframes=None, positive_keyframe_indices=None,mix_latent=True):

        total_latent_sections =calcTotalLatentSections(total_second_length,latent_window_size )
        section_length = total_second_length / total_latent_sections
        print("total_latent_sections: ", total_latent_sections)
        keyframe_indices=[]
        for keyframe_second in keyframe_seconds:
            keyframe_indices.append(keyframe_second/section_length)


        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if start_latent is not None:
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

        if image_embeds is not None:
            start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(device, base_dtype)
            image_encoder_last_hidden_state=start_image_encoder_last_hidden_state
        keyframes_image_encoder_last_hidden_state = []
        if keyframes_embeds is not None:
            for kf in keyframes_embeds:
                kf = kf["last_hidden_state"].to(device, base_dtype)
                keyframes_image_encoder_last_hidden_state.append(kf)
        has_end_image = end_latent is not None
        if has_end_image:
            assert end_image_embeds is not None
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(device, base_dtype)
        else:
            if image_embeds is not None:
                end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state)

        llama_vec = positive[0][0].to(device, base_dtype)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        clip_l_pooler = positive[0][1]["pooled_output"].to(device, base_dtype)
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
            llama_vec_n = negative[0][0].to(device, base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(device, base_dtype)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)


        # Sampling

        rnd = torch.Generator("cpu").manual_seed(seed)

        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32).cpu()

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

        interpolateList=self.calcInterpolationList(start_latent, start_embed_strength, start_image_encoder_last_hidden_state,
                          total_latent_sections-1, end_latent, end_embed_strength, end_image_encoder_last_hidden_state,
                          keyframe_indices, keyframes, keyframes_embed_strengths, keyframes_image_encoder_last_hidden_state,
                                                   mix_latent)
        for section_no, latent_padding in enumerate(latent_paddings):
            print(f"latent_padding: {latent_padding}")
            print(f"section no: {section_no}")
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            start_latent_frames = T  # 0 or 1
            indices = torch.arange(0, sum([start_latent_frames, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([start_latent_frames, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)


            # --- キーフレーム選択・weightロジック（先頭区間の特別扱いを追加） ---
            total_sections = len(latent_paddings)
            forward_section_no = total_sections - 1 - section_no
            current_keyframe,image_encoder_last_hidden_state,userIndex,latent_userIndex=self.getInterpolation(interpolateList, forward_section_no,mix_latent)
            print(f"Interpolation: {userIndex} latentIndex: {latent_userIndex} interpolateList:{len(interpolateList)} forward_section_no:{forward_section_no} total_sections:{len(latent_paddings)}")
            clean_latents_pre=current_keyframe.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Use end image latent for the first section if provided
            if is_first_section:
                clean_latents_post = interpolateList[-1][1].to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)


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
                    # One can try to remove below trick and just
                    # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
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

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]            

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        return {"samples": real_history_latents / vae_scaling_factor},
    

class TimestampPromptParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": 
                    "A cute girl is standing\n"
                    "[0s-2s: She claps her hands cheerfully]\n"
                    "[2s-: The girl spins around with a smile]", "tooltip": "FramePack timestamp prompt (use 'sec' for seconds, 's' for section index)"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for encoding"}),
            },
            "optional": {
                "total_second_length": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 120.0, "step": 0.1, "tooltip": "動画全体の長さ（秒）。未指定時は12秒 or timestamp最大値"}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The size of the latent window to use for sampling."}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Prompt weight (applied to all prompts)"}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "STRING")
    RETURN_NAMES = ("positive_keyframes", "positive_keyframe_indices", "keyframe_prompts")
    FUNCTION = "parse"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = (
        "Parses FramePack-style timestamp prompts and encodes them with CLIP for each section. "
        "General prompts (not enclosed in brackets) are included in every section. "
        "Timestamp prompts are appended to the relevant sections, and multiple timestamp prompts can overlap. "
        "Section length is 1.2 seconds. If total_second_length is not specified, it defaults to 12 seconds or the maximum timestamp.\n"
        "\n"
        "Timestamp prompt format examples:\n"
        "  [1sec: The person waves hello] [2sec: The person jumps up and down] [4sec: The person does a spin]\n"
        "  [0sec-2sec: The person stands still, looking at the camera] [2sec-4sec: The person raises both arms gracefully above their head] [4sec-6sec: The person does a gentle spin with arms extended] [6sec: The person bows elegantly with a smile]\n"
        "  [-1sec: Applies from the beginning to 1sec] [5sec-: Applies from 5sec to the end]\n"
        "General prompts (not in brackets) are always included in all sections.\n"
        "\n"
        "Supported timestamp prompt formats:\n"
        "  [startsec: description]         e.g. [1sec: ...]\n"
        "  [startsec-endsec: description]  e.g. [0sec-2sec: ...]\n"
        "  [-endsec: description]          e.g. [-1sec: ...] (from start to end)\n"
        "  [startsec-: description]        e.g. [5sec-: ...] (from start to end of video)\n"
        "\n"
        "If you use 's' instead of 'sec', it is interpreted as section index (not seconds)."
    )

    def parse(self, text, clip, total_second_length=12.0,latent_window_size=9.0, weight=1.0):
        # timestamp promptのパース
        # 'sec'（秒）と's'（section index）両対応
        pattern = r'\[(?:(-)?(\d+\.?\d*)(sec|s|\%))?(?:-(?:(\d+\.?\d*)(sec|s|\%))?)?:\s*([^\]]+)\]'
        matches = re.findall(pattern, text)
        timestamp_prompts = []
        max_time = 0.0
        section_length = total_second_length/calcTotalLatentSections(total_second_length,latent_window_size)  # 秒
        for minus, start, start_unit, end, end_unit, desc in matches:
            # 区間をsection indexで管理
            if start_unit == 's' or end_unit == 's':
                # section index指定
                def to_section(val):
                    return int(float(val))
                if minus == '-':
                    section_start = 0
                    section_end = to_section(start)
                elif start and end:
                    section_start = to_section(start)
                    section_end = to_section(end)
                elif start and not end:
                    section_start = to_section(start)
                    section_end = None
                elif end and not start:
                    section_start = 0
                    section_end = to_section(end)
                else:
                    section_start = to_section(start) if start else 0
                    section_end = section_start
            elif start_unit == '%' or end_unit == '%':
                # %指定（sec）→section indexに変換
                def to_sec(val):
                    return float(val)

                if minus == '-':
                    p_start = 0.0
                    p_end = to_sec(start)
                elif start and end:
                    p_start = to_sec(start)
                    p_end = to_sec(end)
                elif start and not end:
                    p_start = to_sec(start)
                    p_end = None
                elif end and not start:
                    p_start = 0.0
                    p_end = to_sec(end)
                else:
                    p_start = to_sec(start) if start else 0.0
                    p_end = p_start

                sec_start=total_second_length*p_start/100
                sec_end=None
                if p_end is not None:
                    sec_end=total_second_length*p_end/100

                section_start = int(sec_start // section_length)
                section_end = int(sec_end // section_length) if sec_end is not None else None
                if sec_end is not None:
                    max_time = max(max_time, sec_end)
                else:
                    max_time = max(max_time, sec_start)
            else:
                # 秒指定（sec）→section indexに変換
                def to_sec(val):
                    return float(val)
                if minus == '-':
                    sec_start = 0.0
                    sec_end = to_sec(start)
                elif start and end:
                    sec_start = to_sec(start)
                    sec_end = to_sec(end)
                elif start and not end:
                    sec_start = to_sec(start)
                    sec_end = None
                elif end and not start:
                    sec_start = 0.0
                    sec_end = to_sec(end)
                else:
                    sec_start = to_sec(start) if start else 0.0
                    sec_end = sec_start
                section_start = int(sec_start // section_length)
                section_end = int(sec_end // section_length) if sec_end is not None else None
                if sec_end is not None:
                    max_time = max(max_time, sec_end)
                else:
                    max_time = max(max_time, sec_start)
            timestamp_prompts.append({
                "section_start": section_start,
                "section_end": section_end,
                "desc": desc.strip()
            })
        # generalプロンプト（時刻指定なし）を抽出
        text_wo_timestamps = re.sub(pattern, '', text)
        general_prompt = text_wo_timestamps.strip() if text_wo_timestamps.strip() else None

        # section数の決定
        if not total_second_length or total_second_length < 1.0:
            total_second_length = max(12.0, max_time)
        else:
            total_second_length = max(total_second_length, max_time)
        num_sections = math.ceil(total_second_length / section_length)

        # 各sectionごとにプロンプトリストを作成
        section_prompts = []
        for section_no in range(num_sections):
            prompts = []
            for tp in timestamp_prompts:
                tp_start = tp["section_start"]
                tp_end = tp["section_end"] if tp["section_end"] is not None else num_sections
                if tp_start <= section_no < tp_end:
                    prompts.append(tp["desc"])
            if general_prompt:
                prompts.append(general_prompt)  # general promptを末尾に追加
            section_prompts.append(" ".join(prompts))

        # プロンプトごとにCLIPエンコードし、同じプロンプトはまとめる（最適化）
        keyframes = []
        indices = []
        keyframe_prompts = []
        last_prompt = None
        for i, prompt in enumerate(section_prompts):
            if prompt != last_prompt:
                tokens = clip.tokenize(prompt)
                cond = clip.encode_from_tokens_scheduled(tokens)
                # weightを適用
                if weight != 1.0:
                    # condはタプルやリストの可能性があるので、最初のテンソルにweightを掛ける
                    if isinstance(cond, (tuple, list)) and hasattr(cond[0][0], 'mul_'):
                        cond[0][0] = cond[0][0] * weight
                keyframes.append(cond)
                indices.append(i)
                keyframe_prompts.append(prompt)
                last_prompt = prompt
        keyframe_prompts_str = ",\n".join(keyframe_prompts)
        return keyframes, indices, keyframe_prompts_str

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFramePackModel": DownloadAndLoadFramePackModel,
    "FramePackSampler": FramePackSampler,
    "CreateKeyframes": CreateKeyframes,
    "CreatePositiveKeyframes": CreatePositiveKeyframes,
    "FramePackTorchCompileSettings": FramePackTorchCompileSettings,
    "FramePackFindNearestBucket": FramePackFindNearestBucket,
    "LoadFramePackModel": LoadFramePackModel,
    "TimestampPromptParser": TimestampPromptParser,
    "FramePackCascadeSampler": FramePackCascadeSampler,
    "FramePackLoraSelect": FramePackLoraSelect,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFramePackModel": "(Down)Load FramePackModel",
    "FramePackSampler": "FramePackSampler",
    "CreateKeyframes": "Create Keyframes",
    "CreatePositiveKeyframes": "Create Positive Keyframes",
    "FramePackTorchCompileSettings": "Torch Compile Settings",
    "FramePackFindNearestBucket": "Find Nearest Bucket",
    "LoadFramePackModel": "Load FramePackModel",
    "TimestampPromptParser": "Timestamp Prompt Parser",
    "FramePackCascadeSampler": "FramePackCascadeSampler",
    "FramePackLoraSelect": "Select Lora",
    }

