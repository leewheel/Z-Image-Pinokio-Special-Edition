import os
import sys
import gc
import uuid
import random
import re
import datetime
import json
import tempfile
import locale

# =========================
# 路径配置和文件夹创建
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_DIR = os.path.join(BASE_DIR, "MOD")
MOD_TRANSFORMER = os.path.join(MOD_DIR, "transformer")
MOD_VAE = os.path.join(MOD_DIR, "vae")
LORA_ROOT = os.path.join(BASE_DIR, "lora")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for p in [MOD_TRANSFORMER, MOD_VAE, LORA_ROOT, OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

# =========================
# 语言检测
# =========================
try:
    system_lang = locale.getdefaultlocale()[0]
    is_chinese = system_lang and system_lang.startswith('zh')
except:
    is_chinese = False

def get_message(key, *args):
    messages = {
        "peft_loaded": ("✅ PEFT 库已加载，LoRA 功能可用。", "✅ PEFT library loaded, LoRA functionality available."),
        "peft_not_detected": ("⚠️ 警告: 未检测到 PEFT 库。LoRA 功能将禁用。", "⚠️ Warning: PEFT library not detected. LoRA functionality will be disabled."),
        "lora_skipped": ("⚠️ [LoRA] 已跳过加载：PEFT 库未安装。", "⚠️ [LoRA] Skipped loading: PEFT library not installed."),
        "transformer_not_loaded": ("⚠️ Transformer 未加载，无法应用 LoRA", "⚠️ Transformer not loaded, cannot apply LoRA"),
        "lora_file_not_exist": ("⚠️ LoRA 文件不存在: {}", "⚠️ LoRA file does not exist: {}"),
        "lora_loading": ("  [LoRA] 正在加载: {} (权重: {} * {} = {:.2f})", "  [LoRA] Loading: {} (weight: {} * {} = {:.2f})"),
        "lora_loaded": ("✅ LoRA 加载成功: {}", "✅ LoRA loaded successfully: {}"),
        "lora_failed": ("❌ LoRA 加载严重失败: {}", "❌ LoRA loading failed critically: {}"),
        "applying_vae": ("正在应用自定义 VAE: {}", "Applying custom VAE: {}"),
        "vae_loaded": ("✅ 自定义 VAE 加载成功", "✅ Custom VAE loaded successfully"),
        "vae_failed": ("⚠️ 自定义 VAE 加载失败: {}", "⚠️ Custom VAE loading failed: {}"),
        "forcing_to_ram": ("  [System] 正在强制将模型搬运至 RAM (请稍候)...", "  [System] Forcing model to RAM (please wait)..."),
        "model_to_ram": ("  [System] 模型已加载至 RAM。", "  [System] Model loaded to RAM."),
        "t2i_low_vram": ("  [T2I] 已启用低显存优化模式", "  [T2I] Low VRAM optimization mode enabled"),
        "t2i_high_end": ("  [T2I] 已启用高端机模式", "  [T2I] High-end GPU mode enabled"),
        "t2i_pipeline_loaded": ("✅ 文生图 Pipeline 加载完成", "✅ Text-to-Image Pipeline loaded"),
        "i2i_pipeline_failed": ("加载图生图 Pipeline 失败：{}", "Failed to load Image-to-Image Pipeline: {}"),
        "i2i_pipeline_loaded": ("✅ 图生图 Pipeline 加载完成", "✅ Image-to-Image Pipeline loaded"),
        "i2i_low_vram": ("  [I2I] 已启用低显存优化模式", "  [I2I] Low VRAM optimization mode enabled"),
        "i2i_high_end": ("  [I2I] 已启用高端机模式", "  [I2I] High-end GPU mode enabled"),
        "generation_stopped": ("🛑 生成已被用户手动停止", "🛑 Generation stopped by user"),
        "upload_image_first": ("⚠️ 请先上传图片！", "⚠️ Please upload an image first!"),
        "i2i_model_failed": ("加载图生图模型失败: {}", "Failed to load Image-to-Image model: {}"),
        "native_inpaint_failed": ("⚠️ 原生 Inpaint 失败 ({})，使用手动混合模式...", "⚠️ Native Inpaint failed ({}), using manual blending mode..."),
        "paint_area": ("⚠️ 请使用画笔在图片上涂抹要修改的区域。", "⚠️ Please use the brush to paint the area to modify on the image."),
        "mask_invalid": ("⚠️ Mask 无效，请确保涂抹了区域。", "⚠️ Mask invalid, please ensure an area is painted."),
        "model_load_failed": ("模型加载失败: {}", "Model loading failed: {}"),
        "inpainting_failed": ("局部重绘失败: {}", "Inpainting failed: {}"),
        "generating": ("生成中", "Generating"),
        "img2img_processing": ("图生图中", "Img2Img processing"),
    }
    zh, en = messages[key]
    return (zh if is_chinese else en).format(*args)

# 环境配置
os.environ.pop("PYTHONHOME", None)
os.environ.pop("PYTHONPATH", None)
os.environ["DIFFUSERS_USE_PEFT_BACKEND"] = "true"
os.environ["PEFT_DEBUG"] = "false"

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw

import gradio as gr
from diffusers import (
    ZImagePipeline, 
    ZImageImg2ImgPipeline,
    AutoencoderKL, 
    ZImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# =========================
# 检测 PEFT 环境
# =========================
PEFT_AVAILABLE = False
try:
    import peft
    from diffusers.utils import is_peft_available
    if is_peft_available():
        PEFT_AVAILABLE = True
        print(get_message("peft_loaded"))
    else:
        raise ImportError
except ImportError:
    print(get_message("peft_not_detected"))

# =========================
# 双语文本字典
# =========================
TEXT = {
    "zh": {
        "title": "# 🎨 Z-Image-Turbo Low Vram Edition",
        "lang_btn": "EN",
        "tab_generate": "图像生成", "tab_edit": "图片编辑", "tab_img2img": "图生图 (增强版)", "tab_inpaint": "局部重绘",
        "prompt": "Prompt", "prompt_placeholder": "输入你的描述...", "negative_prompt": "负面提示词", "negative_placeholder": "low quality, blurry, bad anatomy",
        "refresh_lora": "🔄 刷新 LoRA", "refresh_model": "🔄 刷新模型", "lora_label": "LoRA", "lora_strength": "LoRA 强度", "lora_weight": "权重",
        "model_section": "### 模型选择/Model Selection", "transformer": "Transformer", "vae": "VAE", "vram_type": "显存类型",
        "vram_low": "24GB以下 (优化模式)", "vram_high": "高端机模式 (>=24GB)", "device": "设备", "num_images": "生成张数",
        "output_format": "输出格式", "width": "宽度", "height": "高度", "steps": "步数", "cfg": "CFG", "seed": "种子", "random_seed": "随机种子",
        "generate": "🚀 生成", "stop": "🛑 停止生成", "gallery": "生成结果", "used_seed": "使用种子",
        "edit_upload": "上传图片", "rotate": "旋转角度 (度)", "crop_x": "裁剪 X (%)", "crop_y": "裁剪 Y (%)", "crop_w": "裁剪宽度 (%)", "crop_h": "裁剪高度 (%)",
        "hflip": "水平翻转", "vflip": "垂直翻转", "edit_btn": "开始编辑", "edited_image": "编辑后的图片",
        "filter": "应用滤镜", "brightness": "亮度调整 (%)", "contrast": "对比度调整 (%)", "saturation": "饱和度调整 (%)",
        "i2i_ref": "上传参考图", "i2i_prompt": "修改提示词", "i2i_ph": "描述你希望图中发生的变化...", "i2i_mode": "Img2Img 模式",
        "i2i_mode_a": "A. 严格保结构（微调风格）", "i2i_mode_b": "B. 强烈听 prompt（允许大改）", "i2i_out_w": "输出宽 (0=自动)", "i2i_out_h": "输出高 (0=自动)",
        "i2i_tip": "**提示：** 宽高都为0时自动保持上传图比例并接近1024。", "i2i_strength": "重绘强度", "i2i_btn": "🎨 开始修改", "i2i_note": "注：使用官方 Z-Image Img2Img 引擎。",
        "inpaint_editor": "绘制 Mask (白色为修改区，黑色为保留区)", "inpaint_tip": "提示：先上传图片，然后用画笔涂抹要修改的区域。", "inpaint_upload": "上传原图并绘制", "inpaint_desc": "📖 使用指南：涂抹区域（白色/彩色）将被重新生成，未涂抹区域保持原样。",
    },
    "en": {
        "title": "# 🎨 Z-Image-Turbo Low Vram Edition", "lang_btn": "中文",
        "tab_generate": "Image Generation", "tab_edit": "Image Editing", "tab_img2img": "Img2Img (Enhanced)", "tab_inpaint": "Inpainting",
        "prompt": "Prompt", "prompt_placeholder": "Enter your description...", "negative_prompt": "Negative Prompt", "negative_placeholder": "low quality, blurry",
        "refresh_lora": "🔄 Refresh LoRA", "refresh_model": "🔄 Refresh Models", "lora_label": "LoRA", "lora_strength": "LoRA Strength", "lora_weight": "Weight",
        "model_section": "### Model Selection", "transformer": "Transformer", "vae": "VAE", "vram_type": "VRAM Type",
        "vram_low": "Under 24GB (Optimized)", "vram_high": "High-End GPU Mode (>=24GB)", "device": "Device", "num_images": "Number of Images",
        "output_format": "Output Format", "width": "Width", "height": "Height", "steps": "Steps", "cfg": "CFG", "seed": "Seed", "random_seed": "Random Seed",
        "generate": "🚀 Generate", "stop": "🛑 Stop Generation", "gallery": "Generated Images", "used_seed": "Used Seed",
        "edit_upload": "Upload Image", "rotate": "Rotation (degrees)", "crop_x": "Crop X (%)", "crop_y": "Crop Y (%)", "crop_w": "Crop Width (%)", "crop_h": "Crop Height (%)",
        "hflip": "Horizontal Flip", "vflip": "Vertical Flip", "edit_btn": "Apply Edit", "edited_image": "Edited Image",
        "filter": "Apply Filter", "brightness": "Brightness (%)", "contrast": "Contrast (%)", "saturation": "Saturation (%)",
        "i2i_ref": "Upload Reference", "i2i_prompt": "Modification Prompt", "i2i_ph": "Describe changes...", "i2i_mode": "Img2Img Mode",
        "i2i_mode_a": "A. Strict Structure (Style tweak)", "i2i_mode_b": "B. Strong Prompt (Allow changes)", "i2i_out_w": "Output Width (0=Auto)", "i2i_out_h": "Output Height (0=Auto)",
        "i2i_tip": "**Tip:** Auto ratio if both 0.", "i2i_strength": "Denoising Strength", "i2i_btn": "🎨 Start Modification", "i2i_note": "Using official Z-Image Img2Img engine.",
        "inpaint_editor": "Draw Mask (White=Modify, Black=Keep)", "inpaint_tip": "Tip: Upload image, then paint area to modify.", "inpaint_upload": "Upload & Paint", "inpaint_desc": "📖 Guide: Painted areas (white/color) will be regenerated. Unpainted areas stay original.",
    }
}

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SNAPSHOT_DIR = os.path.join(BASE_DIR, "cache", "HF_HOME", "hub", "models--Tongyi-MAI--Z-Image-Turbo", "snapshots", "5f4b9cbb80cc95ba44fe6667dfd75710f7db2947")
if not os.path.exists(BASE_SNAPSHOT_DIR):
    BASE_SNAPSHOT_DIR = os.path.join(BASE_DIR, "ckpts", "Z-Image-Turbo")
    if not os.path.exists(BASE_SNAPSHOT_DIR):
        BASE_SNAPSHOT_DIR = "."

TRANSFORMER_ROOT = os.path.join(BASE_SNAPSHOT_DIR, "transformer")
TEXT_ENCODER_ROOT = os.path.join(BASE_SNAPSHOT_DIR, "text_encoder")
VAE_ROOT = os.path.join(BASE_SNAPSHOT_DIR, "vae")

pipe_t2i = None
pipe_i2i = None
current_model_config = {"transformer": "default", "vae": "default", "is_low_vram": True}
is_generating_interrupted = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def auto_flush_vram():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# =========================
# 核心优化：LoRA 加载逻辑
# =========================
def apply_lora_to_pipeline(pipe_local, lora_choice, lora_alpha, lora_scale=1.0):
    if not PEFT_AVAILABLE:
        print(get_message("lora_skipped"))
        return pipe_local

    if pipe_local is None:
        return pipe_local
    if pipe_local.transformer is None:
        print(get_message("transformer_not_loaded"))
        return pipe_local

    if hasattr(pipe_local, "unload_lora_weights"):
        try:
            pipe_local.unload_lora_weights()
        except Exception:
            pass

    if not lora_choice or lora_choice.lower() == "none":
        return pipe_local

    lora_path = os.path.join(LORA_ROOT, lora_choice)
    if not os.path.exists(lora_path):
        print(get_message("lora_file_not_exist", lora_path))
        return pipe_local

    try:
        raw_alpha = float(lora_alpha)
        effective_alpha = raw_alpha * lora_scale

        if effective_alpha <= 0:
            return pipe_local

        adapter_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(lora_choice)[0])

        print(get_message("lora_loading", lora_choice, raw_alpha, lora_scale, effective_alpha))
        pipe_local.load_lora_weights(
            LORA_ROOT,
            weight_name=lora_choice,
            adapter_name=adapter_name
        )
        pipe_local.set_adapters([adapter_name], adapter_weights=[effective_alpha])
        print(get_message("lora_loaded", adapter_name))

    except Exception as e:
        import traceback
        print(get_message("lora_failed", e))

    return pipe_local

def scan_lora_items():
    if not os.path.isdir(LORA_ROOT):
        return []
    return sorted([f for f in os.listdir(LORA_ROOT) if f.lower().endswith((".safetensors", ".pt", ".pth"))])

def update_prompt_with_lora(prompt, lora_choice, lora_alpha):
    prompt = (prompt or "").strip()
    prompt_clean = re.sub(r"<lora:[^>]+>", "", prompt).strip()
    if lora_choice and lora_choice.lower() != "none":
        try:
            alpha = float(lora_alpha)
        except: alpha = 1.0
        if alpha > 0:
            name = os.path.splitext(lora_choice)[0]
            alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
            return f"{prompt_clean} <lora:{name}:{alpha_str}>"
    return prompt_clean

# =========================
# 模型加载逻辑
# =========================
def load_t2i_pipeline(transformer_choice, vae_choice, is_low_vram):
    global pipe_t2i, current_model_config
    config_key = ("t2i", transformer_choice, vae_choice, is_low_vram)
    if pipe_t2i is not None and current_model_config.get("t2i") == config_key:
        return pipe_t2i

    auto_flush_vram()
    pipe_t2i = None
    
    transformer = ZImageTransformer2DModel.from_pretrained(TRANSFORMER_ROOT, torch_dtype=DTYPE, local_files_only=True)
    if transformer_choice != "default":
        t_path = resolve_model_path(transformer_choice, MOD_TRANSFORMER)
        if t_path:
            if os.path.isdir(t_path):
                custom_t = ZImageTransformer2DModel.from_pretrained(t_path, torch_dtype=DTYPE, local_files_only=True)
                transformer = custom_t
            else:
                state = load_file(t_path, device="cpu")
                processed = {}
                prefix = "model.diffusion_model."
                for k, v in state.items():
                    new_k = k[len(prefix):] if k.startswith(prefix) else k
                    processed[new_k] = v.to(DTYPE)
                transformer.load_state_dict(processed, strict=False)
                del state, processed

    text_encoder = AutoModelForCausalLM.from_pretrained(TEXT_ENCODER_ROOT, torch_dtype=DTYPE, local_files_only=True)
    
    pipe_t2i = ZImagePipeline.from_pretrained(
        BASE_SNAPSHOT_DIR,
        local_files_only=True,
        transformer=transformer,
        text_encoder=text_encoder,
    )
    pipe_t2i.to(dtype=DTYPE)

    if vae_choice != "default":
        v_path = resolve_model_path(vae_choice, MOD_VAE)
        if v_path:
            print(get_message("applying_vae", vae_choice))
            vae_device_map = {"": "cpu"} if is_low_vram else None
            try:
                if os.path.isfile(v_path):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        config_file_path = os.path.join(tmpdir, "config.json")
                        vae_config_dict = dict(pipe_t2i.vae.config)
                        with open(config_file_path, "w", encoding="utf-8") as f:
                            json.dump(vae_config_dict, f, indent=2)
                        try:
                            pipe_t2i.vae = AutoencoderKL.from_single_file(v_path, dtype=DTYPE, config=tmpdir, device_map=vae_device_map)
                        except TypeError:
                            pipe_t2i.vae = AutoencoderKL.from_single_file(v_path, torch_dtype=DTYPE, config=tmpdir, device_map=vae_device_map)
                        print(get_message("vae_loaded"))
                else:
                    pipe_t2i.vae = AutoencoderKL.from_pretrained(v_path, torch_dtype=DTYPE, device_map=vae_device_map)
            except Exception as e:
                print(get_message("vae_failed", e))

    if DEVICE == "cuda":
        if is_low_vram:
            print(get_message("forcing_to_ram"))
            pipe_t2i.to("cpu")
            print(get_message("model_to_ram"))
            pipe_t2i.enable_sequential_cpu_offload()
            print(get_message("t2i_low_vram"))
        else:
            pipe_t2i.to("cuda")
            print(get_message("t2i_high_end"))

    current_model_config["t2i"] = config_key
    print("✅ 文生图 Pipeline 加载完成")
    return pipe_t2i

def load_i2i_pipeline(transformer_choice, vae_choice, is_low_vram):
    global pipe_i2i, current_model_config
    config_key = ("i2i", transformer_choice, vae_choice, is_low_vram)
    if pipe_i2i is not None and current_model_config.get("i2i") == config_key:
        return pipe_i2i

    auto_flush_vram()
    pipe_i2i = None
    
    transformer = ZImageTransformer2DModel.from_pretrained(TRANSFORMER_ROOT, torch_dtype=DTYPE, local_files_only=True)
    if transformer_choice != "default":
        t_path = resolve_model_path(transformer_choice, MOD_TRANSFORMER)
        if t_path:
            if os.path.isdir(t_path):
                custom_t = ZImageTransformer2DModel.from_pretrained(t_path, torch_dtype=DTYPE, local_files_only=True)
                transformer = custom_t
            else:
                state = load_file(t_path, device="cpu")
                processed = {}
                prefix = "model.diffusion_model."
                for k, v in state.items():
                    new_k = k[len(prefix):] if k.startswith(prefix) else k
                    processed[new_k] = v.to(DTYPE)
                transformer.load_state_dict(processed, strict=False)
                del state, processed

    try:
        pipe_i2i = ZImageImg2ImgPipeline.from_pretrained(
            BASE_SNAPSHOT_DIR,
            local_files_only=True,
            transformer=transformer,
        )
    except Exception as e:
        raise gr.Error(f"加载图生图 Pipeline 失败：{str(e)}")
        
    pipe_i2i.to(dtype=DTYPE)

    if vae_choice != "default":
        v_path = resolve_model_path(vae_choice, MOD_VAE)
        if v_path:
            print(get_message("applying_vae", vae_choice))
            vae_device_map = {"": "cpu"} if is_low_vram else None
            try:
                if os.path.isfile(v_path):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        config_file_path = os.path.join(tmpdir, "config.json")
                        vae_config_dict = dict(pipe_i2i.vae.config)
                        with open(config_file_path, "w", encoding="utf-8") as f:
                            json.dump(vae_config_dict, f, indent=2)
                        try:
                            pipe_i2i.vae = AutoencoderKL.from_single_file(v_path, dtype=DTYPE, config=tmpdir, device_map=vae_device_map)
                        except TypeError:
                            pipe_i2i.vae = AutoencoderKL.from_single_file(v_path, torch_dtype=DTYPE, config=tmpdir, device_map=vae_device_map)
                        print(get_message("vae_loaded"))
                else:
                    pipe_i2i.vae = AutoencoderKL.from_pretrained(v_path, torch_dtype=DTYPE, device_map=vae_device_map)
            except Exception as e:
                print(get_message("vae_failed", e))

    if DEVICE == "cuda":
        if is_low_vram:
            print(get_message("forcing_to_ram"))
            pipe_i2i.to("cpu")
            print(get_message("model_to_ram"))
            pipe_i2i.enable_sequential_cpu_offload()
            print(get_message("i2i_low_vram"))
        else:
            pipe_i2i.to("cuda")
            print(get_message("i2i_high_end"))

    current_model_config["i2i"] = config_key
    print("✅ 图生图 Pipeline 加载完成")
    return pipe_i2i

def interrupt_callback(pipe, step, timestep, callback_kwargs):
    global is_generating_interrupted
    if is_generating_interrupted:
        raise gr.Error("🛑 生成已被用户手动停止")
    return callback_kwargs

def scan_model_variants(root_dir):
    if not os.path.isdir(root_dir):
        return []
    items = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, "config.json")):
                items.append(name)
        elif name.lower().endswith((".safetensors", ".bin")):
            items.append(name)
    return sorted(items)

def get_choices(mod_root):
    return ["default"] + scan_model_variants(mod_root)

def resolve_model_path(choice, mod_root):
    if choice == "default":
        return None
    path = os.path.join(mod_root, choice)
    if os.path.exists(path):
        return path
    return None

def process_mask_for_inpaint(mask_image):
    if mask_image is None:
        return None
    if mask_image.mode == 'RGBA':
        import numpy as np
        mask_array = np.array(mask_image)
        alpha = mask_array[:, :, 3] if mask_array.shape[2] > 3 else None
        rgb = mask_array[:, :, :3]
        rgb_gray = np.dot(rgb, [0.299, 0.587, 0.114])
        if alpha is not None:
            mask_gray = np.where(alpha > 10, 255, 0).astype(np.uint8)
        else:
            mask_gray = np.where(rgb_gray > 10, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_gray, mode='L')
    else:
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')
        mask = mask_image.point(lambda p: 255 if p > 10 else 0)
    
    if mask.getextrema()[1] == 0: 
        return None
    return mask

# =========================
# 生成与编辑函数
# =========================

def generate_image(prompt, lora_choice, lora_alpha, num_images, image_format,
                   width, height, num_inference_steps, guidance_scale, seed, randomize_seed,
                   transformer_choice, vae_choice, vram_type_str, progress=gr.Progress()):
    global is_generating_interrupted
    is_generating_interrupted = False
    
    is_low_vram = "24GB" in vram_type_str or "Under 24GB" in vram_type_str or "24G以下" in vram_type_str or "24GB以下" in vram_type_str
    
    pipe_local = load_t2i_pipeline(transformer_choice, vae_choice, is_low_vram)
    pipe_local = apply_lora_to_pipeline(pipe_local, lora_choice, lora_alpha)

    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(day_dir, exist_ok=True)

    fmt_map = {"png": ("PNG", "png"), "jpeg": ("JPEG", "jpeg"), "webp": ("WEBP", "webp")}
    pil_fmt, ext = fmt_map[image_format.lower()]

    results = []
    try:
        for _ in progress.tqdm(range(int(num_images)), desc="生成中"):
            if is_generating_interrupted:
                break
            img = pipe_local(
                prompt=prompt.strip(),
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=interrupt_callback,
            ).images[0]
            filename = os.path.join(day_dir, f"{datetime.datetime.now():%H%M%S}_{uuid.uuid4().hex[:4]}.{ext}")
            img.save(filename, format=pil_fmt)
            results.append(filename)
    finally:
        auto_flush_vram()

    return results, seed

def run_img2img_enhanced(input_image, prompt, negative_prompt, lora_choice, lora_alpha, 
                         num_images, image_format,
                         out_w, out_h, i2i_mode, strength_ui, steps_ui, cfg_ui, 
                         seed, randomize_seed,
                         transformer_choice, vae_choice, vram_type_str, progress=gr.Progress()):
    global is_generating_interrupted
    is_generating_interrupted = False

    is_low_vram = "24GB" in vram_type_str or "Under 24GB" in vram_type_str or "24G以下" in vram_type_str or "24GB以下" in vram_type_str

    if input_image is None:
        raise gr.Error("⚠️ 请先上传图片！")

    try:
        pipe_local = load_i2i_pipeline(transformer_choice, vae_choice, is_low_vram)
    except Exception as e:
        if isinstance(e, gr.Error): raise e
        raise gr.Error(f"加载图生图模型失败: {str(e)}")

    if i2i_mode.startswith("A"):
        lora_scale = 0.35 
        strength = 0.30
        steps = 8
        cfg = 1.0
    else:
        lora_scale = 0.65 
        strength = 0.45
        steps = 6
        cfg = 1.5
    
    pipe_local = apply_lora_to_pipeline(pipe_local, lora_choice, lora_alpha, lora_scale)

    final_strength = strength_ui
    final_steps = int(steps_ui)
    final_cfg = cfg_ui
    
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    orig_w, orig_h = input_image.size
    if out_w == 0 or out_h == 0:
        target_size = 1024
        ratio = orig_w / orig_h
        if ratio > 1:
            w, h = target_size, int(target_size / ratio)
        else:
            w, h = int(target_size * ratio), target_size
    else:
        w, h = out_w, out_h
    
    w = (w // 16) * 16
    h = (h // 16) * 16
    input_image = input_image.resize((w, h), Image.LANCZOS)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(day_dir, exist_ok=True)

    fmt_map = {"png": ("PNG", "png"), "jpeg": ("JPEG", "jpeg"), "webp": ("WEBP", "webp")}
    pil_fmt, ext = fmt_map[image_format.lower()]

    results = []
    
    try:
        for _ in progress.tqdm(range(int(num_images)), desc="图生图中"):
            if is_generating_interrupted:
                break
            
            img = pipe_local(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip(),
                image=input_image,
                strength=final_strength,
                num_inference_steps=final_steps,
                guidance_scale=final_cfg,
                generator=generator,
                callback_on_step_end=interrupt_callback,
            ).images[0]
            
            filename = os.path.join(day_dir, f"i2i_{datetime.datetime.now():%H%M%S}_{uuid.uuid4().hex[:4]}.{ext}")
            img.save(filename, format=pil_fmt)
            results.append(filename)
    finally:
        auto_flush_vram()

    return results, seed

def run_inpainting(image_editor_data, prompt, negative_prompt, lora_choice, lora_alpha,
                   strength, steps, cfg, seed, randomize_seed,
                   transformer_choice, vae_choice, vram_type_str, progress=gr.Progress()):
    global is_generating_interrupted
    is_generating_interrupted = False

    is_low_vram = "24GB" in vram_type_str or "Under 24GB" in vram_type_str or "24G以下" in vram_type_str or "24GB以下" in vram_type_str

    input_image = None
    mask_layer = None
    
    if isinstance(image_editor_data, dict):
        if 'background' in image_editor_data:
            input_image = image_editor_data['background']
            if image_editor_data.get('layers'):
                mask_layer = image_editor_data['layers'][0]
    elif isinstance(image_editor_data, (tuple, list)):
        input_image = image_editor_data[0]
        mask_layer = image_editor_data[1]
    elif isinstance(image_editor_data, Image.Image):
        input_image = image_editor_data

    if input_image is None:
        raise gr.Error("⚠️ 请先上传图片！")
    
    if input_image.mode == 'RGBA':
        background = Image.new('RGB', input_image.size, (255,255,255))
        background.paste(input_image, (0, 0), input_image)
        input_image = background
    else:
        input_image = input_image.convert("RGB")

    if mask_layer is None:
        raise gr.Error("⚠️ 请使用画笔在图片上涂抹要修改的区域。")
    
    mask = process_mask_for_inpaint(mask_layer)
    if mask is None:
        raise gr.Error("⚠️ Mask 无效，请确保涂抹了区域。")

    try:
        pipe_local = load_i2i_pipeline(transformer_choice, vae_choice, is_low_vram)
    except Exception as e:
        raise gr.Error(f"模型加载失败: {str(e)}")
    
    pipe_local = apply_lora_to_pipeline(pipe_local, lora_choice, lora_alpha, lora_scale=0.6)

    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    orig_w, orig_h = input_image.size
    if mask.size != (orig_w, orig_h):
        mask = mask.resize((orig_w, orig_h), Image.LANCZOS)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(day_dir, exist_ok=True)

    result_img = None

    try:
        try:
            result_img = pipe_local(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip(),
                image=input_image,
                mask_image=mask,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                generator=generator,
                callback_on_step_end=interrupt_callback
            ).images[0]
        except (TypeError, AttributeError) as e:
            print(f"⚠️ 原生 Inpaint 失败 ({e})，使用手动混合模式...")
            
            img_array = np.array(input_image).astype(np.float32) /255.0
            mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0
            mask_3d = np.expand_dims(mask_array, axis=2)
            mask_3d = np.repeat(mask_3d,3, axis=2)
            
            noise = np.random.randn(*img_array.shape).astype(np.float32) * 0.1
            inpaint_input_array = img_array * (1 - mask_3d) + (img_array + noise) * mask_3d
            inpaint_input_array = np.clip(inpaint_input_array, 0, 1)
            inpaint_input = Image.fromarray((inpaint_input_array * 255).astype(np.uint8))
            
            generated = pipe_local(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip(),
                image=inpaint_input,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                generator=generator,
                callback_on_step_end=interrupt_callback
            ).images[0]
            
            if generated.size != (orig_w, orig_h):
                generated = generated.resize((orig_w, orig_h), Image.LANCZOS)
            
            gen_array = np.array(generated).astype(np.float32) / 255.0
            orig_array = np.array(input_image).astype(np.float32) / 255.0
            
            final_array = orig_array * (1 - mask_3d) + gen_array * mask_3d
            final_array = np.clip(final_array, 0, 1)
            result_img = Image.fromarray((final_array * 255).astype(np.uint8))

        filename = os.path.join(day_dir, f"inpaint_{datetime.datetime.now():%H%M%S}_{uuid.uuid4().hex[:4]}.png")
        result_img.save(filename)
        
    except Exception as e:
        if "任务已手动停止" in str(e): raise
        import traceback
        traceback.print_exc()
        raise gr.Error(f"局部重绘失败: {str(e)}")
    finally:
        auto_flush_vram()

    return [result_img], seed

def edit_image(image, angle, x, y, w, h, hflip, vflip, filter_name, brightness, contrast, saturation):
    if image is None:
        return None
    img = image.copy()
    if angle != 0:
        img = img.rotate(angle, expand=True)
    if x or y or w < 100 or h < 100:
        ow, oh = img.size
        left = int(ow * x / 100)
        top = int(oh * y / 100)
        right = int(ow * (x + w) / 100)
        bottom = int(oh * (y + h) / 100)
        img = img.crop((left, top, right, bottom))
    if hflip:
        img = ImageOps.mirror(img)
    if vflip:
        img = ImageOps.flip(img)
    if filter_name:
        filter_map = {
            "模糊": ImageFilter.BLUR, "轮廓": ImageFilter.CONTOUR, "细节": ImageFilter.DETAIL,
            "边缘增强": ImageFilter.EDGE_ENHANCE, "更多边缘增强": ImageFilter.EDGE_ENHANCE_MORE,
            "浮雕": ImageFilter.EMBOSS, "查找边缘": ImageFilter.FIND_EDGES,
            "锐化": ImageFilter.SHARPEN, "平滑": ImageFilter.SMOOTH, "更多平滑": ImageFilter.SMOOTH_MORE,
        }
        f = filter_map.get(filter_name)
        if f:
            img = img.filter(f)
    if brightness != 0:
        img = ImageEnhance.Brightness(img).enhance(1 + brightness / 100)
    if contrast != 0:
        img = ImageEnhance.Contrast(img).enhance(1 + contrast / 100)
    if saturation != 0:
        img = ImageEnhance.Color(img).enhance(1 + saturation / 100)
    return img

# =========================
# Gradio 界面构建
# =========================
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory if DEVICE == "cuda" else 0
DEFAULT_PERF_MODE = "高端机模式 (>=24GB)" if TOTAL_VRAM >= 24 * 1024**3 else "24GB以下 (优化模式)"

with gr.Blocks() as demo:
    lang_state = gr.State("zh")

    with gr.Row():
        title_md = gr.Markdown(TEXT["zh"]["title"])
        lang_btn = gr.Button(TEXT["zh"]["lang_btn"], size="sm")

    with gr.Tabs() as tabs:
        with gr.Tab(TEXT["zh"]["tab_generate"]) as tab_gen:
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(label=TEXT["zh"]["prompt"], lines=4, placeholder=TEXT["zh"]["prompt_placeholder"])
                    with gr.Row():
                        refresh_lora = gr.Button(TEXT["zh"]["refresh_lora"], size="sm")
                        refresh_model_t2i = gr.Button(TEXT["zh"]["refresh_model"], size="sm")
                    
                    lora_choices = ["None"] + scan_lora_items()
                    lora_drop = gr.Dropdown(label=TEXT["zh"]["lora_label"], choices=lora_choices, value="None")
                    lora_alpha = gr.Slider(0, 2, 1, step=0.05, label=TEXT["zh"]["lora_strength"])

                    model_section_md = gr.Markdown(TEXT["zh"]["model_section"])

                    with gr.Row():
                        transformer_choice = gr.Dropdown(label=TEXT["zh"]["transformer"], choices=get_choices(MOD_TRANSFORMER), value="default")
                        vae_choice = gr.Dropdown(label=TEXT["zh"]["vae"], choices=get_choices(MOD_VAE), value="default")

                    vram_type = gr.Radio(
                        [TEXT["zh"]["vram_low"], TEXT["zh"]["vram_high"]],
                        label=TEXT["zh"]["vram_type"],
                        value=DEFAULT_PERF_MODE
                    )
                    device_ui = gr.Radio(["cuda", "cpu"], label=TEXT["zh"]["device"], value="cuda" if torch.cuda.is_available() else "cpu", visible=False)
                    
                    num_images = gr.Slider(1, 8, 1, step=1, label=TEXT["zh"]["num_images"])
                    image_format = gr.Dropdown(["png", "jpeg", "webp"], value="png", label=TEXT["zh"]["output_format"])

                    with gr.Row():
                        width = gr.Slider(512, 2048, 1024, step=64, label=TEXT["zh"]["width"])
                        height = gr.Slider(512, 2048, 1024, step=64, label=TEXT["zh"]["height"])
                    num_inference_steps = gr.Slider(1, 50, 10, step=1, label=TEXT["zh"]["steps"])
                    guidance_scale = gr.Slider(0, 10, 0, step=0.1, label=TEXT["zh"]["cfg"])
                    seed = gr.Number(label=TEXT["zh"]["seed"], value=42, precision=0)
                    randomize_seed = gr.Checkbox(label=TEXT["zh"]["random_seed"], value=True)

                    with gr.Row():
                        generate_btn = gr.Button(TEXT["zh"]["generate"], variant="primary", size="lg")
                        stop_btn = gr.Button(TEXT["zh"]["stop"], variant="stop", size="lg", interactive=False)

                with gr.Column(scale=6):
                    gallery = gr.Gallery(label=TEXT["zh"]["gallery"], columns=2, height="80vh")
                    used_seed = gr.Number(label=TEXT["zh"]["used_seed"], interactive=False)

        with gr.Tab(TEXT["zh"]["tab_edit"]) as tab_edit:
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label=TEXT["zh"]["edit_upload"], type="pil")
                    with gr.Group():
                        rotate_angle = gr.Slider(-360, 360, 0, step=1, label=TEXT["zh"]["rotate"])
                        crop_x = gr.Slider(0, 100, 0, step=1, label=TEXT["zh"]["crop_x"])
                        crop_y = gr.Slider(0, 100, 0, step=1, label=TEXT["zh"]["crop_y"])
                        crop_width = gr.Slider(0, 100, 100, step=1, label=TEXT["zh"]["crop_w"])
                        crop_height = gr.Slider(0, 100, 100, step=1, label=TEXT["zh"]["crop_h"])
                        flip_horizontal = gr.Checkbox(label=TEXT["zh"]["hflip"])
                        flip_vertical = gr.Checkbox(label=TEXT["zh"]["vflip"])
                    edit_btn = gr.Button(TEXT["zh"]["edit_btn"], variant="primary")

                with gr.Column():
                    edited_image_output = gr.Image(label=TEXT["zh"]["edited_image"], type="pil")
                    with gr.Group():
                        apply_filter = gr.Dropdown(
                            ["模糊", "轮廓", "细节", "边缘增强", "更多边缘增强", "浮雕", "查找边缘", "锐化", "平滑", "更多平滑"],
                            label=TEXT["zh"]["filter"]
                        )
                        brightness = gr.Slider(-100, 100, 0, step=1, label=TEXT["zh"]["brightness"])
                        contrast = gr.Slider(-100, 100, 0, step=1, label=TEXT["zh"]["contrast"])
                        saturation = gr.Slider(-100, 100, 0, step=1, label=TEXT["zh"]["saturation"])

        with gr.Tab(TEXT["zh"]["tab_img2img"]) as tab_img2img:
            i2i_status_md = gr.Markdown(TEXT["zh"]["i2i_note"])
            with gr.Row():
                with gr.Column(scale=4):
                    i2i_image_input = gr.Image(label=TEXT["zh"]["i2i_ref"], type="pil")
                    
                    i2i_prompt = gr.Textbox(label=TEXT["zh"]["i2i_prompt"], lines=3, placeholder=TEXT["zh"]["i2i_ph"])
                    i2i_negative_prompt = gr.Textbox(label=TEXT["zh"]["negative_prompt"], lines=2, placeholder=TEXT["zh"]["negative_placeholder"])
                    
                    with gr.Row():
                        i2i_refresh_lora = gr.Button(TEXT["zh"]["refresh_lora"], size="sm")
                        i2i_refresh_model = gr.Button(TEXT["zh"]["refresh_model"], size="sm")
                    
                    i2i_lora_choices = ["None"] + scan_lora_items()
                    i2i_lora_drop = gr.Dropdown(label=TEXT["zh"]["lora_label"], choices=i2i_lora_choices, value="None")
                    i2i_lora_alpha = gr.Slider(0, 2, 1, step=0.05, label=TEXT["zh"]["lora_strength"])

                    with gr.Accordion(TEXT["zh"]["model_section"], open=False):
                        i2i_transformer_choice = gr.Dropdown(label=TEXT["zh"]["transformer"], choices=get_choices(MOD_TRANSFORMER), value="default")
                        i2i_vae_choice = gr.Dropdown(label=TEXT["zh"]["vae"], choices=get_choices(MOD_VAE), value="default")
                        i2i_vram_type = gr.Radio(
                            [TEXT["zh"]["vram_low"], TEXT["zh"]["vram_high"]],
                            label=TEXT["zh"]["vram_type"],
                            value=DEFAULT_PERF_MODE
                        )

                    i2i_mode = gr.Radio(
                        [TEXT["zh"]["i2i_mode_a"], TEXT["zh"]["i2i_mode_b"]],
                        label=TEXT["zh"]["i2i_mode"],
                        value=TEXT["zh"]["i2i_mode_a"]
                    )
                    
                    with gr.Row():
                        i2i_out_w = gr.Slider(0, 2048, 0, step=16, label=TEXT["zh"]["i2i_out_w"])
                        i2i_out_h = gr.Slider(0, 2048, 0, step=16, label=TEXT["zh"]["i2i_out_h"])
                    i2i_tip_md = gr.Markdown(TEXT["zh"]["i2i_tip"])
                    
                    i2i_strength = gr.Slider(0.1, 1.0, 0.4, step=0.05, label=TEXT["zh"]["i2i_strength"])
                    i2i_steps = gr.Slider(1, 50, 6, step=1, label=TEXT["zh"]["steps"])
                    i2i_cfg = gr.Slider(0.0, 5.0, 1.0, step=0.1, label=TEXT["zh"]["cfg"])
                    
                    i2i_num_images = gr.Slider(1, 4, 1, step=1, label=TEXT["zh"]["num_images"])
                    i2i_image_format = gr.Dropdown(["png", "jpeg", "webp"], value="png", label=TEXT["zh"]["output_format"])
                    i2i_seed = gr.Number(label=TEXT["zh"]["seed"], value=42, precision=0)
                    i2i_randomize_seed = gr.Checkbox(label=TEXT["zh"]["random_seed"], value=True)

                    with gr.Row():
                        i2i_generate_btn = gr.Button(TEXT["zh"]["i2i_btn"], variant="primary", size="lg")
                        i2i_stop_btn = gr.Button(TEXT["zh"]["stop"], variant="stop", size="lg", interactive=False)

                with gr.Column(scale=6):
                    i2i_gallery = gr.Gallery(label=TEXT["zh"]["gallery"], columns=2, height="80vh")
                    i2i_used_seed = gr.Number(label=TEXT["zh"]["used_seed"], interactive=False)

        with gr.Tab(TEXT["zh"]["tab_inpaint"]) as tab_inpaint:
            with gr.Row():
                with gr.Column(scale=4):
                    inpaint_editor = gr.ImageEditor(
                        label=TEXT["zh"]["inpaint_upload"],
                        type="pil",
                        layers=True,
                        eraser=True,
                        brush=gr.Brush(colors=["#FFFFFF", "#000000", "#FF0000"], color_mode="fixed")
                    )
                    inpaint_tip_md = gr.Markdown(TEXT["zh"]["inpaint_desc"])
                    
                    inpaint_prompt = gr.Textbox(label=TEXT["zh"]["i2i_prompt"], lines=3, placeholder=TEXT["zh"]["i2i_ph"])
                    inpaint_negative_prompt = gr.Textbox(label=TEXT["zh"]["negative_prompt"], lines=2, placeholder=TEXT["zh"]["negative_placeholder"])

                    with gr.Row():
                        inpaint_refresh_lora = gr.Button(TEXT["zh"]["refresh_lora"], size="sm")
                        inpaint_refresh_model = gr.Button(TEXT["zh"]["refresh_model"], size="sm")
                    
                    inpaint_lora_choices = ["None"] + scan_lora_items()
                    inpaint_lora_drop = gr.Dropdown(label=TEXT["zh"]["lora_label"], choices=inpaint_lora_choices, value="None")
                    inpaint_lora_alpha = gr.Slider(0, 2, 1, step=0.05, label=TEXT["zh"]["lora_strength"])

                    with gr.Accordion(TEXT["zh"]["model_section"], open=False):
                        inpaint_transformer_choice = gr.Dropdown(label=TEXT["zh"]["transformer"], choices=get_choices(MOD_TRANSFORMER), value="default")
                        inpaint_vae_choice = gr.Dropdown(label=TEXT["zh"]["vae"], choices=get_choices(MOD_VAE), value="default")
                        inpaint_vram_type = gr.Radio(
                            [TEXT["zh"]["vram_low"], TEXT["zh"]["vram_high"]],
                            label=TEXT["zh"]["vram_type"],
                            value=DEFAULT_PERF_MODE
                        )
                    
                    inpaint_strength = gr.Slider(0.1, 1.0, 0.7, step=0.05, label=TEXT["zh"]["i2i_strength"])
                    inpaint_steps = gr.Slider(1, 50, 8, step=1, label=TEXT["zh"]["steps"])
                    inpaint_cfg = gr.Slider(0.0, 5.0, 1.0, step=0.1, label=TEXT["zh"]["cfg"])
                    
                    inpaint_seed = gr.Number(label=TEXT["zh"]["seed"], value=42, precision=0)
                    inpaint_randomize_seed = gr.Checkbox(label=TEXT["zh"]["random_seed"], value=True)

                    with gr.Row():
                        inpaint_generate_btn = gr.Button(TEXT["zh"]["i2i_btn"], variant="primary", size="lg")
                        inpaint_stop_btn = gr.Button(TEXT["zh"]["stop"], variant="stop", size="lg", interactive=False)

                with gr.Column(scale=6):
                    inpaint_gallery = gr.Gallery(label=TEXT["zh"]["gallery"], columns=2, height="80vh")
                    inpaint_used_seed = gr.Number(label=TEXT["zh"]["used_seed"], interactive=False)

    def switch_language_full(lang):
        new_lang = "en" if lang == "zh" else "zh"
        t = TEXT[new_lang]
        
        # 修复：根据硬件显存大小，决定当前应该选哪个语言版本的选项
        is_low_vram_hardware = TOTAL_VRAM < 24 * 1024**3
        current_vram_val = t['vram_low'] if is_low_vram_hardware else t['vram_high']
        
        # 修复：更新显存选项的值，不仅仅是选项列表
        return (
            new_lang, t['title'], t['lang_btn'],
            gr.update(label=t['tab_generate']), gr.update(label=t['tab_edit']), 
            gr.update(label=t['tab_img2img']), gr.update(label=t['tab_inpaint']),
            gr.update(label=t['prompt'], placeholder=t['prompt_placeholder']),
            gr.update(value=t['refresh_lora']), gr.update(value=t['refresh_model']),
            gr.update(label=t['lora_label']), gr.update(label=t['lora_strength']),
            t['model_section'], gr.update(label=t['transformer']), gr.update(label=t['vae']),
            # T2I VRAM: 更新选项和值
            gr.update(label=t['vram_type'], choices=[t['vram_low'], t['vram_high']], value=current_vram_val), 
            gr.update(label=t['device']),
            gr.update(label=t['num_images']), gr.update(label=t['output_format']),
            gr.update(label=t['width']), gr.update(label=t['height']),
            gr.update(label=t['steps']), gr.update(label=t['cfg']),
            gr.update(label=t['seed']), gr.update(label=t['random_seed']),
            gr.update(value=t['generate']), gr.update(value=t['stop']),
            gr.update(label=t['gallery']), gr.update(label=t['used_seed']),
            
            gr.update(label=t['edit_upload']),
            gr.update(label=t['rotate']), gr.update(label=t['crop_x']), gr.update(label=t['crop_y']),
            gr.update(label=t['crop_w']), gr.update(label=t['crop_h']),
            gr.update(label=t['hflip']), gr.update(label=t['vflip']),
            gr.update(value=t['edit_btn']), gr.update(label=t['edited_image']),
            gr.update(label=t['filter']), gr.update(label=t['brightness']), gr.update(label=t['contrast']), gr.update(label=t['saturation']),
            
            gr.update(value=t['i2i_note']),
            gr.update(label=t['i2i_ref']),
            gr.update(label=t['i2i_prompt'], placeholder=t['i2i_ph']),
            gr.update(label=t['negative_prompt'], placeholder=t['negative_placeholder']),
            gr.update(value=t['refresh_lora']), gr.update(value=t['refresh_model']),
            gr.update(label=t['lora_label']), gr.update(label=t['lora_strength']),
            gr.update(label=t['transformer']), gr.update(label=t['vae']),
            # Img2Img VRAM: 更新选项和值
            gr.update(label=t['vram_type'], choices=[t['vram_low'], t['vram_high']], value=current_vram_val),
            gr.update(label=t['i2i_mode'], choices=[t['i2i_mode_a'], t['i2i_mode_b']]),
            gr.update(label=t['i2i_out_w']), gr.update(label=t['i2i_out_h']),
            gr.update(value=t['i2i_tip']),
            gr.update(label=t['i2i_strength']),
            gr.update(label=t['steps']), gr.update(label=t['cfg']),
            gr.update(label=t['num_images']), gr.update(label=t['output_format']),
            gr.update(label=t['seed']), gr.update(label=t['random_seed']),
            gr.update(value=t['i2i_btn']), gr.update(value=t['stop']),
            gr.update(label=t['gallery']), gr.update(label=t['used_seed']),

            gr.update(label=t['inpaint_upload']),
            gr.update(value=t['inpaint_desc']),
            gr.update(label=t['i2i_prompt'], placeholder=t['i2i_ph']),
            gr.update(label=t['negative_prompt'], placeholder=t['negative_placeholder']),
            gr.update(value=t['refresh_lora']), gr.update(value=t['refresh_model']),
            gr.update(label=t['lora_label']), gr.update(label=t['lora_strength']),
            gr.update(label=t['transformer']), gr.update(label=t['vae']),
            # Inpaint VRAM: 更新选项和值
            gr.update(label=t['vram_type'], choices=[t['vram_low'], t['vram_high']], value=current_vram_val),
            gr.update(label=t['i2i_strength']),
            gr.update(label=t['steps']), gr.update(label=t['cfg']),
            gr.update(label=t['seed']), gr.update(label=t['random_seed']),
            gr.update(value=t['i2i_btn']), gr.update(value=t['stop']),
            gr.update(label=t['gallery']), gr.update(label=t['used_seed']),
        )

    lang_btn.click(
        fn=switch_language_full,
        inputs=lang_state,
        outputs=[
            lang_state, title_md, lang_btn,
            tab_gen, tab_edit, tab_img2img, tab_inpaint,
            prompt, refresh_lora, refresh_model_t2i, lora_drop, lora_alpha, model_section_md,
            transformer_choice, vae_choice, vram_type, device_ui, num_images, image_format,
            width, height, num_inference_steps, guidance_scale, seed, randomize_seed,
            generate_btn, stop_btn, gallery, used_seed,
            image_input, rotate_angle, crop_x, crop_y, crop_width, crop_height,
            flip_horizontal, flip_vertical, edit_btn, edited_image_output,
            apply_filter, brightness, contrast, saturation,
            i2i_status_md, i2i_image_input, i2i_prompt, i2i_negative_prompt, 
            i2i_refresh_lora, i2i_refresh_model, i2i_lora_drop, i2i_lora_alpha,
            i2i_transformer_choice, i2i_vae_choice, i2i_vram_type, i2i_mode,
            i2i_out_w, i2i_out_h, i2i_tip_md,
            i2i_strength, i2i_steps, i2i_cfg,
            i2i_num_images, i2i_image_format, i2i_seed, i2i_randomize_seed,
            i2i_generate_btn, i2i_stop_btn, i2i_gallery, i2i_used_seed,
            inpaint_editor, inpaint_tip_md,
            inpaint_prompt, inpaint_negative_prompt,
            inpaint_refresh_lora, inpaint_refresh_model, inpaint_lora_drop, inpaint_lora_alpha,
            inpaint_transformer_choice, inpaint_vae_choice, inpaint_vram_type,
            inpaint_strength, inpaint_steps, inpaint_cfg,
            inpaint_seed, inpaint_randomize_seed,
            inpaint_generate_btn, inpaint_stop_btn, inpaint_gallery, inpaint_used_seed
        ]
    )

    refresh_lora.click(fn=scan_lora_items, outputs=[lora_drop, i2i_lora_drop, inpaint_lora_drop])
    lora_drop.change(update_prompt_with_lora, [prompt, lora_drop, lora_alpha], prompt)

    def refresh_models_t2i():
        return gr.update(choices=get_choices(MOD_TRANSFORMER)), gr.update(choices=get_choices(MOD_VAE))
    refresh_model_t2i.click(fn=refresh_models_t2i, outputs=[transformer_choice, vae_choice])

    def start_gen(): return gr.update(interactive=False), gr.update(interactive=True)
    def end_gen(): return gr.update(interactive=True), gr.update(interactive=False)
    def trigger_stop():
        global is_generating_interrupted
        is_generating_interrupted = True

    generate_event = generate_btn.click(fn=start_gen, outputs=[generate_btn, stop_btn]).then(
        fn=generate_image,
        inputs=[prompt, lora_drop, lora_alpha, num_images, image_format,
                width, height, num_inference_steps, guidance_scale, seed, randomize_seed,
                transformer_choice, vae_choice, vram_type], 
        outputs=[gallery, used_seed]
    ).then(fn=end_gen, outputs=[generate_btn, stop_btn])

    stop_btn.click(fn=trigger_stop).then(fn=end_gen, outputs=[generate_btn, stop_btn], cancels=[generate_event])

    i2i_refresh_lora.click(fn=scan_lora_items, outputs=[lora_drop, i2i_lora_drop, inpaint_lora_drop])
    i2i_lora_drop.change(update_prompt_with_lora, [i2i_prompt, i2i_lora_drop, i2i_lora_alpha], i2i_prompt)

    def refresh_models_i2i():
        return gr.update(choices=get_choices(MOD_TRANSFORMER)), gr.update(choices=get_choices(MOD_VAE))
    i2i_refresh_model.click(fn=refresh_models_i2i, outputs=[i2i_transformer_choice, i2i_vae_choice])

    def start_i2i(): return gr.update(interactive=False), gr.update(interactive=True)
    def end_i2i(): return gr.update(interactive=True), gr.update(interactive=False)

    i2i_generate_event = i2i_generate_btn.click(fn=start_i2i, outputs=[i2i_generate_btn, i2i_stop_btn]).then(
        fn=run_img2img_enhanced,
        inputs=[i2i_image_input, i2i_prompt, i2i_negative_prompt, i2i_lora_drop, i2i_lora_alpha, 
                i2i_num_images, i2i_image_format,
                i2i_out_w, i2i_out_h, i2i_mode, i2i_strength, i2i_steps, i2i_cfg,
                i2i_seed, i2i_randomize_seed,
                i2i_transformer_choice, i2i_vae_choice, i2i_vram_type], 
        outputs=[i2i_gallery, i2i_used_seed]
    ).then(fn=end_i2i, outputs=[i2i_generate_btn, i2i_stop_btn])

    i2i_stop_btn.click(fn=trigger_stop).then(fn=end_i2i, outputs=[i2i_generate_btn, i2i_stop_btn], cancels=[i2i_generate_event])

    inpaint_refresh_lora.click(fn=scan_lora_items, outputs=[lora_drop, i2i_lora_drop, inpaint_lora_drop])
    inpaint_lora_drop.change(update_prompt_with_lora, [inpaint_prompt, inpaint_lora_drop, inpaint_lora_alpha], inpaint_prompt)

    def refresh_models_inpaint():
        return gr.update(choices=get_choices(MOD_TRANSFORMER)), gr.update(choices=get_choices(MOD_VAE))
    inpaint_refresh_model.click(fn=refresh_models_inpaint, outputs=[inpaint_transformer_choice, inpaint_vae_choice])

    def start_inpaint(): return gr.update(interactive=False), gr.update(interactive=True)
    def end_inpaint(): return gr.update(interactive=True), gr.update(interactive=False)

    inpaint_generate_event = inpaint_generate_btn.click(fn=start_inpaint, outputs=[inpaint_generate_btn, inpaint_stop_btn]).then(
        fn=run_inpainting,
        inputs=[inpaint_editor, inpaint_prompt, inpaint_negative_prompt, inpaint_lora_drop, inpaint_lora_alpha,
                inpaint_strength, inpaint_steps, inpaint_cfg,
                inpaint_seed, inpaint_randomize_seed,
                inpaint_transformer_choice, inpaint_vae_choice, inpaint_vram_type], 
        outputs=[inpaint_gallery, inpaint_used_seed]
    ).then(fn=end_inpaint, outputs=[inpaint_generate_btn, inpaint_stop_btn])

    inpaint_stop_btn.click(fn=trigger_stop).then(fn=end_inpaint, outputs=[inpaint_generate_btn, inpaint_stop_btn], cancels=[inpaint_generate_event])

    edit_btn.click(
        fn=edit_image,
        inputs=[image_input, rotate_angle, crop_x, crop_y, crop_width, crop_height,
                flip_horizontal, flip_vertical, apply_filter, brightness, contrast, saturation],
        outputs=edited_image_output
    )

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(show_error=True)
