import os
os.environ["DIFFUSERS_USE_PEFT_BACKEND"] = "1"

import uuid
import random
import gc
from datetime import datetime
import re
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

import torch
import gradio as gr
from diffusers import ZImagePipeline, AutoencoderKL, ZImageTransformer2DModel
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

# =========================
# 双语文本字典
# =========================
TEXT = {
    "zh": {
        "title": "Z-Image-Turbo Pro（Pinokio 专用版）",
        "lang_btn": "EN",
        "tab_generate": "图像生成",
        "tab_edit": "图片编辑",
        "prompt": "Prompt",
        "prompt_placeholder": "输入你的描述...",
        "refresh_lora": "🔄 刷新 LoRA",
        "lora_label": "LoRA",
        "lora_strength": "LoRA 强度",
        "model_section": "### 模型选择",
        "transformer": "Transformer",
        "vae": "VAE",
        "vram_type": "显存类型（决定运行模式，最小支持4GB显存）",
        "vram_low": "12G以下 (优化模式)",
        "vram_high": "高端机模式",
        "device": "设备",
        "num_images": "生成张数",
        "output_format": "输出格式",
        "width": "宽度",
        "height": "高度",
        "steps": "步数",
        "cfg": "CFG",
        "seed": "种子",
        "random_seed": "随机种子",
        "generate": "🚀 生成",
        "stop": "🛑 停止生成",
        "gallery": "生成结果",
        "used_seed": "使用种子",
        "edit_upload": "上传图片",
        "rotate": "旋转角度 (度)",
        "crop_x": "裁剪 X (%)",
        "crop_y": "裁剪 Y (%)",
        "crop_w": "裁剪宽度 (%)",
        "crop_h": "裁剪高度 (%)",
        "hflip": "水平翻转",
        "vflip": "垂直翻转",
        "edit_btn": "开始编辑",
        "edited_image": "编辑后的图片",
        "filter": "应用滤镜",
        "brightness": "亮度调整 (%)",
        "contrast": "对比度调整 (%)",
        "saturation": "饱和度调整 (%)",
    },
    "en": {
        "title": "Z-Image-Turbo Pro (Pinokio Edition)",
        "lang_btn": "中文",
        "tab_generate": "Image Generation",
        "tab_edit": "Image Editing",
        "prompt": "Prompt",
        "prompt_placeholder": "Enter your description...",
        "refresh_lora": "🔄 Refresh LoRA",
        "lora_label": "LoRA",
        "lora_strength": "LoRA Strength",
        "model_section": "### Model Selection",
        "transformer": "Transformer",
        "vae": "VAE",
        "vram_type": "VRAM Type (Performance Mode Minimum 4GB graphics memory is supported.)",
        "vram_low": "Under 12GB (Optimized Mode)",
        "vram_high": "High-End GPU Mode",
        "device": "Device",
        "num_images": "Number of Images",
        "output_format": "Output Format",
        "width": "Width",
        "height": "Height",
        "steps": "Steps",
        "cfg": "CFG",
        "seed": "Seed",
        "random_seed": "Random Seed",
        "generate": "🚀 Generate",
        "stop": "🛑 Stop Generation",
        "gallery": "Generated Images",
        "used_seed": "Used Seed",
        "edit_upload": "Upload Image",
        "rotate": "Rotation Angle (degrees)",
        "crop_x": "Crop X (%)",
        "crop_y": "Crop Y (%)",
        "crop_w": "Crop Width (%)",
        "crop_h": "Crop Height (%)",
        "hflip": "Horizontal Flip",
        "vflip": "Vertical Flip",
        "edit_btn": "Apply Edit",
        "edited_image": "Edited Image",
        "filter": "Apply Filter",
        "brightness": "Brightness Adjustment (%)",
        "contrast": "Contrast Adjustment (%)",
        "saturation": "Saturation Adjustment (%)",
    }
}

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_SNAPSHOT_DIR = os.path.join(
    BASE_DIR,
    "cache",
    "HF_HOME",
    "hub",
    "models--Tongyi-MAI--Z-Image-Turbo",
    "snapshots",
    "5f4b9cbb80cc95ba44fe6667dfd75710f7db2947",
)

TRANSFORMER_ROOT = os.path.join(BASE_SNAPSHOT_DIR, "transformer")
TEXT_ENCODER_ROOT = os.path.join(BASE_SNAPSHOT_DIR, "text_encoder")
VAE_ROOT = os.path.join(BASE_SNAPSHOT_DIR, "vae")

MOD_DIR = os.path.join(BASE_DIR, "MOD")
MOD_TRANSFORMER = os.path.join(MOD_DIR, "transformer")
MOD_VAE = os.path.join(MOD_DIR, "vae")

LORA_ROOT = os.path.join(BASE_DIR, "lora")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for p in [MOD_TRANSFORMER, MOD_VAE, LORA_ROOT, OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

# 全局变量
pipe = None
current_model_config = {"transformer": "default", "vae": "default", "is_low_vram": True}  # 改为 bool 判断
is_generating_interrupted = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cuda":
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ 运行于 CPU")

def auto_flush_vram():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# =========================
# LoRA 功能
# =========================
def scan_lora_items():
    if not os.path.isdir(LORA_ROOT):
        return []
    return sorted([f for f in os.listdir(LORA_ROOT) if f.lower().endswith(".safetensors")])

def update_prompt_with_lora(prompt, selected_loras, lora_alpha):
    prompt = (prompt or "").strip()
    prompt_clean = re.sub(r"<lora:[^>]+>", "", prompt).strip()
    if not selected_loras:
        return prompt_clean
    try:
        alpha = float(lora_alpha)
    except:
        alpha = 1.0
    alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
    tags = [f"<lora:{os.path.splitext(f)[0]}:{alpha_str}>" for f in selected_loras]
    return f"{prompt_clean} {' '.join(tags)}".strip()

def apply_lora_to_pipeline(pipe_local, selected_loras, lora_alpha):
    if pipe_local is None or not selected_loras:
        return pipe_local

    if pipe_local.transformer is None:
        print("⚠️ Transformer 未加载，无法应用 LoRA")
        return pipe_local

    try:
        alpha = float(lora_alpha)
    except:
        alpha = 1.0

    if hasattr(pipe_local, "unload_lora_weights"):
        try:
            pipe_local.unload_lora_weights()
        except:
            pass

    adapters = []
    for lora_file in selected_loras:
        adapter_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(lora_file)[0])
        pipe_local.load_lora_weights(LORA_ROOT, weight_name=lora_file, adapter_name=adapter_name)
        adapters.append(adapter_name)

    if adapters:
        pipe_local.set_adapters(adapters, adapter_weights=[alpha] * len(adapters))
        print(f"✅ 已加载 {len(adapters)} 个 LoRA")
    return pipe_local

# =========================
# 模型加载（使用 bool 判断显存模式）
# =========================
def scan_model_variants(root_dir):
    if not os.path.isdir(root_dir):
        return []
    items = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, "config.json")):
                items.append(name)
        elif name.lower().endswith(".safetensors"):
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

def load_pipeline(transformer_choice="default", vae_choice="default", is_low_vram=True):
    global pipe, current_model_config

    config_key = (transformer_choice, vae_choice, is_low_vram)
    if pipe is not None and current_model_config == config_key:
        return pipe

    auto_flush_vram()
    pipe = None

    print(f"🛠 正在加载模型 → Transformer: {transformer_choice} | VAE: {vae_choice} | 低显存模式: {is_low_vram}")

    # 始终加载官方 transformer
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

    # VAE
    vae = AutoencoderKL.from_pretrained(VAE_ROOT, torch_dtype=DTYPE, local_files_only=True)
    if vae_choice != "default":
        v_path = resolve_model_path(vae_choice, MOD_VAE)
        if v_path:
            if os.path.isfile(v_path):
                vae = AutoencoderKL.from_single_file(v_path, torch_dtype=DTYPE)
            else:
                vae = AutoencoderKL.from_pretrained(v_path, torch_dtype=DTYPE, local_files_only=True)

    # Text Encoder
    text_encoder = AutoModelForCausalLM.from_pretrained(TEXT_ENCODER_ROOT, torch_dtype=DTYPE, local_files_only=True)

    # 组装 Pipeline
    pipe = ZImagePipeline.from_pretrained(
        BASE_SNAPSHOT_DIR,
        torch_dtype=DTYPE,
        local_files_only=True,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
    )

    # 显存模式（使用 bool 判断，不依赖字符串）
    if DEVICE == "cuda":
        if is_low_vram:
            pipe.enable_sequential_cpu_offload()
            print("  - 已启用低显存优化模式")
        else:
            pipe.to("cuda")
            print("  - 已启用高端机模式")

    current_model_config = config_key
    print("✅ Pipeline 加载完成")
    return pipe

# =========================
# 中断与生成
# =========================
def interrupt_callback(pipe, step, timestep, callback_kwargs):
    global is_generating_interrupted
    if is_generating_interrupted:
        raise gr.Error("🛑 生成已被用户手动停止")
    return callback_kwargs

def generate_image(prompt, selected_loras, lora_alpha, device, num_images, image_format,
                   width, height, num_inference_steps, guidance_scale, seed, randomize_seed,
                   transformer_choice, vae_choice, vram_type_str, progress=gr.Progress()):
    global is_generating_interrupted
    is_generating_interrupted = False

    # 判断是否低显存模式（关键修复）
    is_low_vram = "12G" in vram_type_str or "Under 12GB" in vram_type_str

    pipe_local = load_pipeline(transformer_choice, vae_choice, is_low_vram)
    pipe_local = apply_lora_to_pipeline(pipe_local, selected_loras, lora_alpha)

    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(int(seed))

    date_str = datetime.now().strftime("%Y-%m-%d")
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
            filename = os.path.join(day_dir, f"{datetime.now():%H%M%S}_{uuid.uuid4().hex[:4]}.{ext}")
            img.save(filename, format=pil_fmt)
            results.append(filename)
    finally:
        auto_flush_vram()

    return results, seed

# =========================
# 图片编辑（保持不变）
# =========================
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
            "模糊": ImageFilter.BLUR,
            "轮廓": ImageFilter.CONTOUR,
            "细节": ImageFilter.DETAIL,
            "边缘增强": ImageFilter.EDGE_ENHANCE,
            "更多边缘增强": ImageFilter.EDGE_ENHANCE_MORE,
            "浮雕": ImageFilter.EMBOSS,
            "查找边缘": ImageFilter.FIND_EDGES,
            "锐化": ImageFilter.SHARPEN,
            "平滑": ImageFilter.SMOOTH,
            "更多平滑": ImageFilter.SMOOTH_MORE,
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
# Gradio 界面（语言切换 + 显存模式保留）
# =========================
with gr.Blocks() as demo:
    lang_state = gr.State("zh")

    with gr.Row():
        title_md = gr.Markdown("## " + TEXT["zh"]["title"])
        lang_btn = gr.Button(TEXT["zh"]["lang_btn"], size="sm")

    with gr.Tabs() as tabs:
        tab_gen = gr.Tab(TEXT["zh"]["tab_generate"])
        with tab_gen:
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(label=TEXT["zh"]["prompt"], lines=4, placeholder=TEXT["zh"]["prompt_placeholder"])
                    with gr.Row():
                        refresh_lora = gr.Button(TEXT["zh"]["refresh_lora"], size="sm")
                    lora_list = gr.CheckboxGroup(label=TEXT["zh"]["lora_label"], choices=scan_lora_items())
                    lora_alpha = gr.Slider(0, 2, 1, step=0.05, label=TEXT["zh"]["lora_strength"])

                    model_section_md = gr.Markdown(TEXT["zh"]["model_section"])

                    with gr.Row():
                        transformer_choice = gr.Dropdown(label=TEXT["zh"]["transformer"], choices=get_choices(MOD_TRANSFORMER), value="default")
                        vae_choice = gr.Dropdown(label=TEXT["zh"]["vae"], choices=get_choices(MOD_VAE), value="default")

                    vram_type = gr.Radio(
                        [TEXT["zh"]["vram_low"], TEXT["zh"]["vram_high"]],
                        label=TEXT["zh"]["vram_type"],
                        value=TEXT["zh"]["vram_low"]
                    )

                    device = gr.Radio(["cuda", "cpu"], label=TEXT["zh"]["device"], value="cuda" if torch.cuda.is_available() else "cpu")
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

        tab_edit = gr.Tab(TEXT["zh"]["tab_edit"])
        with tab_edit:
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

    # 语言切换（不再依赖字符串匹配）
    def switch_language(lang):
        new_lang = "en" if lang == "zh" else "zh"
        t = TEXT[new_lang]
        return (
            new_lang,
            f"## {t['title']}",
            t['lang_btn'],
            gr.update(label=t['tab_generate']),
            gr.update(label=t['tab_edit']),
            gr.update(label=t['prompt'], placeholder=t['prompt_placeholder']),
            gr.update(value=t['refresh_lora']),
            gr.update(label=t['lora_label']),
            gr.update(label=t['lora_strength']),
            t['model_section'],
            gr.update(label=t['transformer']),
            gr.update(label=t['vae']),
            gr.update(label=t['vram_type'], choices=[t['vram_low'], t['vram_high']]),
            gr.update(label=t['device']),
            gr.update(label=t['num_images']),
            gr.update(label=t['output_format']),
            gr.update(label=t['width']),
            gr.update(label=t['height']),
            gr.update(label=t['steps']),
            gr.update(label=t['cfg']),
            gr.update(label=t['seed']),
            gr.update(label=t['random_seed']),
            gr.update(value=t['generate']),
            gr.update(value=t['stop']),
            gr.update(label=t['gallery']),
            gr.update(label=t['used_seed']),
            gr.update(label=t['edit_upload']),
            gr.update(label=t['rotate']),
            gr.update(label=t['crop_x']),
            gr.update(label=t['crop_y']),
            gr.update(label=t['crop_w']),
            gr.update(label=t['crop_h']),
            gr.update(label=t['hflip']),
            gr.update(label=t['vflip']),
            gr.update(value=t['edit_btn']),
            gr.update(label=t['edited_image']),
            gr.update(label=t['filter']),
            gr.update(label=t['brightness']),
            gr.update(label=t['contrast']),
            gr.update(label=t['saturation']),
        )

    lang_btn.click(
        fn=switch_language,
        inputs=lang_state,
        outputs=[
            lang_state, title_md, lang_btn,
            tab_gen, tab_edit,
            prompt, refresh_lora, lora_list, lora_alpha,
            model_section_md,
            transformer_choice, vae_choice, vram_type, device,
            num_images, image_format, width, height,
            num_inference_steps, guidance_scale, seed, randomize_seed,
            generate_btn, stop_btn, gallery, used_seed,
            image_input, rotate_angle, crop_x, crop_y, crop_width, crop_height,
            flip_horizontal, flip_vertical, edit_btn, edited_image_output,
            apply_filter, brightness, contrast, saturation
        ]
    )

    # 其他事件
    refresh_lora.click(fn=scan_lora_items, outputs=lora_list)
    lora_list.change(update_prompt_with_lora, [prompt, lora_list, lora_alpha], prompt)
    lora_alpha.change(update_prompt_with_lora, [prompt, lora_list, lora_alpha], prompt)

    def start_gen(): return gr.update(interactive=False), gr.update(interactive=True)
    def end_gen(): return gr.update(interactive=True), gr.update(interactive=False)
    def trigger_stop():
        global is_generating_interrupted
        is_generating_interrupted = True

    generate_event = generate_btn.click(fn=start_gen, outputs=[generate_btn, stop_btn]).then(
        fn=generate_image,
        inputs=[prompt, lora_list, lora_alpha, device, num_images, image_format,
                width, height, num_inference_steps, guidance_scale, seed, randomize_seed,
                transformer_choice, vae_choice, vram_type],
        outputs=[gallery, used_seed]
    ).then(fn=end_gen, outputs=[generate_btn, stop_btn])

    stop_btn.click(fn=trigger_stop).then(fn=end_gen, outputs=[generate_btn, stop_btn], cancels=[generate_event])

    edit_btn.click(
        fn=edit_image,
        inputs=[image_input, rotate_angle, crop_x, crop_y, crop_width, crop_height,
                flip_horizontal, flip_vertical, apply_filter, brightness, contrast, saturation],
        outputs=edited_image_output
    )

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(show_error=True)