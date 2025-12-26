import os
os.environ["DIFFUSERS_USE_PEFT_BACKEND"] = "1"  # Enable The PEFT LoRA backend of diffusers

import uuid
import random
from datetime import datetime
import re

import torch
import gradio as gr
from diffusers import ZImagePipeline, AutoencoderKL, ZImageTransformer2DModel
from transformers import AutoModelForCausalLM


# =========================
# Path configuration (all based on the directory where app.py is located)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Official model: Z-Image-Turbo snapshot in the HF cache
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

# Custom model directory (All the models you added later will be placed here)
MOD_DIR = os.path.join(BASE_DIR, "MOD")
MOD_TRANSFORMER = os.path.join(MOD_DIR, "transformer")
MOD_TEXT_ENCODER = os.path.join(MOD_DIR, "text_encoder")
MOD_VAE = os.path.join(MOD_DIR, "vae")

# LoRA and output
LORA_ROOT = os.path.join(BASE_DIR, "lora")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for p in [MOD_TRANSFORMER, MOD_TEXT_ENCODER, MOD_VAE, LORA_ROOT, OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

print("=== BASE_DIR ===", BASE_DIR)
print("=== OFFICIAL SNAPSHOT DIR ===", BASE_SNAPSHOT_DIR)
print("=== MOD DIR ===", MOD_DIR)

# Global pipeline cache
pipe = None
current_model_config = {
    "transformer": "default",
    "text_encoder": "default",
    "vae": "default",
}


# =========================
# LoRA Tools
# =========================

def scan_lora_items():
    """Scan all .safetensors files under ./lora as optional LoRAs."""
    if not os.path.isdir(LORA_ROOT):
        return []
    items = []
    for name in sorted(os.listdir(LORA_ROOT)):
        full = os.path.join(LORA_ROOT, name)
        if os.path.isfile(full) and name.lower().endswith(".safetensors"):
            items.append(name)
    return items


def build_lora_tags(selected_loras, lora_alpha):
    """Generate <lora:name:alpha> tags and unify alpha."""
    tags = []
    try:
        alpha = float(lora_alpha)
    except Exception:
        alpha = 1.0

    alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")

    for fname in selected_loras or []:
        base = os.path.splitext(os.path.basename(fname))[0]
        if not base:
            continue
        tags.append(f"<lora:{base}:{alpha_str}>")
    return tags


def update_prompt_with_lora(prompt, selected_loras, lora_alpha):
    """Embed/Update LoRA tags in the prompt"""
    prompt = prompt or ""
    # 先清理掉旧的 <lora:...> 标签
    prompt_clean = re.sub(r"<lora:[^>]+>", "", prompt).strip()
    tags = build_lora_tags(selected_loras, lora_alpha)
    if tags:
        if prompt_clean:
            prompt_clean = prompt_clean + " " + " ".join(tags)
        else:
            prompt_clean = " ".join(tags)
    return prompt_clean


def apply_lora_to_pipeline(pipe_local, selected_loras, lora_alpha):
    """Inject LoRA into the pipeline (diffusers PEFT backend, multiple LoRAs + alpha)"""
    if pipe_local is None:
        return None
    if not selected_loras:
        return pipe_local

    try:
        alpha = float(lora_alpha)
    except Exception:
        alpha = 1.0

    adapter_names = []

    for lora_file in selected_loras:
        lora_path = os.path.join(LORA_ROOT, lora_file)
        if not os.path.isfile(lora_path):
            print(f"[LoRA] The file does not exist, skipping.: {lora_path}")
            continue

        base_name = os.path.splitext(os.path.basename(lora_file))[0]
        safe_adapter_name = re.sub(r"[^a-zA-Z0-9_]", "_", base_name)

        try:
            print(f"[LoRA] Loading: {lora_path} as adapter '{safe_adapter_name}'")
            pipe_local.load_lora_weights(
                lora_path,
                adapter_name=safe_adapter_name,
            )
            adapter_names.append(safe_adapter_name)
        except Exception as e:
            print(f"❌ [LoRA] load failed {lora_file}: {e}")

    if adapter_names:
        pipe_local.set_adapters(
            adapter_names=adapter_names,
            adapter_weights=[alpha] * len(adapter_names),
        )
        print(f"✅ [LoRA] activated {len(adapter_names)} LoRAs, alpha={alpha}")
    else:
        print("[LoRA] No LoRA adapters were successfully loaded.")

    return pipe_local


# =========================
# Model scanning and loading
# =========================

def scan_model_variants(root_dir, label="Model"):
    """
    Scan the "available model subdirectories" under root_dir.

    Rules: Only consider a directory as an optional model if:
    - It is a subdirectory
    - The subdirectory contains config.json
    - And it contains at least one .safetensors or .safetensors.index.json

    This allows compatibility with:
    - Diffusers style: config.json + diffusion_pytorch_model.safetensors
    - Z-Image AE: config.json + ae.safetensors
    """
    if not os.path.isdir(root_dir):
        return []

    variants = []
    print(f"🔍 [Scan] {label}: {root_dir}")

    for name in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, name)
        if not os.path.isdir(subdir):
            continue

        has_config = os.path.isfile(os.path.join(subdir, "config.json"))

        has_safetensors = False
        for f in os.listdir(subdir):
            if f.endswith(".safetensors") or f.endswith(".safetensors.index.json"):
                has_safetensors = True
                break

        if has_config and has_safetensors:
            variants.append(name)

    return variants


def get_choices(mod_root, label):
    """Scan custom models only from the MOD folder, default = official snapshot"""
    variants = scan_model_variants(mod_root, label=f"Custom-{label}")
    return ["default"] + sorted(list(set(variants)))


def resolve_model_dir(choice, mod_root):
    """Parse to the corresponding directory based on the drop-down selection result; return None by default."""
    if choice == "default":
        return None
    subdir = os.path.join(mod_root, choice)
    if os.path.isdir(subdir):
        return subdir
    print(f"❌ [Model] 未找到模型目录: {subdir}")
    return None


def pick_vae_weight_name(vae_dir):
    """
    为 VAE 选择合适的 safetensors 文件名：
    - 优先 ae.safetensors
    - 其次 diffusion_pytorch_model.safetensors
    - 否则 None（交给 diffusers 自动判断）
    """
    candidates = [
        "ae.safetensors",
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
    ]
    for name in candidates:
        if os.path.isfile(os.path.join(vae_dir, name)):
            return name
    return None


def load_pipeline(
    transformer_choice: str = "default",
    text_encoder_choice: str = "default",
    vae_choice: str = "default",
):
    """按选择（T / TE / VAE）组装或复用 Z-Image pipeline"""
    global pipe, current_model_config

    config_tuple = {
        "transformer": transformer_choice,
        "text_encoder": text_encoder_choice,
        "vae": vae_choice,
    }

    # 如果配置没变，直接复用
    if pipe is not None and config_tuple == current_model_config:
        return pipe

    pipe = None
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    use_default = (
        transformer_choice == "default"
        and text_encoder_choice == "default"
        and vae_choice == "default"
    )

    if use_default:
        print("🛠 正在加载默认 Z-Image-Turbo Pipeline（全官方组件）...")
        pipe_local = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
    else:
        print(
            f"🛠 正在加载自定义 Pipeline: "
            f"T={transformer_choice}, TE={text_encoder_choice}, VAE={vae_choice}"
        )
        base_repo = "Tongyi-MAI/Z-Image-Turbo"

        # ==== Transformer ====
        transformer_dir = resolve_model_dir(transformer_choice, MOD_TRANSFORMER)
        if transformer_dir is not None:
            print(f"  - 自定义 Transformer: {transformer_dir}")
            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_dir,
                torch_dtype=torch.bfloat16,
            )
        else:
            transformer = None

        # ==== Text Encoder ====
        text_encoder_dir = resolve_model_dir(text_encoder_choice, MOD_TEXT_ENCODER)
        if text_encoder_dir is not None:
            print(f"  - 自定义 Text Encoder: {text_encoder_dir}")
            text_encoder = AutoModelForCausalLM.from_pretrained(
                text_encoder_dir,
                torch_dtype=torch.bfloat16,
            )
        else:
            text_encoder = None

        # ==== VAE ====
        vae_dir = resolve_model_dir(vae_choice, MOD_VAE)
        if vae_dir is not None:
            print(f"  - 自定义 VAE: {vae_dir}")
            weight_name = pick_vae_weight_name(vae_dir)
            if weight_name:
                print(f"    - 使用权重文件: {weight_name}")
                vae = AutoencoderKL.from_pretrained(
                    vae_dir,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    weight_name=weight_name,
                )
            else:
                print("    - 未显式找到 safetensors，尝试默认加载")
                vae = AutoencoderKL.from_pretrained(
                    vae_dir,
                    torch_dtype=torch.bfloat16,
                )
        else:
            vae = None

        pipe_local = ZImagePipeline.from_pretrained(
            base_repo,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
        )

    pipe = pipe_local
    current_model_config = config_tuple
    print("✅ Pipeline 已加载:", current_model_config)
    return pipe


def normalize_format(fmt: str):
    fmt = (fmt or "png").lower()
    if fmt == "jpeg":
        return "JPEG", "jpeg"
    if fmt == "webp":
        return "WEBP", "webp"
    return "PNG", "png"


# =========================
# 核心生成函数
# =========================

def generate_image(
    prompt,
    selected_loras,
    lora_alpha,
    device,
    num_images,
    image_format,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    seed,
    randomize_seed,
    transformer_choice,
    text_encoder_choice,
    vae_choice,
):
    # 1. 加载 / 切换 pipeline
    pipe_local = load_pipeline(
        transformer_choice=transformer_choice,
        text_encoder_choice=text_encoder_choice,
        vae_choice=vae_choice,
    )

    if pipe_local is None:
        raise gr.Error("Pipeline 加载失败，请查看控制台日志。")

    # 2. 设备
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ 选择了 cuda 但当前环境没有可用 GPU，自动切换到 cpu。")
        device = "cpu"
    pipe_local.to(device)

    # 3. 注入 LoRA
    pipe_local = apply_lora_to_pipeline(
        pipe_local,
        selected_loras,
        lora_alpha,
    )

    # 4. 种子
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(generator_device).manual_seed(seed)

    # 5. 输出目录
    date_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(day_dir, exist_ok=True)

    pil_format, ext = normalize_format(image_format)
    effective_prompt = (prompt or "").strip()

    print(
        f"🚀 生成中: {width}x{height}, steps={num_inference_steps}, "
        f"guidance={guidance_scale}, seed={seed}, device={device}"
    )

    filepaths = []
    try:
        for _ in range(int(num_images)):
            result = pipe_local(
                prompt=effective_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0]

            timestamp = datetime.now().strftime("%H%M%S")
            unique = str(uuid.uuid4())[:8]
            filename = os.path.join(day_dir, f"image_{timestamp}_{unique}.{ext}")
            image.save(filename, format=pil_format)
            filepaths.append(filename)

        return filepaths, seed
    except Exception as e:
        print(f"💥 生成出错: {e}")
        raise gr.Error(f"生成出错: {e}")


# =========================
# 预扫描 / 默认值
# =========================

default_device = "cuda" if torch.cuda.is_available() else "cpu"
initial_lora_items = scan_lora_items()

transformer_choices = get_choices(MOD_TRANSFORMER, "Transformer")
text_encoder_choices = get_choices(MOD_TEXT_ENCODER, "TextEncoder")
vae_choices = get_choices(MOD_VAE, "VAE")


# =========================
# Gradio 界面
# =========================

with gr.Blocks(title="Z-Image-Turbo Pro") as demo:
    gr.Markdown(
        """
        # 🎨 Z-Image-Turbo Pro（MOD 专业版 LeeWheel）

        - 官方底模：HF cache 中的 `Tongyi-MAI/Z-Image-Turbo`
        - 自定义模型：
          - `MOD/transformer/<Name>/` → Transformer
          - `MOD/text_encoder/<Name>/` → Text Encoder
          - `MOD/vae/<Name>/` → VAE（支持 AE：`config.json + ae.safetensors`）
        - LoRA：`lora/*.safetensors`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt / 提示词",
                placeholder="描述你想生成的图像...",
                lines=3,
            )

            # LoRA 区域
            gr.Markdown("### LoRA 设置（C 站 Z-Image LoRA 放在 ./lora）")
            with gr.Row():
                refresh_lora_btn = gr.Button("🔄 刷新 LoRA 列表", size="sm")
            lora_multiselect = gr.CheckboxGroup(
                label="选择 LoRA（可多选）",
                choices=initial_lora_items,
                value=[],
            )
            lora_alpha = gr.Slider(
                label="LoRA 全局强度 alpha",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
            )

            # 模型选择
            gr.Markdown("### 底模组件选择（官方 + MOD）")

            transformer_choice = gr.Dropdown(
                label="Transformer（底模）",
                choices=transformer_choices,
                value="default",
            )
            text_encoder_choice = gr.Dropdown(
                label="Text Encoder（文本编码器）",
                choices=text_encoder_choices,
                value="default",
            )
            vae_choice = gr.Dropdown(
                label="VAE（图像解码器）",
                choices=vae_choices,
                value="default",
            )

            # 设备与参数
            device = gr.Radio(
                label="推理设备 / Device",
                choices=["cuda", "cpu"],
                value=default_device,
            )

            num_images = gr.Slider(
                label="生成张数 / Number of Images",
                minimum=1,
                maximum=8,
                step=1,
                value=1,
            )

            image_format = gr.Dropdown(
                label="输出格式 / Output Format",
                choices=["png", "jpeg", "webp"],
                value="png",
            )

            gr.Markdown("**分辨率预设 / Resolution Presets**")
            with gr.Row():
                preset_512 = gr.Button("512×512", size="sm")
                preset_768 = gr.Button("768×768", size="sm")
                preset_1024 = gr.Button("1024×1024", size="sm")
                preset_landscape = gr.Button("1024×768", size="sm")
                preset_portrait = gr.Button("768×1024", size="sm")

            with gr.Row():
                width = gr.Slider(
                    label="宽度 Width",
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                )
                height = gr.Slider(
                    label="高度 Height",
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                )

            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="采样步数 / Inference Steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=10,
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                )

            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                )
                randomize_seed = gr.Checkbox(
                    label="Randomize Seed / 随机种子",
                    value=True,
                )

            generate_btn = gr.Button("🚀 生成 / Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                columns=2,
                rows=2,
                type="filepath",
            )
            used_seed = gr.Number(label="Seed Used", interactive=False)

    # LoRA & prompt 绑定
    refresh_lora_btn.click(
        fn=lambda: gr.update(choices=scan_lora_items(), value=[]),
        inputs=[],
        outputs=lora_multiselect,
    )

    lora_multiselect.change(
        fn=update_prompt_with_lora,
        inputs=[prompt, lora_multiselect, lora_alpha],
        outputs=prompt,
    )
    lora_alpha.change(
        fn=update_prompt_with_lora,
        inputs=[prompt, lora_multiselect, lora_alpha],
        outputs=prompt,
    )

    # 分辨率预设
    preset_512.click(fn=lambda: (512, 512), outputs=[width, height])
    preset_768.click(fn=lambda: (768, 768), outputs=[width, height])
    preset_1024.click(fn=lambda: (1024, 1024), outputs=[width, height])
    preset_landscape.click(fn=lambda: (1024, 768), outputs=[width, height])
    preset_portrait.click(fn=lambda: (768, 1024), outputs=[width, height])

    # 生成按钮
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            lora_multiselect,
            lora_alpha,
            device,
            num_images,
            image_format,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
            randomize_seed,
            transformer_choice,
            text_encoder_choice,
            vae_choice,
        ],
        outputs=[output_gallery, used_seed],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
    )
