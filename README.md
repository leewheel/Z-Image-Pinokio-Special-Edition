# Z-Image-Turbo ComfyUI-Style Gradio UI

A Gradio web interface that uses native ComfyUI nodes to run the Z-Image-Turbo workflow locally.

## Features

✨ **Native ComfyUI Implementation**
- Uses actual ComfyUI nodes for exact workflow control
- Supports all workflow features: KSampler, ModelSamplingAuraFlow, ConditioningZeroOut
- Optional Pixel Art LoRA support

🎨 **User-Friendly Interface**
- Clean, modern Gradio UI
- Real-time generation progress
- Preset aspect ratios (1:1, 16:9, 9:16, 4:3)
- Example prompts included

⚙️ **Full Control**
- Adjustable dimensions (512-2048px)
- Steps control (1-20, default 9)
- CFG scale adjustment
- Shift parameter for AuraFlow sampling
- Seed control with randomization
- Auto-save to outputs folder

## Prerequisites

### Option 1: Use Existing ComfyUI Installation (Recommended)

If you already have ComfyUI installed with the required models:

1. **ComfyUI** installed and working
2. **Models** in the correct folders:
   ```
   ComfyUI/
   ├── models/
   │   ├── text_encoders/
   │   │   └── qwen_3_4b.safetensors
   │   ├── diffusion_models/
   │   │   └── z_image_turbo_bf16.safetensors
   │   ├── vae/
   │   │   └── ae.safetensors
   │   └── loras/ (optional)
   │       └── pixel_art_style_z_image_turbo.safetensors
   ```

### Option 2: Standalone Setup

If you don't have ComfyUI:

1. Clone ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. Download models (see Model Downloads section below)

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **ComfyUI path is already configured** for ComfyUI Electron:
   ```python
   # Line 18 in app.py
   COMFY_PATH = Path(r"C:\Users\pierr\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI")
   ```

   If you have a different ComfyUI installation, update this path to:
   - `COMFY_PATH = Path("C:/ComfyUI")` (for standard installation)
   - `COMFY_PATH = Path.home() / "ComfyUI"` (for user directory)
   - Or your custom ComfyUI location

3. **Download models:**
   ```bash
   python download_models.py
   ```

   This will download ~20GB of required models to your ComfyUI models directory.

## Model Downloads

### Required Models

Download these models and place them in your ComfyUI `models` folder:

**Text Encoder (Qwen 3 4B):**
```bash
# Download to: ComfyUI/models/text_encoders/
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
```

**Diffusion Model (Z-Image-Turbo):**
```bash
# Download to: ComfyUI/models/diffusion_models/
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors
```

**VAE:**
```bash
# Download to: ComfyUI/models/vae/
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
```

### Optional Models

**Pixel Art LoRA:**
```bash
# Download to: ComfyUI/models/loras/
https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/pixel_art_style_z_image_turbo.safetensors
```

### Quick Download Script

```bash
# Create directories
mkdir -p ComfyUI/models/text_encoders
mkdir -p ComfyUI/models/diffusion_models
mkdir -p ComfyUI/models/vae
mkdir -p ComfyUI/models/loras

# Download models (using wget or curl)
cd ComfyUI/models

# Text encoder
wget https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors -P text_encoders/

# Diffusion model
wget https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors -P diffusion_models/

# VAE
wget https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors -P vae/

# Optional: Pixel Art LoRA
wget https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/pixel_art_style_z_image_turbo.safetensors -P loras/
```

## Usage

1. **Start the app:**
   ```bash
   python app_comfy.py
   ```

2. **Open browser:**
   Navigate to `http://127.0.0.1:7860`

3. **Generate images:**
   - Enter your prompt
   - Adjust settings (dimensions, steps, etc.)
   - Click "🚀 Generate"
   - Images are saved to `outputs/` folder

## Settings Guide

### Basic Settings

- **Prompt**: Describe what you want to generate (supports English & Chinese)
- **Width/Height**: Image dimensions (512-2048px, multiples of 64)
- **Steps**: Number of inference steps (9 recommended for quality/speed balance)
- **CFG Scale**: Guidance scale (1.0 default, higher = more prompt adherence)

### Advanced Settings

- **Shift (AuraFlow)**: Model sampling shift parameter (3.0 default from workflow)
- **Seed**: Random seed for reproducibility (-1 or check "Random" for random)
- **Enable Pixel Art LoRA**: Apply pixel art style transformation
- **LoRA Strength**: How strongly to apply the LoRA (0.0-2.0)

### Quick Aspect Ratios

- **1:1** - Square (1024x1024)
- **16:9** - Landscape (1344x768)
- **9:16** - Portrait (768x1344)
- **4:3** - Classic landscape (1152x896)

## Workflow Details

This implementation follows the exact ComfyUI workflow:

```
UNETLoader → (Optional: LoraLoaderModelOnly) → ModelSamplingAuraFlow → KSampler
CLIPLoader → CLIPTextEncode → ConditioningZeroOut (negative)
EmptySD3LatentImage → KSampler → VAEDecode → Output
```

### Key Nodes Used:

1. **CLIPLoader** - Loads Qwen 3 4B text encoder
2. **UNETLoader** - Loads Z-Image-Turbo diffusion model
3. **VAELoader** - Loads VAE for decoding
4. **LoraLoaderModelOnly** - Optional LoRA application
5. **ModelSamplingAuraFlow** - Applies shift parameter (3.0)
6. **CLIPTextEncode** - Encodes text prompt
7. **ConditioningZeroOut** - Creates zero negative conditioning
8. **EmptySD3LatentImage** - Creates initial latent
9. **KSampler** - Main sampling (res_multistep, simple scheduler)
10. **VAEDecode** - Decodes latent to image

## Troubleshooting

### "Cannot import ComfyUI nodes"
- Check that `COMFY_PATH` points to your ComfyUI installation
- Verify ComfyUI is properly installed with `pip install -r requirements.txt`

### "Model not found"
- Verify models are in the correct folders
- Check file names match exactly (case-sensitive)
- Ensure models are fully downloaded (not corrupted)

### "CUDA out of memory"
- Reduce image dimensions (try 768x768 or 512x512)
- Close other GPU applications
- Restart Python to clear memory
- Consider using CPU mode (slower but works)

### "LoRA not loading"
- LoRA is optional, generation will work without it
- Check that `pixel_art_style_z_image_turbo.safetensors` is in `ComfyUI/models/loras/`
- Disable LoRA if not needed

## Performance Tips

- **First generation is slow** - Models are loaded on first use
- **Subsequent generations are fast** - Models stay in memory
- **Use "Clear Memory" button** - If you need to free VRAM
- **Optimal settings**: 1024x1024, 9 steps, shift=3.0
- **Enable xformers** - For faster attention computation (if available)

## System Requirements

### Minimum:
- **GPU**: 12GB VRAM (NVIDIA recommended)
- **RAM**: 16GB
- **Storage**: ~15GB for models

### Recommended:
- **GPU**: 16GB+ VRAM (RTX 3090, 4090, A5000, etc.)
- **RAM**: 32GB
- **Storage**: SSD for faster model loading

### CPU Mode:
- Works but very slow (minutes per image)
- Requires 32GB+ RAM

## License

- **Z-Image-Turbo Model**: Apache 2.0
- **ComfyUI**: GPL-3.0
- **This Interface**: MIT

## Credits

- **Z-Image-Turbo**: Tongyi-MAI (Alibaba)
- **ComfyUI**: comfyanonymous
- **Pixel Art LoRA**: tarn59
- **Workflow**: Based on official Z-Image-Turbo ComfyUI workflow

## Links

- [Z-Image-Turbo Model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Pixel Art LoRA](https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo)
- [Original Workflow](https://comfyworkflows.com/)

---

**Enjoy creating amazing images with Z-Image-Turbo! 🎨✨**
