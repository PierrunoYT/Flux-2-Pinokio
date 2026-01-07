# FLUX.2-dev-NVFP4 Gradio UI (Pinokio App)

A user-friendly Gradio interface for running FLUX.2-dev with 4-bit quantization (NVFP4), optimized for 24GB VRAM GPUs. This is a Pinokio app that simplifies installation and management.

![FLUX.2-dev](https://img.shields.io/badge/FLUX.2-dev-blue)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange)
![VRAM](https://img.shields.io/badge/VRAM-24GB-green)
![Pinokio](https://img.shields.io/badge/Pinokio-App-purple)

## Features

- 🎨 **Text-to-Image Generation**: Create high-quality images from text descriptions
- 🖼️ **Image-to-Image Editing**: Transform existing images with text prompts
- 💾 **Memory Efficient**: Uses 4-bit quantization + remote text encoder
- ⚡ **Fast**: ~3x faster than BF16 with minimal quality loss
- 🎯 **24GB VRAM Optimized**: Perfect for RTX 4090, RTX 5090, or similar GPUs

## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 24GB VRAM (RTX 4090, RTX 5090, A5000, etc.)
- **RAM**: 32GB system RAM recommended
- **Storage**: ~50GB free space for models
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11 with WSL2, or macOS with CUDA support

### Software Requirements
- Python 3.8 or higher
- CUDA 11.8 or higher
- PyTorch 2.1.0 or higher with CUDA support

## Installation

### Pinokio Installation (Recommended)

This is a Pinokio app - the easiest way to install and run:

1. **Install Pinokio** (if not already installed):
   - Download from: https://pinokio.computer
   - Follow the installation instructions for your OS

2. **Add this app to Pinokio**:
   - Open Pinokio
   - Click "Add" or use the "+" button
   - Select "From Folder" or "From URL" if hosted
   - Navigate to this repository folder

3. **Install the app**:
   - Click the **"Install"** button in Pinokio
   - Wait for installation to complete (creates virtual environment and installs dependencies)

4. **Login to HuggingFace** (required before first use):
   ```bash
   # Open terminal in Pinokio or your system terminal
   huggingface-cli login
   ```
   - You'll need a HuggingFace account
   - Create an access token at: https://huggingface.co/settings/tokens
   - Accept the FLUX.2-dev license at: https://huggingface.co/black-forest-labs/FLUX.2-dev

5. **Start the app**:
   - Click the **"Start"** button in Pinokio
   - The UI will open automatically at `http://127.0.0.1:7860`

### Manual Installation (Without Pinokio)

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```
   - You'll need a HuggingFace account
   - Create an access token at: https://huggingface.co/settings/tokens
   - Accept the FLUX.2-dev license at: https://huggingface.co/black-forest-labs/FLUX.2-dev

4. **Run the UI**:
   ```bash
   python app.py
   ```

## Usage

### First Time Setup

1. **Launch the application**:
   - **Pinokio**: Click "Start" - the interface will open automatically at `http://127.0.0.1:7860`
   - **Manual**: Run `python app.py` - open `http://localhost:7860` in your browser

2. **Click "Initialize Pipeline"** - This loads the model (takes 2-5 minutes first time)

3. **Start generating!** - Once initialized, you can generate images

### Text-to-Image Generation

1. Go to the **"Text-to-Image"** tab
2. Enter your prompt (e.g., "A futuristic cityscape at night with neon lights")
3. Adjust settings:
   - **Steps**: 28 (recommended) - 50 (high quality)
   - **Guidance Scale**: 3.5-4.0 for most prompts
   - **Resolution**: 1024x1024 (safe for 24GB VRAM)
4. Click **"Generate Images"**

### Image-to-Image Editing

1. Go to the **"Image-to-Image"** tab
2. Upload an input image
3. Enter transformation prompt (e.g., "Transform into a watercolor painting")
4. Adjust **Strength**: 
   - 0.3-0.5: Minor changes
   - 0.6-0.8: Moderate transformation
   - 0.9-1.0: Major changes
5. Click **"Transform Image"**

## Configuration

### Recommended Settings

| Use Case | Steps | Guidance | Resolution | Expected Time |
|----------|-------|----------|------------|---------------|
| Quick Preview | 20 | 3.0 | 512x512 | ~5 seconds |
| Standard Quality | 28 | 3.5 | 1024x1024 | ~15 seconds |
| High Quality | 40 | 4.0 | 1024x1024 | ~25 seconds |
| Maximum Quality | 50 | 4.0 | 1536x1536 | ~45 seconds |

### VRAM Usage

- **1024x1024**: ~15-18GB VRAM
- **1536x1536**: ~20-22GB VRAM
- **2048x2048**: May exceed 24GB (use with caution)

### Performance Tips

1. **Start Small**: Test with 512x512 or 1024x1024 before going higher
2. **Monitor VRAM**: Use `nvidia-smi` to check GPU memory usage
3. **Batch Size**: Keep num_images=1 for 1024x1024 on 24GB VRAM
4. **Close Other Apps**: Free up VRAM by closing other GPU-intensive applications

## Troubleshooting

### "CUDA out of memory" Error
- **Solution**: Reduce resolution, reduce number of images, or restart the UI
- Try 768x768 or 512x512 instead of 1024x1024

### Model Loading Fails
- **Check**: Have you accepted the FLUX.2-dev license?
- **Check**: Are you logged into HuggingFace CLI?
- **Run**: `huggingface-cli whoami` to verify login

### Remote Text Encoder Fails
- **Check**: Internet connection (remote encoder needs internet)
- **Alternative**: Modify code to use local text encoder (uses more VRAM)

### Slow Generation
- **Normal**: First generation is slower (model loading)
- **Expected**: ~15-30 seconds per image at 1024x1024 with 28 steps
- **Check**: Are other processes using GPU?

### Poor Image Quality
- **Increase Steps**: Try 40-50 instead of 28
- **Adjust Guidance**: Try 3.5-5.0 range
- **Better Prompts**: Be more specific and descriptive
- **Check Settings**: Ensure you're not using too low resolution

## Prompt Engineering Tips

### Good Prompts
- ✅ "A professional photograph of a red sports car on a mountain road at sunset, golden hour lighting, 8k, highly detailed"
- ✅ "Oil painting portrait of an elderly wizard with a long white beard, fantasy art style, dramatic lighting, intricate details"
- ✅ "Minimalist modern kitchen interior with marble countertops, natural lighting, architectural photography"

### Tips
1. **Be Specific**: Include details about style, lighting, mood, quality
2. **Use Style Keywords**: "photorealistic", "oil painting", "watercolor", "digital art"
3. **Add Quality Tags**: "highly detailed", "8k", "professional photography"
4. **Describe Lighting**: "golden hour", "dramatic lighting", "soft natural light"
5. **Camera Angles**: "aerial view", "close-up", "wide angle", "macro"

## Advanced Configuration

### Using Local Text Encoder

If you have extra VRAM and want to avoid the remote encoder dependency:

```python
# In app.py, modify the initialize_pipeline function:
from transformers import Mistral3ForConditionalGeneration

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    REPO_ID,
    subfolder="text_encoder",
    torch_dtype=TORCH_DTYPE,
    device_map="cpu"  # Or "auto"
)

pipe = Flux2Pipeline.from_pretrained(
    REPO_ID,
    text_encoder=text_encoder,  # Use local instead of None
    torch_dtype=TORCH_DTYPE
)
```

### Enabling Model CPU Offloading

For lower VRAM GPUs (16GB), you can enable CPU offloading:

```python
pipe.enable_model_cpu_offload()
```

This will use more system RAM but reduce VRAM usage.

## Technical Details

### Model Information
- **Model**: FLUX.2-dev
- **Parameters**: 32 billion
- **Quantization**: 4-bit (BitsAndBytes NF4)
- **Architecture**: Rectified Flow Transformer
- **Text Encoder**: Mistral 3 Small (remote offloaded)
- **VAE**: BF16 (not quantized)

### Quantization Benefits
- **VRAM Savings**: ~4x reduction vs BF16
- **Speed**: ~3x faster inference
- **Quality Loss**: Minimal (nearly imperceptible)
- **Support**: Optimized for RTX 40/50 series

### Remote Text Encoder
- **Purpose**: Saves ~10GB VRAM
- **Hosted**: HuggingFace servers
- **Latency**: +1-2 seconds per generation
- **Alternative**: Use local encoder if you have extra VRAM

## License

This UI code is provided as-is for use with FLUX.2-dev.

**FLUX.2-dev Model License**: Non-commercial use only. See [license](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt)

## Credits

- **FLUX.2**: [Black Forest Labs](https://blackforestlabs.ai/)
- **Diffusers**: [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- **Gradio**: [Gradio Team](https://gradio.app/)

## Changelog

### Version 1.0.0
- Initial release
- Text-to-Image generation
- Image-to-Image editing
- 4-bit quantization support
- Remote text encoder
- Optimized for 24GB VRAM

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Visit [FLUX.2-dev on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-dev)
3. Check [Diffusers documentation](https://huggingface.co/docs/diffusers)

## Future Enhancements

- [ ] LoRA support
- [ ] Multi-reference editing
- [ ] Batch processing
- [ ] Custom aspect ratios
- [ ] Advanced controlnet support
- [ ] Image upscaling integration
- [ ] Save/load presets

---

**Happy Generating! 🎨**