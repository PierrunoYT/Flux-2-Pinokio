# Quick Start Guide - FLUX.2-dev Gradio UI (Pinokio App)

## 🚀 Get Started in 5 Minutes

### Step 1: Prerequisites Check

Before you begin, make sure you have:
- ✅ NVIDIA GPU with 24GB VRAM (RTX 4090, 5090, A5000, etc.)
- ✅ Python 3.8 or higher installed
- ✅ CUDA 11.8+ installed
- ✅ ~50GB free disk space

**Check your GPU:**
```bash
nvidia-smi
```

### Step 2: HuggingFace Setup

1. **Create account**: Go to https://huggingface.co and sign up

2. **Accept model license**: Visit https://huggingface.co/black-forest-labs/FLUX.2-dev and click "Agree"

3. **Create access token**: 
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "flux2-ui")
   - Select "Read" permissions
   - Copy the token

4. **Login via CLI**:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   # Paste your token when prompted
   ```

### Step 3: Installation

**Option A: Pinokio Installation (Recommended)**

1. **Install Pinokio** (if needed):
   - Download from: https://pinokio.computer
   - Install following the instructions for your OS

2. **Add this app to Pinokio**:
   - Open Pinokio
   - Click "Add" or "+" button
   - Select "From Folder" and navigate to this repository

3. **Click "Install"** in Pinokio:
   - This will create a virtual environment and install all dependencies
   - Wait for installation to complete

**Option B: Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 4: Launch

**Pinokio Method:**
- Click the **"Start"** button in Pinokio
- The UI will open automatically at **http://127.0.0.1:7860**

**Manual Method:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```
Then open **http://localhost:7860** in your browser

### Step 5: First Generation

1. **Click "Initialize Pipeline"** in the UI (takes 2-5 min first time)
2. Wait for "✅ Pipeline initialized successfully!"
3. Go to **"Text-to-Image"** tab
4. Enter a prompt, e.g.:
   ```
   A beautiful sunset over mountains, golden hour, photorealistic, 8k
   ```
5. Click **"Generate Images"**
6. Wait ~15-20 seconds for your first image!

---

## 🎯 Common First-Time Issues

### Issue: "CUDA not available"
**Fix**: Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "You need to accept the license"
**Fix**: Visit https://huggingface.co/black-forest-labs/FLUX.2-dev and click "Agree"

### Issue: "Authentication required"
**Fix**: Run `huggingface-cli login` and paste your token

### Issue: "CUDA out of memory"
**Fix**: Reduce resolution to 768x768 or 512x512

---

## 📝 Example Prompts to Try

### Photorealistic
```
Professional photograph of a vintage car on a desert highway at sunset, 
golden hour lighting, cinematic, 8k, highly detailed
```

### Digital Art
```
Cyberpunk city at night with neon signs, flying cars, rain-soaked streets, 
digital art, highly detailed, vibrant colors
```

### Portrait
```
Portrait of a wise elderly woman with silver hair, soft natural lighting, 
professional photography, 85mm lens, shallow depth of field
```

### Landscape
```
Misty mountain landscape at dawn, pine trees, lake reflection, 
nature photography, atmospheric, 8k quality
```

### Fantasy
```
Epic fantasy castle on a cliff overlooking the ocean, magical atmosphere, 
dramatic clouds, fantasy art, highly detailed
```

---

## ⚙️ Recommended First Settings

| Setting | Value | Why |
|---------|-------|-----|
| Resolution | 1024x1024 | Safe for 24GB VRAM |
| Steps | 28 | Good balance |
| Guidance | 3.5 | Works for most prompts |
| Seed | -1 | Random each time |

---

## 🎓 Learning Path

1. **Day 1**: Try the example prompts above
2. **Day 2**: Experiment with different styles (add "oil painting", "watercolor", etc.)
3. **Day 3**: Try Image-to-Image with your own photos
4. **Day 4**: Learn to adjust Guidance Scale and Steps
5. **Day 5**: Master prompt engineering with complex descriptions

---

## 📊 Performance Expectations

On a RTX 4090 with 24GB VRAM:

| Resolution | Steps | Time per Image |
|------------|-------|----------------|
| 512x512 | 28 | ~5 seconds |
| 768x768 | 28 | ~10 seconds |
| 1024x1024 | 28 | ~15 seconds |
| 1536x1536 | 28 | ~30 seconds |

*First generation adds 30-60 seconds for model loading*

---

## 🆘 Need Help?

1. **Read the full README.md** for detailed information
2. **Check the Troubleshooting section** in README.md
3. **Monitor GPU**: Run `watch -n 1 nvidia-smi` in another terminal (Linux/Mac) or use Task Manager on Windows
4. **Check logs**: 
   - **Pinokio**: Check the terminal output in Pinokio's interface
   - **Manual**: Look at terminal output for error messages
5. **Pinokio Issues**: Make sure you've logged into HuggingFace CLI before starting

---

## 🎉 Next Steps

Once you're comfortable with the basics:

- **Explore Advanced Settings**: Try different resolutions and step counts
- **Master Prompt Engineering**: Read prompt tips in README.md
- **Try Image-to-Image**: Upload your own photos and transform them
- **Experiment with Styles**: Add style keywords to your prompts
- **Share Your Results**: Show off what you create!

---

**Happy Creating! 🎨**

*Remember: The first generation is always slower. Be patient!*

---

## 📌 Pinokio-Specific Notes

- **Installation**: The Pinokio app handles virtual environment creation automatically
- **Updates**: Use the "Update" button in Pinokio to update dependencies
- **Reset**: Use the "Reset" button to revert to pre-install state if needed
- **Terminal Access**: Click "Terminal" in Pinokio to access the virtual environment shell
- **Auto-Launch**: Pinokio will automatically open the UI when you click "Start"