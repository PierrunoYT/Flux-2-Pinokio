#!/usr/bin/env python3
"""
FLUX.2-dev Gradio UI
Supports both text-to-image and image-to-image generation with NVFP4/4-bit quantization
Optimized for 24GB VRAM GPUs
"""

import os
import torch
import gradio as gr
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from huggingface_hub import get_token, login
import requests
import io
import time
import pickle

# Configuration
REPO_ID = "diffusers/FLUX.2-dev-bnb-4bit"  # 4-bit quantized model
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16

# Global pipeline variable
pipe = None
hf_token = None  # Store HuggingFace token

def remote_text_encoder(prompts, token=None, max_retries=3):
    """
    Use remote text encoder to save VRAM.
    This offloads the large text encoder to HuggingFace's servers.
    """
    # Use provided token, or try to get from environment
    auth_token = token or hf_token or get_token()
    if not auth_token:
        raise Exception("No HuggingFace token provided. Please enter your token in the UI.")
    
    prompt_list = prompts if isinstance(prompts, list) else [prompts]
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://remote-text-encoder-flux-2.huggingface.co/predict",
                json={"prompt": prompt_list},
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json"
                },
                timeout=60,  # Increased timeout
                stream=False  # Don't stream to ensure complete response
            )
            
            # Check response status
            response.raise_for_status()
            
            # Verify response content is not empty
            if not response.content:
                raise Exception("Empty response from remote text encoder")
            
            # Check content length header if available
            content_length = response.headers.get('Content-Length')
            if content_length and len(response.content) < int(content_length):
                raise Exception(f"Response incomplete: received {len(response.content)} bytes, expected {content_length}")
            
            # Try to load the pickle data
            try:
                prompt_embeds = torch.load(io.BytesIO(response.content), map_location=DEVICE)
                # Verify the loaded data is valid
                if prompt_embeds is None:
                    raise Exception("Failed to load prompt embeddings from response")
                return prompt_embeds.to(DEVICE)
            except (pickle.UnpicklingError, EOFError, RuntimeError) as pickle_error:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Pickle loading failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Failed to load pickle data after {max_retries} attempts: {str(pickle_error)}\n"
                                  f"Response size: {len(response.content)} bytes\n"
                                  f"This might be a network issue. Try again or check your internet connection.")
            
        except requests.exceptions.RequestException as req_error:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Remote text encoder request failed after {max_retries} attempts: {str(req_error)}\n"
                              f"Check your internet connection and try again.")
        except Exception as e:
            # For other exceptions, don't retry
            raise Exception(f"Remote text encoder error: {str(e)}\nMake sure you've entered a valid HuggingFace token.")
    
    raise Exception(f"Remote text encoder failed after {max_retries} attempts")

def initialize_pipeline(hf_token_input):
    """Initialize the FLUX.2 pipeline with optimizations for 24GB VRAM"""
    global pipe, hf_token
    
    if pipe is not None:
        return "Pipeline already initialized!"
    
    # Validate token
    if not hf_token_input or not hf_token_input.strip():
        return "❌ Please enter your HuggingFace token first!\n\nGet your token from: https://huggingface.co/settings/tokens"
    
    try:
        # Store token globally
        hf_token = hf_token_input.strip()
        
        # Set token in environment for huggingface_hub
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
        # Login with the token
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Successfully authenticated with HuggingFace")
        except Exception as login_error:
            print(f"Warning: Could not login via API: {login_error}")
            # Continue anyway, token might still work
        
        print("Loading FLUX.2-dev-bnb-4bit model...")
        print("This may take a few minutes on first run...")
        
        # Load pipeline with remote text encoder to save VRAM
        pipe = Flux2Pipeline.from_pretrained(
            REPO_ID,
            text_encoder=None,  # Use remote text encoder
            torch_dtype=TORCH_DTYPE,
            token=hf_token
        ).to(DEVICE)
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("XFormers memory efficient attention enabled")
        except:
            print("XFormers not available, using default attention")
        
        return "✅ Pipeline initialized successfully! Ready to generate images."
    
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
            return f"❌ Authentication failed: {error_msg}\n\nPlease check:\n1. Your token is correct\n2. You have accepted the license at https://huggingface.co/black-forest-labs/FLUX.2-dev"
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            return f"❌ Access denied: {error_msg}\n\nPlease make sure you have accepted the model license at https://huggingface.co/black-forest-labs/FLUX.2-dev"
        else:
            return f"❌ Error initializing pipeline: {error_msg}\n\nMake sure you:\n1. Have accepted the model license at https://huggingface.co/black-forest-labs/FLUX.2-dev\n2. Entered a valid HuggingFace token"

def text_to_image(
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    width,
    height,
    seed,
    num_images
):
    """Generate images from text prompt"""
    global pipe
    
    if pipe is None:
        return None, "⚠️ Please initialize the pipeline first!"
    
    try:
        # Get prompt embeddings from remote encoder
        prompt_list = [prompt] * num_images
        prompt_embeds = remote_text_encoder(prompt_list, token=hf_token)
        
        # Set random seed for reproducibility
        generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed >= 0 else None
        
        # Generate images
        output = pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=1
        )
        
        images = output.images
        
        return images, f"✅ Generated {len(images)} image(s) successfully!"
    
    except Exception as e:
        return None, f"❌ Error during generation: {str(e)}"

def image_to_image(
    prompt,
    input_image,
    strength,
    num_inference_steps,
    guidance_scale,
    width,
    height,
    seed
):
    """Generate images from input image and prompt"""
    global pipe
    
    if pipe is None:
        return None, "⚠️ Please initialize the pipeline first!"
    
    if input_image is None:
        return None, "⚠️ Please upload an input image!"
    
    try:
        # Get prompt embeddings
        prompt_embeds = remote_text_encoder([prompt], token=hf_token)
        
        # Prepare input image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Resize image to target dimensions
        input_image = input_image.resize((width, height), Image.LANCZOS)
        
        # Set random seed
        generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed >= 0 else None
        
        # Generate image
        output = pipe(
            prompt_embeds=prompt_embeds,
            image=input_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return output.images[0], "✅ Image generated successfully!"
    
    except Exception as e:
        return None, f"❌ Error during generation: {str(e)}"

def get_system_info():
    """Get system information"""
    if not torch.cuda.is_available():
        return "⚠️ CUDA not available! Running on CPU (very slow)"
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    
    return f"""
🖥️ **System Information:**
- GPU: {gpu_name}
- Total VRAM: {gpu_memory:.2f} GB
- Currently Allocated: {allocated:.2f} GB
- PyTorch Version: {torch.__version__}
- CUDA Available: ✅
"""

# Create Gradio Interface
with gr.Blocks(title="FLUX.2-dev-NVFP4 Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 FLUX.2-dev Image Generator (NVFP4/4-bit)
    
    High-quality image generation optimized for 24GB VRAM using 4-bit quantization.
    
    **Features:**
    - Text-to-Image generation
    - Image-to-Image editing
    - Multi-reference support
    - Remote text encoder (saves VRAM)
    
    **Note:** You must accept the license at [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-dev) and provide your access token below.
    """)
    
    # System info
    with gr.Row():
        system_info = gr.Markdown(get_system_info())
    
    # HuggingFace Token Input
    with gr.Row():
        with gr.Column():
            hf_token_input = gr.Textbox(
                label="🔑 HuggingFace Access Token",
                placeholder="Enter your HuggingFace token here (hf_...)",
                type="password",
                info="Get your token from: https://huggingface.co/settings/tokens",
                value=""
            )
            gr.Markdown("""
            **How to get your token:**
            1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
            2. Click "New token"
            3. Name it (e.g., "flux2-ui")
            4. Select "Read" permissions
            5. Copy the token and paste it above
            6. Make sure you've accepted the [FLUX.2-dev license](https://huggingface.co/black-forest-labs/FLUX.2-dev)
            """)
    
    # Initialize button
    with gr.Row():
        init_button = gr.Button("🚀 Initialize Pipeline (Click After Entering Token!)", variant="primary", size="lg")
        init_status = gr.Textbox(label="Status", interactive=False, lines=5)
    
    init_button.click(fn=initialize_pipeline, inputs=[hf_token_input], outputs=init_status)
    
    # Main interface
    with gr.Tabs():
        # Text-to-Image Tab
        with gr.Tab("📝 Text-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A beautiful sunset over mountains...",
                        lines=3,
                        value="A futuristic cityscape at night with neon lights, cyberpunk style, highly detailed, 8k"
                    )
                    
                    t2i_negative = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="blurry, low quality...",
                        lines=2
                    )
                    
                    with gr.Row():
                        t2i_width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                        t2i_height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                    
                    with gr.Row():
                        t2i_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28,
                            info="28 is recommended for quality/speed balance"
                        )
                        t2i_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=3.5,
                            info="Higher = follows prompt more closely"
                        )
                    
                    with gr.Row():
                        t2i_seed = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                        t2i_num_images = gr.Slider(
                            label="Number of Images",
                            minimum=1,
                            maximum=4,
                            step=1,
                            value=1
                        )
                    
                    t2i_generate = gr.Button("🎨 Generate Images", variant="primary")
                
                with gr.Column(scale=1):
                    t2i_output = gr.Gallery(
                        label="Generated Images",
                        show_label=True,
                        columns=2,
                        height=600,
                        object_fit="contain"
                    )
                    t2i_status = gr.Textbox(label="Status", interactive=False)
            
            t2i_generate.click(
                fn=text_to_image,
                inputs=[
                    t2i_prompt,
                    t2i_negative,
                    t2i_steps,
                    t2i_guidance,
                    t2i_width,
                    t2i_height,
                    t2i_seed,
                    t2i_num_images
                ],
                outputs=[t2i_output, t2i_status]
            )
        
        # Image-to-Image Tab
        with gr.Tab("🖼️ Image-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=400
                    )
                    
                    i2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Transform the image to...",
                        lines=3,
                        value="Transform this into a watercolor painting style"
                    )
                    
                    i2i_strength = gr.Slider(
                        label="Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.75,
                        info="Higher = more changes to original"
                    )
                    
                    with gr.Row():
                        i2i_width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                        i2i_height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                    
                    with gr.Row():
                        i2i_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28
                        )
                        i2i_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=3.5
                        )
                    
                    i2i_seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                    
                    i2i_generate = gr.Button("🎨 Transform Image", variant="primary")
                
                with gr.Column(scale=1):
                    i2i_output = gr.Image(
                        label="Generated Image",
                        height=600
                    )
                    i2i_status = gr.Textbox(label="Status", interactive=False)
            
            i2i_generate.click(
                fn=image_to_image,
                inputs=[
                    i2i_prompt,
                    i2i_input,
                    i2i_strength,
                    i2i_steps,
                    i2i_guidance,
                    i2i_width,
                    i2i_height,
                    i2i_seed
                ],
                outputs=[i2i_output, i2i_status]
            )
    
    # Tips and Info
    with gr.Accordion("💡 Tips & Tricks", open=False):
        gr.Markdown("""
        ### Recommended Settings:
        - **Steps**: 28 for good quality/speed balance, 40-50 for maximum quality
        - **Guidance Scale**: 3.5-4.0 works well for most prompts
        - **Resolution**: Start with 1024x1024, increase if needed
        - **Seed**: Use -1 for random, or set a specific number for reproducible results
        
        ### VRAM Usage:
        - Text-to-Image: ~15-18GB at 1024x1024
        - Image-to-Image: ~16-20GB at 1024x1024
        - Higher resolutions will use more VRAM
        
        ### Performance:
        - First generation will be slower (loading models)
        - With 24GB VRAM, you can comfortably run 1024x1024
        - 4-bit quantization provides 3x speedup vs BF16
        - Remote text encoder saves ~10GB VRAM
        
        ### Prompt Tips:
        - Be specific and descriptive
        - Include style keywords (e.g., "photorealistic", "oil painting", "8k")
        - Mention lighting, camera angle, mood
        - For best results, describe what you WANT rather than what you DON'T want
        """)

if __name__ == "__main__":
    print("Starting FLUX.2-dev Gradio UI...")
    print(get_system_info())
    print("\nMake sure you have:")
    print("1. Accepted the license at: https://huggingface.co/black-forest-labs/FLUX.2-dev")
    print("2. Created an access token at: https://huggingface.co/settings/tokens")
    print("3. Enter your token in the UI before initializing the pipeline")
    print("\nLaunching interface...")
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )