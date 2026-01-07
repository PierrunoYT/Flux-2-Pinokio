# FLUX.2-dev Gradio UI Configuration
# Copy this file to config.py and modify as needed

# Model Configuration
MODEL_CONFIG = {
    "repo_id": "diffusers/FLUX.2-dev-bnb-4bit",  # 4-bit quantized model
    "torch_dtype": "bfloat16",  # or "float16"
    "device": "cuda:0",  # or "cuda:1" for multi-GPU systems
    "use_remote_encoder": True,  # Set False to use local encoder (uses more VRAM)
}

# Default Generation Settings
DEFAULT_TEXT_TO_IMAGE = {
    "steps": 28,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024,
    "seed": -1,  # -1 for random
    "num_images": 1,
}

DEFAULT_IMAGE_TO_IMAGE = {
    "steps": 28,
    "guidance_scale": 3.5,
    "strength": 0.75,
    "width": 1024,
    "height": 1024,
    "seed": -1,
}

# UI Configuration
UI_CONFIG = {
    "server_name": "0.0.0.0",  # "127.0.0.1" for local only
    "server_port": 7860,
    "share": False,  # Set True to create public link
    "show_error": True,
    "theme": "soft",  # gradio theme
}

# Performance Options
PERFORMANCE = {
    "enable_xformers": True,  # Requires xformers package
    "enable_cpu_offload": False,  # Set True for lower VRAM (slower)
    "use_flash_attention": False,  # Experimental
}

# Advanced Options
ADVANCED = {
    "max_resolution": 2048,  # Maximum allowed resolution
    "max_batch_size": 4,  # Maximum number of images per generation
    "cache_dir": None,  # Custom cache directory (None for default)
}

# Preset Prompts (shown as examples)
PROMPT_PRESETS = {
    "Photorealistic": "Professional photograph, highly detailed, 8k quality, perfect lighting",
    "Digital Art": "Digital art, trending on artstation, highly detailed, vibrant colors",
    "Oil Painting": "Oil painting style, classical art, museum quality, detailed brushstrokes",
    "Anime": "Anime style, studio quality, detailed character, beautiful composition",
    "Cinematic": "Cinematic lighting, dramatic atmosphere, movie still, high production value",
}

# Negative Prompt Presets
NEGATIVE_PRESETS = {
    "Default": "blurry, low quality, distorted, ugly, bad anatomy",
    "Photography": "blurry, low quality, overexposed, grainy, amateur",
    "Art": "low quality, sketch, unfinished, messy, chaotic",
}

# Resolution Presets (width, height)
RESOLUTION_PRESETS = {
    "Square - 512px": (512, 512),
    "Square - 768px": (768, 768),
    "Square - 1024px": (1024, 1024),
    "Square - 1536px": (1536, 1536),
    "Portrait - 768x1024": (768, 1024),
    "Portrait - 1024x1536": (1024, 1536),
    "Landscape - 1024x768": (1024, 768),
    "Landscape - 1536x1024": (1536, 1024),
    "Widescreen - 1920x1080": (1920, 1080),
    "Ultrawide - 2560x1080": (2560, 1080),
}