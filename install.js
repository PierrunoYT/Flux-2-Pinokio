module.exports = {
  run: [
    // Create virtual environment
    {
      method: "shell.run",
      params: {
        message: [
          "python -m venv env"
        ],
      }
    },
    // Install dependencies for FLUX.2-dev
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt"
        ],
      }
    },
    // Install PyTorch with CUDA support (platform-specific)
    // torch.js handles platform detection and installs appropriate PyTorch version
    // xformers enabled for memory-efficient attention
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true  // Enable xformers for memory-efficient attention
        }
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch FLUX.2-dev Image Generator."
      }
    }
  ]
}
