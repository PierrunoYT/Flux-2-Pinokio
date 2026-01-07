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
    // Install PyTorch with CUDA support (platform-specific)
    // torch.js handles platform detection and installs appropriate PyTorch version
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
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
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch FLUX.2-dev Image Generator."
      }
    }
  ]
}
