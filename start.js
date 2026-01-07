module.exports = {
  daemon: true,
  run: [
    // Launch FLUX.2-dev Gradio UI
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        message: [
          "python app.py"
        ],
        on: [{
          // Monitor for HTTP server startup
          "event": "/http:\\/\\/[^\\s\\/]+:\\d{2,5}(?=[^\\w]|$)/",
          "done": true
        }]
      }
    },
    // Set the local URL variable for the UI button
    {
      method: "local.set",
      params: {
        url: "http://127.0.0.1:7860"
      }
    },
    {
      method: "notify",
      params: {
        html: "FLUX.2-dev is running! Click the button to open the Image Generator."
      }
    }
  ]
}

