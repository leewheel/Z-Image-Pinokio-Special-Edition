module.exports = {
  run: [
    // Install PyTorch with CUDA support first
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true   // Enable xformers for memory-efficient attention
        }
      }
    },
    // Install Z-Image-Turbo dependencies from requirements.txt
    // This includes diffusers from source with ZImagePipeline support
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt"
        ],
      }
    },
    // Pre-download Z-Image-Turbo model (~12GB)
    // Using smart_download.bat to auto-detect best source
    {
      method: "shell.run",
      params: {
        venv: "env", 
        message: [
          "smart_download.bat"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Z-Image-Turbo LowVram Edition."
      }
    }
  ]
}
