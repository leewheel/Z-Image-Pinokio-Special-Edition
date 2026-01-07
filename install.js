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
    // Using && dir to ensure clean prompt termination
    {
      method: "shell.run",
            params: {
              venv: "env",  // 这里保持 env，这样 huggingface-cli 命令才能被找到
              message: [
                "smart_download.bat"
              ],
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Z-Image-Turbo LowVram Edition."
      }
    }
  ]
}
