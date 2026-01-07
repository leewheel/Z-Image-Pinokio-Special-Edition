module.exports = {
  run: [
    // Step 1: Install PyTorch (保持不变，这部分是好的)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true   // Enable xformers
        }
      }
    },
    // Step 2: Install Dependencies (保持不变，记得 requirements.txt 里没写 torch)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt"
        ],
      }
    },
    // Step 3: Download Model (核心优化：直接克隆到 ckpts)
    {
      method: "shell.run",
      params: {
        // 这里不需要 venv，因为 git 是系统命令
        // 使用 shell.run 可以直接执行数组里的命令
        message: [
          "git lfs install",
          "git clone https://hf-mirror.com/Tongyi-MAI/Z-Image-Turbo ckpts\\Z-Image-Turbo"
        ]
      }
    },
    // Step 4: Finish
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Z-Image-Turbo LowVram Edition."
      }
    }
  ]
}
