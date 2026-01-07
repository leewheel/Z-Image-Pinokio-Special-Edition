@echo off
echo Checking network connectivity to HuggingFace...

REM 尝试连接 huggingface.co，超时设为 3 秒
curl -I https://huggingface.co --connect-timeout 3 >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] HuggingFace is reachable. Using Official Source.
    huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir-use-symlinks False
) else (
    echo [FAIL] HuggingFace is unreachable. Switching to Mirror (hf-mirror.com).
    huggingface-cli download --endpoint-url https://hf-mirror.com Tongyi-MAI/Z-Image-Turbo --local-dir-use-symlinks False
)

echo.
echo Done.
dir
