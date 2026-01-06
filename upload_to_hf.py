# TOKEN = "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"
import os
from huggingface_hub import HfApi, login

# 1. 配置信息 (确保你的仓库名已经更新)
TOKEN = "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"
REPO_ID = "leewheel/Z-Image-Turbo-For-Pinokio"

def sync_force_folders():
    login(token=TOKEN)
    api = HfApi()
    
    # 手动列出你红圈里最关心的文件夹，确保它们被处理
    # 如果后续有新文件夹，脚本也会自动扫描
    important_dirs = ["env", "logs", "lora", "MOD", "outputs"]
    
    # 获取当前目录下所有的项目
    items = os.listdir(".")
    
    print(f"🚀 开始强力同步至: {REPO_ID}")
    print("-" * 60)

    for item in items:
        # 排除项
        if item in ["upload_to_hf.py", ".git", "cache"] or item.startswith("."):
            continue
            
        print(f"🔍 正在处理: {item} ...")
        
        try:
            # 不再判断 isdir，直接用 upload_folder。
            # 对于文件夹，它会同步内容；对于文件，它也能正常处理。
            api.upload_folder(
                folder_path=item if os.path.isdir(item) else ".", # 如果是文件，我们换个思路
                path_in_repo=item if os.path.isdir(item) else "", 
                repo_id=REPO_ID,
                repo_type="model",
                allow_patterns=[f"{item}/*"] if os.path.isdir(item) else [item],
                ignore_patterns=["**/__pycache__/*", "**/.cache/*"]
            )
            print(f"✅ {item} 处理完成。")
        except Exception as e:
            # 如果 upload_folder 失败，尝试用 upload_file 兜底
            try:
                if os.path.isfile(item):
                    api.upload_file(path_or_fileobj=item, path_in_repo=item, repo_id=REPO_ID)
                    print(f"📄 文件同步成功: {item}")
                else:
                    print(f"⚠️ 文件夹 {item} 同步遇到挑战: {e}")
            except:
                print(f"❌ 无法同步 {item}")

    print("-" * 60)
    print(f"🎉 任务强制执行结束！")

if __name__ == "__main__":
    sync_force_folders()
    input("\n请现在刷新 Hugging Face 页面，检查文件夹是否出现。按回车退出...")