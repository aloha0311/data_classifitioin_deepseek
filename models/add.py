import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 这个镜像网站可能也可以换掉

from huggingface_hub import snapshot_download

snapshot_download(repo_id="deepseek-ai/deepseek-llm-7b-chat",
                  local_dir_use_symlinks=False,
                  local_dir="./models/deepseek-llm-7b-chat")
