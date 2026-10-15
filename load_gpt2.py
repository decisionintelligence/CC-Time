import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download, login

snapshot_download("openai-community/gpt2",local_dir="./gpt2")
