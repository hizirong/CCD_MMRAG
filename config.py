from pathlib import Path

# 路徑
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "image"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 模型
SIGLIP_NAME = "google/siglip-base-patch16-224"
LLAMA_MODEL = "llama3.2-vision"

# 其它
EMBED_DIM = 512          # clip embedding dim
DEVICE = "cpu"          # or "cuda"
PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
