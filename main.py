from logger import logger
from embedding import ClipEmbeddingProcessor
from data_processor import DataProcessor
from qa_system import QASystem
from config import IMAGE_DIR, CHROMA_DIR
from evaluate import run_eval   # 你自己寫的評估腳本

def build_index():
    embedder = ClipEmbeddingProcessor(
        persist_directory=str(CHROMA_DIR),
        image_dir=str(IMAGE_DIR)
    )
    dp = DataProcessor(embedder)
    # 路徑自行替換
    qa_pairs, imgs = dp.process_csv_with_images("post_response_filtered.xlsx")
    logger.info(f"Indexed {len(qa_pairs)} QA pairs, {len(imgs)} images.")
    return embedder

def chat(embedder):
    qa = QASystem(embedder)
    while True:
        q = input("🐶> ")
        if q.lower() in {"exit","quit"}: break
        print(qa.ask(q))         # 你原本的生成函式

if __name__ == "__main__":
    emb = build_index()
    # chat(emb)
    run_eval(emb,"test_questions.xlsx")
