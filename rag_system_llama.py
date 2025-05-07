#!/usr/bin/env python
# coding: utf-8

# #### 轉py

# In[307]:


get_ipython().system('jupyter nbconvert --to script rag_system_llama.ipynb')


# ### Code

# ##### import

# In[161]:


import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import re

# Display and Image handling
from IPython.display import display
from PIL import Image as PILImage  # 使用 PILImage 作为 PIL.Image 的别名
from IPython.display import Image as IPyImage  # 使用 IPyImage 作为 IPython 的 Image

# Vector DB
import chromadb


# LLM
import ollama

# PDF处理
import PyPDF2

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查并创建必要的目录
Path('chroma_db').mkdir(exist_ok=True)
Path('image').mkdir(exist_ok=True)


# In[22]:


import sys
import torch
import transformers
import accelerate
# print(f"Python version: {sys.version}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"Transformers version: {transformers.__version__}")
# print(f"Accelerate version: {accelerate.__version__}")


# In[162]:


# 放在檔案最上方 (import 之後)
TYPE_MAP = {
    "acupoint"    : ["針灸", "acupuncture"],
    "herb"        : ["herbology", "herbal", "方劑"],
    "ccd"         : ["ccd", "認知", "cognition"],
    "social"      : [],                   # csv 直接指定
    "professional": [],
    "image":[]                                       # 其他未分類
}


# ##### voice to text
# 

# In[ ]:


# whisper /Users/zirong/Desktop/test.mp4 --language Chinese --model tiny
import whisper
def transcribe_file(file_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    return result["text"]

# def main():
#     audio_file = "no_upload/test_mp3/01.mp3"  # 修改為你的音檔路徑
#     transcription = transcribe_file(audio_file)
#     print("Transcription:", transcription)

# if __name__ == "__main__":
#     main()


# #### ImageProcessor

# In[163]:


from typing import Union  # 添加 Union 导入
from pathlib import Path

class ImageProcessor:
    def __init__(self, image_dir: str = "image"):
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(exist_ok=True)
        
    def process_and_save(
        self,
        image_path: Union[str, Path],  # 使用 Union 替代 |
        target_size: Tuple[int, int],
        prefix: str = "resized_",
        quality: int = 95
    ) -> Optional[Path]:
        """统一的图片处理方法，处理并保存图片"""
        try:
            # 确保 image_path 是 Path 对象
            image_path = Path(image_path)
            if not str(image_path).startswith(str(self.image_dir)):
                image_path = self.image_dir / image_path
                
            # 检查图片是否存在
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None
                
            # 读取并处理图片
            image = PILImage.open(image_path)
            
            # 转换为 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # 计算等比例缩放的大小
            width, height = image.size
            ratio = min(target_size[0]/width, target_size[1]/height)
            new_size = (int(width * ratio), int(height * ratio))
            
            # 缩放图片
            image = image.resize(new_size, PILImage.Resampling.LANCZOS)
            
            # 创建新的白色背景图片
            new_image = PILImage.new('RGB', target_size, (255, 255, 255))
            
            # 计算居中位置
            x = (target_size[0] - new_size[0]) // 2
            y = (target_size[1] - new_size[1]) // 2
            
            # 贴上缩放后的图片
            new_image.paste(image, (x, y))
            
            # 生成输出路径
            output_path = self.image_dir / f"{image_path.name}" #output_path = self.image_dir / f"{prefix}{image_path.name}"
            # 保存处理后的图片
            new_image.save(output_path, quality=quality)
            logger.info(f"Saved processed image to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def load_for_display(self, 
                        image_path: Union[str, Path],  # 使用 Union 替代 |
                        display_size: Tuple[int, int]) -> Optional[PILImage.Image]:
        """载入图片用于显示"""
        try:
            processed_path = self.process_and_save(image_path, display_size, prefix="display_")
            if processed_path:
                return PILImage.open(processed_path)
            return None
        except Exception as e:
            logger.error(f"Error loading image for display {image_path}: {str(e)}")
            return None


# #### EmbeddingProcessor

# In[328]:


get_ipython().run_line_magic('matplotlib', 'inline')
from transformers import AutoProcessor, AutoModel
import torch
import sentencepiece as spm 
import uuid

class EmbeddingProcessor:

    MAX_TOKEN = 56          # 56 + BOS + EOS = 58 < 64
    OVERLAP   = 16
    DEFAULT_COLLECTION = "ccd_docs_siglip"

    # 初始化 embedding processor
    def __init__(self, 
                persist_directory: str = "chroma_db",
                image_dir: str = "image",
                image_size: tuple = (224, 224),
                collection_name:str = DEFAULT_COLLECTION,
                reset: bool = False 
                ):
        """
        初始化: 建立clip_collection,使用CLIP(or OpenCLIP)做embedding
        """
        # ---------- 路徑 & 基本設定 ----------
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.collection_name = collection_name
        self.image_processor = ImageProcessor(image_dir)

        # ---------- 1) 建立 Chroma client ----------
        logger.info(f"Initializing Chroma with directory: {persist_directory}")
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # ---------- 2) reset (= 刪掉舊庫) ----------
        if reset:
            try:
                self.chroma_client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            except (chromadb.errors.NotFoundError, ValueError):
                logger.info("No old collection to delete")

        
        # ---------- 3) 初始化 SigLIP ----------
        SIGLIP_NAME = "google/siglip-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        self.siglip    = AutoModel.from_pretrained(SIGLIP_NAME)

        # 取輸出向量長度 (base = 768)
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224))
            self.clip_dim = self.siglip.get_image_features(dummy).shape[1]

        # ---------- 4) 取得或建立 collection ----------
        self.clip_collection = self.chroma_client.get_or_create_collection(
            name     = self.collection_name,
            metadata = {"dimension": self.clip_dim}
        )
        logger.info(
            f"Using collection '{self.collection_name}' "
            f"(dimension={self.clip_dim}, reset={reset})"
        )

    def to_2d(self,x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, list):
            x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:        # (512,) → (1,512)
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"embedding ndim should be 1 or 2, got {x.shape}")
        return x.tolist()      # List[List[float]]

    def chunk_text_by_token(
            self,
            text: str,
            max_tokens: Optional[int] = None,
            overlap: Optional[int] = None
        ) -> List[str]:
        CH_SENT_SPLIT = re.compile(r'([。！？；\n])')
        """句號優先斷句；任何子句最終都 ≤ 56 token"""
        max_tokens = max_tokens or self.MAX_TOKEN      # 56
        overlap    = overlap    or self.OVERLAP        # 16

        # --- 1) 以中文標點切成子句 ---
        sentences, buf, parts = [], "", CH_SENT_SPLIT.split(text)
        for frag in parts:
            if CH_SENT_SPLIT.match(frag):
                buf += frag          # 把標點加回來
                sentences.append(buf.strip())
                buf = ""
            else:
                buf += frag
        if buf: sentences.append(buf.strip())

        # --- 2) 任何 >56 token 的句子再滑窗切 ---
        chunks = []
        for s in sentences:
            ids = self.processor.tokenizer(s).input_ids
            if len(ids) <= max_tokens:
                chunks.append(s)
            else:
                step = max_tokens - overlap
                for i in range(0, len(ids), step):
                    seg_ids = ids[i:i + max_tokens]
                    seg = self.processor.tokenizer.decode(seg_ids,
                                                        skip_special_tokens=True)
                    chunks.append(seg)
        return chunks


    def encode_text_to_vec(self, text: str) -> Optional[np.ndarray]:
        """
        用 CLIP 的 text encoder 將文字轉為512維向量
        """
        try:
            chunks = self.chunk_text_by_token(text)
            if not chunks:
                logger.error("No valid chunks generated for the text.")
                return None
            all_vecs = []
            for ch in chunks:
                inp = self.processor(text=[ch], return_tensors="pt").to(self.siglip.device)
                with torch.no_grad():
                    vec = self.siglip.get_text_features(**inp)
                all_vecs.append(vec)
            # 這裡可取平均或直接回傳多條向量
            embs = torch.stack(all_vecs).mean(dim=0)
            emb = embs / embs.norm(dim=-1, keepdim=True)
            return emb.squeeze(0).cpu().tolist()   
        except Exception as e:
            logger.error(f"Error in encode_text_to_vec: {e}")
            return None
        
    def add_qa_pairs(self,
                questions: List[str],
                answers: List[str],
                question_metadatas: List[Dict],
                answer_metadatas: List[Dict],
                images: Optional[List[str]] = None):
        """添加问答对到不同的集合"""
        try:
            # 添加问题
            if questions and question_metadatas:
                logger.info(f"Adding {len(questions)} questions")
                self.question_collection.add(
                    documents=questions,
                    metadatas=question_metadatas,
                    ids=[f"q_{i}" for i in range(len(questions))]
                )
            
            # 添加回答
            if answers and answer_metadatas:
                logger.info(f"Adding {len(answers)} answers")
                self.answer_collection.add(
                    documents=answers,
                    metadatas=answer_metadatas,
                    ids=[f"a_{i}" for i in range(len(answers))]
                )
            
            # 处理图片
            if images:
                logger.info(f"Processing {len(images)} images")
                all_ids=[]
                all_embeddings=[]
                all_metadatas = []

                
                for i, (img_path,question_text) in enumerate(zip(images, questions)):
                    img_emb = self.process_image(str(self.image_dir / img_path))
                    txt_emb = self.encode_text_to_vec(question_text)

                    if img_emb is not None:
                        all_embeddings.append(img_emb.tolist())
                        all_metadatas.append({
                            "type": "image", 
                            "path": img_path,
                            "associated_question": question_text
                        })
                        all_ids.append(f"img_{i}")
                    if txt_emb is not None:
                        all_embeddings.append(txt_emb.tolist())  
                        all_metadatas.append({
                            "type": "clip_text", 
                            "text": question_text,
                            "related_image": img_path
                        })
                        all_ids.append(f"txt_{i}")

                if len(all_embeddings)>0:
                    logger.info(f"Adding {len(all_embeddings)} total embeddings to collection")
                    self.image_collection.add(
                        embeddings=all_embeddings,
                        metadatas=all_metadatas,
                        ids=all_ids
                    )
            
        except Exception as e:
            logger.error(f"Error adding QA pairs: {str(e)}")
            raise

    def encode_image_to_vec(self, image_path: str) -> Optional[np.ndarray]:
            """
            用 CLIP image encoder 將圖片轉為512維向量
            """
            try:
                # 先做基礎處理,縮放或另存
                processed_path = self.image_processor.process_and_save(
                    image_path, self.image_size
                )
                if not processed_path:
                    return None

                image = PILImage.open(processed_path)
                inputs = self.processor(images=[image], return_tensors="pt").to(self.siglip.device)
                with torch.no_grad():
                    embs = self.siglip.get_image_features(**inputs)
                return (embs / embs.norm(dim=-1, keepdim=True)).cpu().numpy()
            except Exception as e:
                print(f"Error in encode_image_to_vec: {str(e)}")
                return None

    def add_vectors(
        self,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        images: Optional[List[str]] = None,
        ):
        """
        統一把文字 / 圖片寫進 clip_collection
        """
        texts     = texts or []
        images    = images or []
        metadatas = metadatas or []

        all_embs, all_metas, docs, all_ids = [], [], [], []
        idx = 0

        # -------------------- 文字 --------------------
        for i, txt in enumerate(texts):
            emb = self.encode_text_to_vec(txt)
            if emb is None:
                continue

            # ① 取 metadata 且保證是 dict
            src_meta = metadatas[i] if i < len(metadatas) else {}
            if not isinstance(src_meta, dict):
                src_meta = {"note": str(src_meta)}

            # ② domain → type 映射（只做一次）
            domain = src_meta.pop("domain", "").lower()
            if domain in {"針灸學", "acupuncture"}:
                src_meta["type"] = "acupoint"
            elif domain in {"herb","herbology"}:
                src_meta["type"] = "herb"
            elif domain in {"ccd","canine"}:
                src_meta["type"] = "ccd"

            for vec in self.to_2d(emb):
                md = {
                    "type": src_meta.get("type", "professional"),
                    "content": txt,
                    **src_meta,            # 其餘欄位保留
                }
                all_embs.append(vec)
                all_metas.append(md)
                docs.append(txt)
                all_ids.append(f"text_{idx}")
                idx += 1

        # -------------------- 圖片 --------------------
        for j, img_name in enumerate(images):
            full_path = str(self.image_dir / img_name)
            emb = self.encode_image_to_vec(full_path)
            if emb is None:
                continue

            src_meta = metadatas[j] if j < len(metadatas) else {}
            if not isinstance(src_meta, dict):
                src_meta = {"note": str(src_meta)}

            # --- 2-1 圖片向量 ---
            img_meta = {
                "type": "image",
                "path": img_name,
                **src_meta,
            }
            all_embs.append(self.to_2d(emb)[0])
            all_metas.append(img_meta)
            docs.append("")                         # 圖片沒有 document
            img_id = f"img_{uuid.uuid4().hex}"
            all_ids.append(img_id)

            # --- 2-2 caption 向量（若有）---
            cap = src_meta.get("caption") or src_meta.get("image_description")
            if cap:
                cap_emb = self.encode_text_to_vec(cap)
                if cap_emb is not None:
                    all_embs.append(self.to_2d(cap_emb)[0])
                    all_metas.append({
                        "type": "caption",          # 方便前端辨識
                        "ref_image": img_name,      # 日後可聚合
                        "content": cap,
                        **src_meta,
                    })
                    docs.append(cap)
                    all_ids.append(f"{img_id}_cap")

            # md = {
            #     "type": "image",
            #     "path": img_name,
            #     **src_meta,
            # }
            # for vec in self.to_2d(emb):
            #     all_embs.append(vec)
            #     all_metas.append(md)
            #     docs.append("")          # 占位
            #     all_ids.append(f"img_{uuid.uuid4().hex}")#(f"img_{idx}")
            #     idx += 1

        # -------------------- 寫入 Chroma --------------------
        if all_embs:
            self.clip_collection.add(
                embeddings = all_embs,
                metadatas  = all_metas,
                documents  = docs,
                ids        = all_ids,
            )
            logger.info(f"Added {len(all_embs)} items to '{self.collection_name}'")


    def similarity_search(self, query: str, k=25) -> Dict:
        """
        對query做CLIP text embedding後,在clip_collection裡找最相似的k筆
        """
        try:
            emb = self.encode_text_to_vec(query)
            if emb is None:
                return {"metadatas":[],"documents":[],"distances":[]}
        
            results = self.clip_collection.query(
                    query_embeddings=[emb],
                    n_results=k,
                    include=["documents","metadatas","distances","embeddings"] #include=["distances", "metadatas", "documents"]
            ) 
            # ---------- ▌動態降權 + re-rank ----------------
            q = query.lower()
            if re.search(r"(st|cv|gv|bl|pc)-\d{1,2}|穴位", q):
                weight = {"herb": 0.3, "ccd": 0.3}     # acupoint = 1.0
            elif any(w in q for w in ["柴胡", "黃芩", "清熱"]):
                weight = {"acupoint": 0.3, "ccd": 0.3}
            elif any(w in q for w in ["認知", "nlrp3", "發炎"]):
                weight = {"herb": 0.3, "acupoint": 0.3}
            else:
                weight = {}

            metas = results["metadatas"][0]
            dists = results["distances"][0]
            docs  = results["documents"][0]

            scored = []
            for i, (m, d) in enumerate(zip(metas, dists)):
                w = weight.get(m.get("type", ""), 1.0)
                scored.append((d * w, i))          # 距離愈小愈好
            scored.sort(key=lambda x: x[0])

            idxs = [i for _, i in scored][:k]       # 取前 k
            # idxs = list(range(len(metas)))[:k]
            
            for key in ["metadatas", "distances", "documents"]:
                results[key][0] = [results[key][0][i] for i in idxs]

                return results
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {"metadatas":[],"documents":[],"distances":[]}


# #### DataProcessor

# In[165]:


class DataProcessor:
    def __init__(self, embedding_processor: 'EmbeddingProcessor'):
        self.embedding_processor = embedding_processor
        
    def extract_social_posts(self, csv_path: str) -> Tuple[List[Dict], List[str]]:
        """处理 CSV 并提取问答对和图片"""
        logger.info(f"Processing CSV: {csv_path}")
        qa_pairs = []
        images = []
        
        df = pd.read_excel(csv_path)
        current_post = None
        current_responses = []
        current_images = []
        current_link = None
        
        for _, row in df.iterrows():
            # 处理新的帖子
            if pd.notna(row['post']):
                # 保存前一个问答对
                if current_post is not None:
                    qa_pair = {
                        'question': current_post,
                        'answers': current_responses.copy(),
                        'images': current_images.copy(),
                        'metadata': {
                            'type': 'social',
                            'source': 'facebook',
                            'images': ','.join(current_images) if current_images else '',
                            'answer_count': len(current_responses),
                            'link': current_link if current_link else ''
                        }
                    }
                    if pd.notna(row.get('image_description')):
                        qa_pair['metadata']['image_desc'] = row['image_description']

                    qa_pairs.append(qa_pair)
                    if current_images:
                        images.extend(current_images)
                
                # 初始化新的问答对
                current_post = row['post']
                current_responses = []
                current_images = []
                current_link = row.get('link', '')
            
            # 添加回复
            if pd.notna(row.get('responses')):
                current_responses.append(row['responses'])
            
            # 处理图片
            if pd.notna(row.get('images')):
                img_path = row['images']
                current_images.append(img_path)
                logger.info(f"Found image: {img_path} for current post")
  
        
        # 保存最后一个问答对
        if current_post is not None and len(current_responses) >= 3:
            qa_pair = {
                'question': current_post,
                'answers': current_responses,
                'images': current_images,
                'metadata': {
                    'type': 'social_qa',
                    'source': 'facebook',
                    'images': ','.join(current_images) if current_images else '',
                    'answer_count': len(current_responses),
                    'link': current_link if current_link else ''
                }
            }
            if pd.notna(row.get('image_description')):
                    qa_pair['metadata']['image_desc'] = row['image_description']

            qa_pairs.append(qa_pair)
            if current_images:
                images.extend(current_images)
        
        # 显示处理结果的详细信息
        for i, qa in enumerate(qa_pairs):
            logger.info(f"\nQA Pair {i+1}:")
            logger.info(f"Question: {qa['question'][:100]}...")
            logger.info(f"Number of answers: {len(qa['answers'])}")
            logger.info(f"Images: {qa['images']}")
            logger.info(f"Link: {qa.get('link', 'No link')}")
        
        return qa_pairs, images

    
    def chunk_text(self,paragraph: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        將給定段落，以 chunk_size 字符為上限進行切分，並且在 chunk 之間保留 overlap 個字的重疊，
        以免上下文斷裂。
        備註: 
        - 這裡以「字符」為單位，適合中文；英文也可用，但若想精確對英文 tokens 可改更先進方法。
        """
        chunks = []
        start = 0
        length = len(paragraph)

        # 去掉前後多餘空白
        paragraph = paragraph.strip()

        while start < length:
            end = start + chunk_size
            # 取 substring
            chunk = paragraph[start:end]
            chunks.append(chunk)
            # 移動指標(下一個 chunk)
            # overlap 預防斷句失去上下文
            start += (chunk_size - overlap)

        return chunks
  

    def process_pdf(self, pdf_path: str,row_type: str) -> List[Dict]:
        logger.info(f"Processing PDF: {pdf_path}")
        professional_qa_pairs = []
        pdf_name = Path(pdf_path).name  # 获取文件名
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                is_formula = self.detect_domain(pdf_name) == "中醫方劑"
                is_acu = self.detect_domain(pdf_name) == "針灸學"


                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    print(f"Page {page_num+1} raw text:", repr(text))
                    if is_formula:
                        paragraphs = self.split_formula_blocks(text)
                    elif is_acu:
                        paragraphs = self.split_acu_blocks(text)
                    else:
                        paragraphs = text.split('\n\n')
                    
                    
                    # 處理每個段落
                    for para in paragraphs:
                        # logger.info(f"Para type: {type(para)}")

                        para_chunks = self.chunk_text(para)
                        # logger.info(f"Got {len(para_chunks)} chunks from chunk_text()")

                        for c in para_chunks:
                            qa_pair = {
                                'question': c[:50] + "...",  
                                'answers': [c],
                                'metadata': {
                                    'type': row_type,
                                    'source_file': pdf_name,  # 添加文件名
                                    'page': str(page_num + 1),
                                    'content_length': str(len(c))
                                } #'domain':self.detect_domain(pdf_name),
                            }
                            professional_qa_pairs.append(qa_pair)
                
                logger.info(f"Extracted {len(professional_qa_pairs)} paragraphs from {pdf_name}")
                return professional_qa_pairs
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_name}: {str(e)}")
            return []
        
    def detect_domain(self, pdf_name: str) -> str:
        lower = pdf_name.lower()

        if "針灸" in pdf_name or "acupuncture" in lower:
            return "針灸學"
        if "herbal" in lower or "herbology" in lower or "方劑" in pdf_name:
            return "中醫方劑"
        return "其他"
    
    def split_formula_blocks(self,text: str) -> list[str]:
        """
        用正則抓出『● 六味地黃丸』或『Liu Wei Di Huang Wan』開頭，
        每遇下一個方名就結束上一塊
        """
        pattern = re.compile(r"(?:●|\s)([\u4e00-\u9fffA-Za-z\- ]{3,40}(?:湯|丸|飲|散|膏))")
        blocks = []
        cur_block = []
        for line in text.splitlines():
            if pattern.search(line):
                # 遇到下一帖藥 → 先收前一帖
                if cur_block:
                    blocks.append("\n".join(cur_block).strip())
                    cur_block = []
            cur_block.append(line)
        if cur_block:
            blocks.append("\n".join(cur_block).strip())
        return [b for b in blocks if len(b) > 60]    

    def split_acu_blocks(self,text: str) -> list[str]:
        # 範例代碼：LI‑11、HT-7、SI 3
        pattern = re.compile(r"\b([A-Z]{1,2}[ -‑]\d{1,3})\b")
        blocks, cur = [], []
        for line in text.splitlines():
            if pattern.search(line):
                if cur: blocks.append("\n".join(cur).strip()); cur = []
            cur.append(line)
        if cur: blocks.append("\n".join(cur).strip())
        return [b for b in blocks if len(b) > 40]

    def process_all(self, csv_path: str, pdf_paths: List[str]):
        """綜合處理社群 CSV + PDFs"""
        try:
            social_qa_pairs, images = [], []  
            # 1. 处理社群数据
            if csv_path: 
                social_qa_pairs, images = self.extract_social_posts(csv_path)
                logger.info(f"\nProcessed social data:")
                logger.info(f"- Social QA pairs: {len(social_qa_pairs)}")
                logger.info(f"- Images found: {len(images)}")
            else:
                logger.info("Skip social CSV, only處理 PDFs")
            # 检查图片文件
            valid_images = []
            for img in images:
                img_path = Path(self.embedding_processor.image_dir) / img
                if img_path.exists():
                    valid_images.append(img)
            
            # 2. 处理所有 PDF
            all_professional_pairs = []
            for pdf_path in pdf_paths:
                pdf_name = pdf_path.name.lower()
                for t, keys in TYPE_MAP.items():
                    if any(k in pdf_name for k in keys):
                        row_type = t; break
                else:
                    row_type = "professional"
                pdf_qa_pairs = self.process_pdf(pdf_path, row_type=row_type)
                #pdf_qa_pairs = self.process_pdf(pdf_path)
                all_professional_pairs.extend(pdf_qa_pairs)
                logger.info(f"\nProcessed {Path(pdf_path).name}:")
                logger.info(f"- Extracted paragraphs: {len(pdf_qa_pairs)}")
            
            # 3. 合并 => all_qa_pairs
            all_qa_pairs = social_qa_pairs + all_professional_pairs
            
            # 4. 準備 texts + metadatas => 你就能一次或多次呼叫 add_vectors
            questions = []
            answers = []
            question_metas = []
            answer_metas = []
            
            # 处理所有问答对
            for qa_pair in all_qa_pairs:
                # question 
                questions.append(qa_pair['question'])
                question_metas.append(qa_pair['metadata'])
                
                # answers
                for ans_text in qa_pair['answers']:
                    answers.append(ans_text)
                    am = qa_pair['metadata'].copy()
                    am['parent_question'] = qa_pair['question']
                    answer_metas.append(am)

            # ------------- 這裡才開始組 professional texts / metas -------------
            prof_texts = [qa["answers"][0] for qa in all_professional_pairs]
            prof_metas = [qa["metadata"]   for qa in all_professional_pairs]

            
            # 输出处理结果
            logger.info(f"\nFinal processing summary:")
            logger.info(f"- Total questions: {len(questions)}")
            logger.info(f"- Total answers: {len(answers)}")
            logger.info(f"- Valid images: {len(valid_images)}")
            logger.info(f"- Social content: {len(social_qa_pairs)} QA pairs")
            logger.info(f"- Professional content: {len(all_professional_pairs)} paragraphs")
            


            # --------- 🔧 把 3 組 metadata 都保證是 dict (放在此處) ---------
            question_metas = [m if isinstance(m, dict) else {"note": str(m)}
                            for m in question_metas]
            prof_metas     = [m if isinstance(m, dict) else {"note": str(m)}
                            for m in prof_metas]
            # 若要用 answer_metas 也一併處理
            answer_metas   = [m if isinstance(m, dict) else {"note": str(m)}
                            for m in answer_metas]


            self.embedding_processor.add_vectors(texts=prof_texts,
                                            metadatas=prof_metas)
            
            # (A) 先加所有 question
            self.embedding_processor.add_vectors(
                texts = questions,
                metadatas = question_metas
            )

            if valid_images:
                meta_for_imgs = []
                for img_name in valid_images:
                    meta_for_imgs.append({
                        "type":"image",
                        "source":"facebook",
                        "filename": img_name
                    })

                self.embedding_processor.add_vectors(
                    images=valid_images,
                    metadatas=meta_for_imgs
                )

            logger.info("All data added to clip_collection.")
            return len(questions), len(valid_images) #return questions, question_metas, all_professional_pairs, valid_images
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise


# #### QA System

# In[ ]:


from deep_translator import GoogleTranslator
class QASystem:
    def __init__(self, embedding_processor: 'EmbeddingProcessor',
                 model_name: str = 'llama3.2-vision'):
        self.embedding_processor = embedding_processor
        self.model_name = model_name
        logger.info(f"Initialized QA System with Ollama model: {model_name}")

    def _classify_collection_results(self, raw_result) -> Dict:
        """
        將 clip_collection 的檢索結果 (metadatas/documents...) 
        轉換成 { 'social': {...}, 'professional': {...}, 'images': {...} } 
        的結構，便於後續 gather_references / format_context。
        """
        # 預設空結構
        structured = {
            "social": {
                "metadata": [],
                "link": [],
                "content": [],
                "documents":[]
            },
            "professional": {
                "metadata": [],
                "content": [],
                "documents":[]
            },
            "images": {
                "metadata": [],
                "paths": [],
                "relevance":[]
            },
            
        }

        # raw_result["metadatas"] 是個 2D list => [ [meta0, meta1, ...] ]
        if raw_result.get("metadatas"):
            meta_list = raw_result["metadatas"][0]  # 因為只有1個 query
            dist_list = raw_result["distances"][0] if raw_result.get("distances") else []
            doc_list = raw_result["documents"][0]
            ids_list  = raw_result["ids"][0] if raw_result.get("ids") else []
            # documents_list = raw_result["documents"][0]

            for i, meta in enumerate(meta_list):
                dist = dist_list[i] if i < len(dist_list) else None
                doc_id = ids_list[i] if i < len(ids_list) else ""
                doc_text = doc_list[i] if i < len(doc_list) else ""

                src_type = meta.get("type","")

                if src_type == "social":
                    structured["social"]["metadata"].append(meta)
                    structured["social"]["documents"].append(doc_text)
                elif src_type in ("acupoint", "herb", "ccd", "professional"):
                    structured["professional"]["metadata"].append(meta)
                    structured["professional"]["documents"].append(doc_text)
                elif src_type ("image","images","herb_img"): #== "image":
                    structured["images"]["metadata"].append(meta)
                    # 放 path
                    path = meta.get("path","")
                    structured["images"]["paths"].append(path)
                    structured["images"]["relevance"].append(dist)

                else:
                    # 將未知 type 全丟 professional，或依需求改 social
                    meta.setdefault("type", "professional")
                    structured["professional"]["metadata"].append(meta)
                    structured["professional"]["documents"].append(doc_text)


        return structured


    def determine_question_type(self,query: str) -> str:
        """
        回傳: "multiple_choice" | "true_false" | "qa"
        支援中英文 & 各種標點
        """
        q = query.strip().lower()

        # --- Multiple‑choice --------------------------------------------------
        # 1) 行首或換行後出現  A～D / 全形Ａ～Ｄ / 「答」，
        #    後面接　. ． : ： 、)
        mc_pattern = re.compile(r'(?:^|\n)\s*(?:[a-dａ-ｄ]|答)[:\.．:：、\)]', re.I)
        # 2) or 句子帶 "which of the following"
        mc_keywords_en = ["which of the following", "which one of the following",
                        "which option", "choose one of"]

        if mc_pattern.search(query) or any(kw in q for kw in mc_keywords_en):
            return "multiple_choice"

        # --- True / False -----------------------------------------------------
        tf_keywords_zh = ["是否", "是嗎", "對嗎", "正確嗎"]
        tf_keywords_en = ['true or false', 'is it', 'is this', 'is that', 
             'is it possible', 'correct or not']

        if any(k in q for k in tf_keywords_zh + tf_keywords_en):
            return "true_false"

        # --- Default ----------------------------------------------------------
        return "qa"

    
    def gather_references(self, search_results: Dict) -> str:
        """
        從 search_results 中擷取 PDF 檔名/社群連結，並組成一個字串
        """
        if not isinstance(search_results, dict):
            logger.error("search_results 格式錯誤: %s", type(search_results))
            return ""

        references = []

        # 處理 social
        for meta in search_results["social"].get("metadata", []):
            if meta.get("type") == "social_qa" and "link" in meta:
                references.append(f"(經驗) {meta['link']}")

        # 處理 professional
        for meta in search_results["professional"].get("metadata", []):
            if meta.get("type") in ["pdf", "professional"]:
                pdf_name = meta.get("source_file", "unknown.pdf")
                references.append(f"(文獻) {pdf_name}")

        # 去重
        unique_refs = list(set(references))
        return "\n".join(unique_refs)


    def build_user_prompt(
        self,
        query: str,
        context: str,
        references_str: str = ""
        ) -> str:
        # 不含任何格式規範！只給題目與資料
        return (
            f"""
            【參考資料】
            {context}\n
            【資料來源】
            {references_str}\n    
            【問題】
            {query}\n
            """
            # "參考資料：\n" + context +
            # "\n來源：\n" + references_str +
            # "問題："＋query
        )


    def merge_adjacent(self, metas, docs, k_keep: int = 5) -> str:
        """
        將同檔同頁且 _id 連號的片段合併，回傳前 k_keep 段文字。
        參數
        ----
        metas : list[dict]    # raw_result["metadatas"][0]
        docs  : list[str]     # raw_result["documents"][0]
        """
        ID_NUM_RE = re.compile(r"_(\d+)$")   # 尾碼取數字：text_123 → 123
        merged, buf = [], ""
        last_src, last_idx = ("", ""), -999

        for md, doc in zip(metas, docs):
            src_key = (md.get("source_file", ""), md.get("page", ""))

            # 取 _id 尾碼；若不存在則設 -1
            _id = md.get("_id", "")
            m = ID_NUM_RE.search(_id)
            cur_idx = int(m.group(1)) if m else -1

            # 同檔同頁且連號 → 視為相鄰
            if src_key == last_src and cur_idx == last_idx + 1:
                buf += doc
            else:
                if buf:
                    merged.append(buf)
                buf = doc
            last_src, last_idx = src_key, cur_idx

        if buf:
            merged.append(buf)

        return "\n\n".join(merged[:k_keep])


    def generate_response(self, query: str,question_type: Optional[str] = None) -> Tuple[str, List[str]]:
        try:
            raw_result = self.embedding_processor.similarity_search(
                query,
                k=25)  
            print(raw_result["metadatas"])
            
            if not raw_result["documents"] or len(raw_result["documents"][0]) == 0:
                logger.warning("No hits for query → 改用 k=50 再試一次")
                raw_result = self.embedding_processor.similarity_search(query, k=50)

            if not raw_result["documents"] or len(raw_result["documents"][0]) == 0:
                return "[NoRef] 無足夠證據判斷", [],[]
            
            # 用後處理
            search_results = self._classify_collection_results(raw_result)
            logger.info("SEARCH RESULT(structured): %s",search_results)

            context = self.merge_adjacent(raw_result["metadatas"][0],
                              raw_result["documents"][0])[:1500]

            context = context[:1500]          # 最多 1500 字

            references_str = self.gather_references(search_results)
            # link應該用傳參數的會成功 可能用context.link之類的抓題目的reference

            # --- ① 題型 --------------------------------------------------------
            q_type = question_type or self.determine_question_type(query)

            user_prompt = self.build_user_prompt(
                query=query,
                context=context[:1500],
                references_str=references_str
            )
            # ---------- ② 根據題型動態組 system 指令 ----------
            if q_type == "multiple_choice":
                format_rules = (
                    "這是一題選擇題，回答格式如下：\n"
                    "先根據題目整理參考資訊、你的理解與常識\n"
                    "用 2-3 句話說明理由。\n"
                    "最後再給出答案，只能回答 A/B/C/D (請勿帶任何標點、文字、也不要只回答選項內容)\n"
                    "若同時出現多個選項，請只選一個最適合的\n"
                )
            elif q_type == "true_false":
                format_rules = (
                    "這是一題是非題，請按照下列格式回答：\n"
                    "先根據題目整理參考資訊、你的理解與常識\n"
                    "最後再給出答案，只能寫「True」或「False」\n"
                )
            else:   # qa
                format_rules = (
                    "請依以下格式回答：\n"
                    "針對問題提供具體答案並用2-3句話說明 \n"
                )

            #"您是一名專業獸醫，1.擅長犬認知功能障礙綜合症（CCD）的診斷和護理 2.擁有豐富的寵物中醫知識 3.常見問題診斷及改善建議\n" "請在答案最後顯示你參考的來源連結或論文名稱，如果來源中包含「(經驗) some_link」，請在回答中以 [Experience: some_link] 形式標示；若包含「(文獻) some.pdf」，就 [reference: some.pdf]\n""如檢索結果仍無相關資訊，請以[NoRef]標示並根據你的常識回答。\n"
            system_prompt = (
                """你是資深獸醫，擅長犬認知功能障礙綜合症（CCD）的診斷和護理並擁有豐富的寵物中醫知識，必須遵守以下規則回答問題：
                    1. 只能根據【檢索結果】內容作答；若資訊不足，請回答「資料不足」，並依照你的常識回答。
                    2. 若需補充一般臨床常識，請將該句放在段落最後並標註［常識］。
                    3. 每一句結尾必須標註引用來源編號，如［1］或［1,3］。
                    4. 並在最後面整理列出每個編號的source_file，如[1] ...pdf 或 [2] Chinese Veterinary Materia Medica """
                + format_rules
            )
        

            # 處理圖片
            image_paths = []
            # 2) 把圖片撈出
            # -- (a) 先掃 retrieved_contexts --
            for meta in search_results.get("images", {}).get("metadata", []):
                if meta.get("type") in ("image", "herb_img") and meta.get("path"):
                    full_path = self.embedding_processor.image_dir / meta["path"]
                    if full_path.exists():
                        image_paths.append(str(full_path.resolve()))

            # -- (b) 保留舊的 social.images 規則 (若還需要) --
            for md in search_results.get("social", {}).get("metadata", []):
                if md.get("images"):
                    for img_name in md["images"].split(","):
                        full_path = self.embedding_processor.image_dir / img_name.strip()
                        if full_path.exists():
                            image_paths.append(str(full_path.resolve()))
            # 3) OLlama 只允許一張, 你可取 image_paths[:1] => message["images"] = ...
            if image_paths:
                print("We found images: ", image_paths)
            else:
                logger.info("No images to display")
            
            

            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
                # {'images': image_paths[:1]}
            ]
            # print("=======sys prompt =======",system_prompt)
            # print("=======user prompt =======",user_prompt)
            # 生成响应
            response = ollama.chat(
                model=self.model_name,
                messages=message
            )


            # 取得檢索段落（文字即可）
            retrieved_contexts = search_results["professional"]["documents"] + \
                                search_results["social"]["documents"]

            # 把三樣都回傳 ----------------------------------▼ 新增
            response_text = response["message"]["content"]

            return response_text, retrieved_contexts, image_paths #response['message']['content'], image_paths

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # return f"出現問題，檢查ollama連線或是generate_response", []
            raise

    def format_context(self, search_results: Dict) -> str:
        """Format context from search results"""
        try:
            context = ""

            # 1) 處理社群討論
            social_metas = search_results["social"].get("metadata", [])
            social_links = search_results["social"].get("link", [])
            social_docs = search_results["social"].get("documents", [])
            social_content = search_results["social"].get("content", [])

            if social_metas or social_links or social_docs:
                context += "\n[社群討論]\n"
                # 這裡示範把 link、documents 都輸出
                for i, meta in enumerate(social_metas):
                    link_str = meta.get("link", "")
                    doc_text = social_docs[i] if i < len(social_docs) else ""
                    context += f"【Link】{link_str}\n" if link_str else ""
                    # doc_text 就是檢索回來的 chunk
                    context += f"【討論片段】{doc_text}\n\n"

            # 2) 處理專業文獻
            prof_metas = search_results["professional"].get("metadata", [])
            prof_docs = search_results["professional"].get("documents", [])

            if prof_metas or prof_docs:
                context += "\n[專業文獻]\n"
                for j, meta in enumerate(prof_metas):
                    source_file = meta.get("source_file", "")
                    doc_text = prof_docs[j] if j < len(prof_docs) else ""
                    # 如果您有另外存放頁碼 page = meta.get("page"), 也可列出
                    page_num = meta.get("page", "")
                    context += f"【文件片段】{doc_text}\n"
                    if source_file:
                        context += f"(檔案: {source_file}"
                        context += f", 頁: {page_num})" if page_num else ")"
                    context += "\n\n"

            # 偵錯用 (可保留也可移除)
            print("social metadata:", social_metas)
            print("social links:", social_links)
            print("professional metadata:", prof_metas)

            return context if context.strip() else "參考資料無法取得"

        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return "Unable to retrieve reference materials"


    def display_response(self, query: str,question_type: Optional[str] = None):
            """Display response with text and images"""
            try:
                logger.info("Starting to generate response...")
                try:
                    response_text, _ , image_paths = self.generate_response(query,question_type)
                except Exception as e:
                    response_text = f"[ERROR] {e}"
                    image_paths = []
                
                print("Question:", query)
                print("\nSystem Response:")
                print(response_text)
                print("\n" + "-"*50 + "\n")

                if image_paths:
                    print("\nRelated Image:")
                    img_path = image_paths[0]  # We now only have one image
                    try:
                        img = PILImage.open(img_path)
                        display(IPyImage(filename=img_path))
                    except Exception as e:
                        logger.error(f"Error displaying image {img_path}: {str(e)}")
                else:
                    logger.info("No images to display")


                return response_text, image_paths # add for response 0406
                    
            except Exception as e:
                logger.error(f"Error in display_response: {str(e)}", exc_info=True)  
                return "", [] 


# ### 題目測試

# ##### 單一題目測試

# In[ ]:


# 測試查詢
test_queries = [
# "CCD 是否與神經發炎相關？有無特定細胞因子（cytokines）或發炎路徑（例如NLRP3 inflammasome）參與？",
# "CCD 是否與腸道微生物群變化有關？是否有特定細菌群落會影響大腦健康？",
# " 失智犬的松果體是否退化",
# " 有刻板形為的犬隻是否會增加CCD風險？",
# " 失智犬分泌褪黑激素的能力是否退化？",
# " 皮質類固醇cortisol或應激荷爾蒙stress hormones是否可作為 CCD 的潛在診斷指標？",
# " 如何區分正常老化與CCD的早期徵兆？ ",
# " B 群維生素是否能降低 CCD 進展風險？",
# " 食用GABA是否對於預防CCD有效？",
# " 警犬、救難犬等工作犬在罹患CCD的機率比較家庭陪伴犬",
# " 目前是否有影像學檢測可以準確區分 CCD 與其他神經退行性疾病？",
# " 如果CCD進展到最後階段，哪些症狀最需要關注？如何平衡狗狗的生活質量與疼痛管理，並且決定狗狗未來的方向",

# "根據資料中對犬認知功能障礙（CCD）神經發炎機制的探討，NLRP3炎症小體在分子層面上如何參與CCD進程？該過程涉及哪些關鍵細胞因子與調控機制？",
# "資料提到腸道微生物群與CCD之間可能存在聯繫，請問文中如何闡述腸道菌群失衡影響神經傳導與免疫反應的分子機制？哪些特定細菌群落的變化被認為與CCD進展相關？",
# "在探討CCD的診斷策略中，該資料對於利用影像學技術（如MRI與CT）區分CCD與其他神經退行性疾病的應用提出了哪些見解？這些技術的優勢與局限性分別是什麼？",
# "資料中對失智犬松果體退化與褪黑激素分泌減少之間的關聯有詳細論述，請問該研究如何描述這一生理變化的分子機制以及其對犬隻睡眠-覺醒週期的影響？",
# "針對CCD的治療策略，資料中提出了哪些基於分子機制的治療方法？請分析這些方法在臨床應用上的現狀、潛在優勢及未來研究中亟待解決的挑戰。",

# "哪種犬容易失智？",
# "大中小型狗的失智照顧方式有什麼不同？"
# "我的狗狗有失智症，晚上總是繞圈圈而且叫個不停，有什麼方法能幫助牠安靜下來睡覺嗎？有人推薦過褪黑激素，這真的有效嗎？",
# "我的老狗有認知障礙，經常卡在角落或家具間不知道如何脫困，有什麼環境安排和居家照護措施可以幫助牠更舒適地生活？其他飼主都是怎麼處理這種情況的？有相關照片嗎？",
# "給我一些照護環境的圖片",
# "針對年長犬隻可能出現的神經病理變化，哪些關鍵指標常被用來對比阿茲海默類型的退化症狀，並且與臨床觀察到的行為衰退有何關聯？",
# "除了藥物介入之外，平時飼養管理與環境調整方面有哪些具體作法，能同時有助於失智犬與失智貓維持較佳的生活品質，並為何多種方式並用的照護策略往往更能延緩認知退化？",
# "若以老犬作為模擬人類老化與失智的實驗模型，進行認知增益或治療性藥物的評估時，最常採用哪些評量方法來確認藥物對行為和神經功能的影響，並且在哪些神經傳導路徑上通常會看到較明顯的指標性變化？",
# "In older dogs, which key indicators are commonly used to compare with Alzheimer-type degeneration, and how do these indicators relate to clinically observed behavioral decline?",
# "Beyond pharmacological intervention, which specific management and environmental adjustments help senior dogs and cats with cognitive impairment maintain a higher quality of life, and why does combining multiple caregiving strategies often slow cognitive decline more effectively?",
# "When using senior dogs as a model for human aging and dementia to evaluate cognitive-enhancing or therapeutic drugs, what assessment methods are most commonly employed to gauge the drug’s effects on behavior and neurological function, and in which neurotransmission pathways are the most prominent changes typically observed?"
"在評估犬隻 CCD 的臨床症狀時，下列哪一項行為面向最常被列為主要觀察指標之一? A. 毛色是否變白 B. 飲水量的增加 C. 定向能力 (Orientation) 與空間辨識度 D. 心跳與呼吸速率"

                    ]

for query in test_queries:
    qa_system.display_response(query)


# Embedding processor

# In[332]:


from pathlib import Path
TEST_MODE = False                           # ← 切換開關
COLLECTION_NAME = "clip_collection_0504"   

# 1) 初始化 embedding_processor，傳入新的 collection_name
embedding_processor = EmbeddingProcessor(
    image_size=(224, 224) ,
    collection_name=COLLECTION_NAME,    # ★若 __init__ 沒這參數，改下方註解方法
    reset=False
)


# In[ ]:


manifest = pd.read_excel("herb_image_manifest.xlsx")

for row in manifest.itertuples(index=False):
    qa_system.embedding_processor.add_vectors(
        images=[row.filename],
        metadatas=[{
            "type":      "images",
            "herb":      row.herb_name,
            "caption":   row.caption,
            "path":      row.filename,      # ★ 確保有 path
            "source_file":"herb_images"
        }]
    )


# In[337]:


col      = qa_system.embedding_processor.clip_collection
manifest = pd.read_excel("herb_image_manifest.xlsx")

for row in manifest.itertuples(index=False):
    path      = row.filename
    herb_name = row.herb_name
    caption   = str(row.caption) if pd.notna(row.caption) else ""

    # --- 用 get() 判斷是否已有該圖 ---
    exists = bool(col.get(where={"path": path})["ids"])

    if exists:
        # 庫裡已有圖片向量 → 只補 caption（768-dim text）
        qa_system.embedding_processor.add_vectors(
            texts=[caption],
            metadatas=[{
                "type":      "caption",
                "ref_image": path,
                "herb":      herb_name,
                "path":      path
            }]
        )
    else:
        # 圖片 + caption 一起加（各 1 筆向量，都是 768-dim）
        qa_system.embedding_processor.add_vectors(
            images   =[path],
            metadatas=[{
                "type":     "image",
                "herb":     herb_name,
                "caption":  caption,
                "path":     path
            }]
        )

print("✅ herb 圖片與 caption 完成同步（仍為 768 維向量空間）")


# In[335]:


raw = embedding_processor.similarity_search("What does GanCao look like?", k=30)
print(raw)  # 輸出完整的回傳資料
print(raw["metadatas"])  # 確保其中有"image"或你設定的type


# In[336]:


resp, imgs = qa_system.display_response("What does GanCao look like?", "qa")


# ##### 重建DB

# In[ ]:


# 2) 初始化資料處理器
data_processor = DataProcessor(embedding_processor)

# 3) 指定測試或正式資料夾
rag_data_dir = Path("RAG_data_test" if TEST_MODE else "RAG_data")
pdf_paths = list(rag_data_dir.glob("*.pdf"))


print("找到以下 PDF：")
for p in pdf_paths: print(" -", p.name)

# 4) 處理資料 （CSV 你可以傳 None 代表不處理社群資料）
_ = data_processor.process_all(
    csv_path="post_response_filtered.xlsx",           # 只測 PDF，可先不管社群
    pdf_paths=pdf_paths
)


# ##### Initialized QA System

# In[333]:


# 建立 QA 系統，沿用同一個 embedding_processor
qa_system = QASystem(
    embedding_processor=embedding_processor,
    model_name='llama3.2-vision'
)


# ##### 翻譯

# In[ ]:


def translate_zh_to_en(chinese_text: str) -> str:
    try:
        # 指定原文語言為 'zh'（中文），目標語言為 'en'（英文）
        translator = GoogleTranslator(source='zh-TW', target='en')
        result = translator.translate(chinese_text)
        return result
    except Exception as e:
        print(f"翻譯錯誤：{e} - 對應中文問題：{chinese_text}")
        return chinese_text  # 若翻譯失敗，返回原文


# ##### 判斷正確答案

# In[ ]:


import re

def parse_llm_answer(resp: str, q_type: str) -> str:
    """
    解析 LLM 回答文字，回傳最終答案：
      • multiple_choice → 'A'|'B'|'C'|'D'|'UNK'
      • true_false      → 'TRUE'|'FALSE'|'UNK'
    """
    txt = resp.lower()
    txt = re.sub(r'[，。、．；：\s]+', ' ', txt)        # 先統一空白

    if q_type == "multiple_choice":
        # 找所有「獨立」的 a-d (含大小寫)，不含 '選項a' 這種組字
        matches = re.findall(r'\b([abcd])\b', txt, flags=re.I)
        return matches[-1].upper() if matches else "UNK"

    elif q_type == "true_false":
        # 找所有 true/false / 對/錯 / 是/否
        tf_matches = re.findall(
            r'\b(true|false|正確|錯誤|對|錯|是|否)\b', txt)
        if not tf_matches:
            return "UNK"
        last = tf_matches[-1]
        return "TRUE" if last in ("true", "正確", "對", "是") else "FALSE"

    else:   # 其餘題型原文返回
        return resp


# 測試

# In[274]:


# 1. 讀檔 + 題型篩選
df = pd.read_excel("test_questions_en.xlsx")#test_questions #test_questions_withANS
# test_df = df[df["type"].isin(["multiple_choice", "true_false"])].copy()
test_df = df[df["type"].isin(["qa"])].copy()

# 2. ★ 建立欄位（一定要在後面的篩選前先加）
test_df["llm_response"] = ""
test_df["predicted"]    = ""
test_df["is_correct"]   = 0

# 3. 再依 domain 篩子集合
# test_df = test_df[test_df["domain"] == "中醫"].copy()
test_df=test_df.head(5)


# In[ ]:


dataset = [] # for ragas

# 4. 迴圈計分
for idx, row in test_df.iterrows():
    q  = row["question_en"]
    q_type = row["type"]
    gt = str(row["answers"]).strip()
    ref_ctx   = [ str(row["RAG"]) ] 

    resp, _ = qa_system.display_response(q, q_type)

    if not resp.strip():
        print(f"[WARN] id={row['id']}  LLM 回傳空白")


    resp, ctxs, _ = qa_system.generate_response(q, q_type)
    pred = parse_llm_answer(resp, q_type)

    test_df.at[idx, "llm_response"] = resp
    test_df.at[idx, "predicted"]    = pred
    test_df.at[idx, "is_correct"]   = int(pred.upper() == gt.upper())
    
    ctxs = [str(c) for c in ctxs]

    dataset.append({
        "user_input":           str(q),           # question
        "response":             str(resp),        # llm response
        "retrieved_contexts":   ctxs,             # llm檢索到的資料
        "reference_contexts":   ref_ctx,          # 出題段落
        "reference":            gt                # answers
    })


# In[ ]:


# 5. 計算 Accuracy
overall_acc = test_df["is_correct"].mean()

print("\n=== 每個 domain 的 Accuracy ===")
domain_stats = (
    test_df.groupby("domain")["is_correct"]
           .agg(["count", "sum"])
           .reset_index()
           .rename(columns={"sum": "correct"})
)
domain_stats["accuracy"] = domain_stats["correct"] / domain_stats["count"]
print(domain_stats.to_string(index=False, 
      formatters={"accuracy": "{:.2%}".format}))

print(f"\nOVERALL Accuracy = {overall_acc:.2%}")


# In[ ]:


# 先挑出答錯的資料列
wrong_df = (
    test_df.loc[test_df["is_correct"] == 0,
                ["id", "question", "answers", "predicted"]]
            .sort_values("id")          # 依題號排序方便查看
)

print("=== 答錯題目一覽 ===")
print(wrong_df.to_string(index=False))


# #### 檢查

# In[ ]:


coll = qa_system.embedding_processor.clip_collection
# 抽 200 筆看 metadata 出現哪些 type
sample = coll.get(limit=1000, include=["metadatas"])
types  = [m.get("type") for m in sample["metadatas"]]
print(set(types))


# In[ ]:


import numpy as np, pandas as pd, re

coll = qa_system.embedding_processor.clip_collection

# ----- 1. similarity_search() -----
raw = qa_system.embedding_processor.similarity_search(
"some Canine behavior", k=20)

# ----- 2. 安全攤平工具 -----
def safe_flat(val):
    if val is None:
        return []
    if isinstance(val, list):
        flat = []
        for v in val:
            flat.extend(v if isinstance(v, list) else [v])
        return flat
    return [val]

flat_ids        = safe_flat(raw.get("ids"))
flat_metas      = safe_flat(raw.get("metadatas"))
flat_docs       = safe_flat(raw.get("documents"))
flat_embeds     = safe_flat(raw.get("embeddings"))

# 若 embeddings 沒回傳，確保長度對齊
if not flat_embeds:
    flat_embeds = [None] * len(flat_ids)

# ----- 3. 只看前 5 筆 -----
rows = []
for _id, meta, doc, emb in zip(flat_ids[:5], flat_metas, flat_docs, flat_embeds):
    if emb is None:                              # 沒帶回向量就額外 get
        emb = coll.get(ids=[_id], include=["embeddings"])["embeddings"][0]
    norm = np.linalg.norm(emb)

    if isinstance(meta, dict):                   # 有些版本 meta 可能空
        src = meta.get("source_file", "?")
        pg  = meta.get("page", "?")
    else:
        src = pg = "?"

    rows.append({
        "id":      _id,
        "norm":    round(norm, 4),
        "source":  src,
        "page":    pg,
        "preview": str(doc).replace("\n", " ")[:60] + "…"
    })

print("🔍 LI-11 向量 L2-norm 檢查")
display(pd.DataFrame(rows))


# In[ ]:


# pip install deep_translator openpyxl pandas
from deep_translator import GoogleTranslator
import pandas as pd, hashlib, sqlite3, time

# --- 1.  建快取 DB    -----------------------
conn = sqlite3.connect("trans_cache.sqlite")
conn.execute("""CREATE TABLE IF NOT EXISTS cache
                (h TEXT PRIMARY KEY, en TEXT)""")
translator = GoogleTranslator(source='zh-TW', target='en')

def zh2en(text):
    if not isinstance(text, str):
        return str(text)
    h = hashlib.md5(text.encode()).hexdigest()
    row = conn.execute("SELECT en FROM cache WHERE h=?", (h,)).fetchone()
    if row:           # 命中快取
        return row[0]
    # 呼叫 Google 翻譯
    try:
        en = translator.translate(text)
    except Exception:
        print("[WARN] Google 翻譯失敗，睡 2 秒重試…")
        time.sleep(2)
        try:
            en = translator.translate(text)
        except Exception as e:
            print("仍失敗，回傳原文 >", e)
            en = text
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (h, en))
    conn.commit()
    return en

# --- 2.  讀題庫，加翻譯 ---------------------
df = pd.read_excel("test_questions_withANS.xlsx")

df["question_en"]   = df["question"].apply(zh2en)
df["reference_en"]  = df["reference"].apply(zh2en)  if "reference" in df.columns else ""
df["answers_en"]    = df["answers"].apply(zh2en)    if "answers"   in df.columns else ""

# --- 3.  存檔 --------------------------------
out_path = "test_questions_en.xlsx"
df.to_excel(out_path, index=False)
print("✅ 翻譯完成，已存", out_path)


# #### RAGAS

# https://docs.ragas.io/en/stable/

# In[ ]:


from dotenv import load_dotenv
env_path = Path("key") / ".env"
load_dotenv(dotenv_path=env_path, override=False)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("找不到 OPENAI_API_KEY")


# 跑評分

# In[ ]:


from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI  # langchain>=0.1
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
evaluator_llm = LangchainLLMWrapper(llm)


# In[ ]:


from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)


# In[ ]:


from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness,ContextPrecision

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        ContextPrecision(),
        # Faithfulness(),      #only QA 忠實度
        # FactualCorrectness(), #only QA 正確性
    ],
    llm=evaluator_llm
)


# In[ ]:


result

