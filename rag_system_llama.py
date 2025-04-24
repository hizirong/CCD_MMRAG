#!/usr/bin/env python
# coding: utf-8

# In[50]:


get_ipython().system('jupyter nbconvert --to script rag_system_llama.ipynb')


# ### import

# In[247]:


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
from chromadb.utils import embedding_functions

# Embedding Models
from transformers import CLIPProcessor, CLIPModel
import open_clip


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


# In[248]:


import sys
import torch
import transformers
import accelerate
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")


# ### voice to text
# 

# In[81]:


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


# ### 圖片處理

# In[249]:


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


# ### Embedding 處理模組

# In[250]:


get_ipython().run_line_magic('matplotlib', 'inline')
from transformers import AutoProcessor, AutoModel
import torch
import sentencepiece as spm 

class ClipEmbeddingProcessor:

    MAX_TOKEN = 56          # 56 + BOS + EOS = 58 < 64, 絕對安全
    OVERLAP   = 16
    DEFAULT_COLLECTION = "ccd_docs_siglip"

    # 初始化 embedding processor
    def __init__(self, 
                persist_directory: str = "chroma_db",
                image_dir: str = "image",
                image_size: tuple = (224, 224),
                collection_name:str = DEFAULT_COLLECTION
                ):
        """
        初始化: 建立clip_collection,使用CLIP(or OpenCLIP)做embedding
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_processor = ImageProcessor(image_dir) 
        self.collection_name = collection_name
        
        # ====== 1) 初始化 CLIP ======

        SIGLIP_NAME = "google/siglip-base-patch16-224"   # 也可以換 small / large
        self.processor   = AutoProcessor.from_pretrained(SIGLIP_NAME)
        self.siglip      = AutoModel.from_pretrained(SIGLIP_NAME)  # 沒 GPU 改 "cpu"

        # 測試一下,取得image特徵維度(通常是512)
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224))
            dim_out = self.siglip.get_image_features(dummy).shape[1]
        self.clip_dim = dim_out

        # ====== 2) 建立Chroma ======
        logger.info(f"Initializing Chroma with directory: {persist_directory}")
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # 先刪除舊的 collection(若存在)
        try:
            self.chroma_client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except (chromadb.errors.NotFoundError, ValueError):
            print(f"No collection named {self.collection_name}, nothing to delete.")
            pass

        # 建立無embedding_function的collection,因為我們手動算embedding
        self.clip_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"dimension": self.clip_dim}
        )
        print(f">> Created '{self.collection_name}' in {persist_directory} with dimension={self.clip_dim}.")


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

    def split_into_chunks(
            self,
            text:str,
            max_tokens=MAX_TOKEN, 
            overlap=OVERLAP)-> list[str]:

        ids    = self.processor.tokenizer(text).input_ids
        chunks = []
        step   = max_tokens - overlap
        for i in range(0, len(ids), step):
            seg = ids[i : i + max_tokens]
            chunks.append(
                self.processor.tokenizer.decode(
                    seg, skip_special_tokens=True
                )
            )
        return chunks

    def process_text_with_clip(self, text: str) -> Optional[np.ndarray]:
        """
        用 CLIP 的 text encoder 將文字轉為512維向量
        """
        try:
            chunks = self.split_into_chunks(text)
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
            logger.error(f"Error in process_text_with_clip: {e}")
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
                    txt_emb = self.process_text_with_clip(question_text)

                    if img_emb is not None:
                        all_embeddings.append(img_emb.tolist())
                        # img_embeddings_list = img_embeddings.tolist()
                        # all_metadatas.append(img_embeddings_list)
                        # valid_images.append(img_path)
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

    def process_image_with_clip(self, image_path: str) -> Optional[np.ndarray]:
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
                print(f"Error in process_image_with_clip: {str(e)}")
                return None


    def add_data(self, 
                texts: Optional[List[str]] = None, 
                metadatas: Optional[List[Dict]] = None,
                images: Optional[List[str]] = None):
        """
        統一方法:把text or image加到 clip_collection
        你可以自行拆成add_text/add_image,或像這樣合併都行
        需確保 texts與metadatas數量相同, 或 images與metadatas數量相同(可依需求調整)
        """
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        idx = 0

        if texts is None:
            texts = []
        if images is None:
            images = []
        if metadatas is None:
            metadatas = []

        # 1) 對文字做embedding
        if texts:
            for i, txt in enumerate(texts):
                emb = self.process_text_with_clip(txt)         
                if emb is not None:
                    for vec in self.to_2d(emb):                 # ← 一次可能回傳多條向量
                        all_embeddings.append(vec)
                        all_metadatas.append({
                            "type": "text",
                            "content": txt,
                            **(metadatas[i] if i < len(metadatas) else {})
                        })
                        all_ids.append(f"text_{idx}")
                        idx += 1

        # 2) 對圖片做embedding
        if images:
            for j, img_path in enumerate(images):
                full_path = str(self.image_dir / img_path)
                emb = self.process_image_with_clip(full_path)
                if emb is  None:
                    continue

                for vec in self.to_2d(emb):              # ← 無論 1‑D / 2‑D 都 OK
                    all_embeddings.append(vec)
                    md = {"type": "image", "path": img_path}
                    if metadatas and j < len(metadatas):
                        md.update(metadatas[j])
                    all_metadatas.append(md)
                    all_ids.append(f"img_{idx}")
                    idx += 1


        # 3) 寫入 clip_collection
        if len(all_embeddings) > 0:
            print(np.asarray(all_embeddings).shape)   # 應該是 (N, 512)

            #準備與 embedding 對齊長度的文件列表
            docs = []
            for i, md in enumerate(all_metadatas):
               # 只把圖片排除，其餘一律當文字
                if md.get("type") == "image":
                    docs.append("")                # 圖片沒有文字
                else:
                    docs.append(md.get("content", texts[i] if i < len(texts) else ""))

            assert all(len(doc) > 0 or md["type"] == "image" for doc, md in zip(docs, all_metadatas))

            self.clip_collection.add(
                embeddings = all_embeddings,
                metadatas = all_metadatas,
                documents  = docs, 
                ids = all_ids
            )
            print(f"Added {len(all_embeddings)} items to '{self.collection_name}'.")


    def search(self, query: str, k=25,domain: Optional[str] = None  ) -> Dict:
        """
        對query做CLIP text embedding後,在clip_collection裡找最相似的k筆
        """
        try:
            emb = self.process_text_with_clip(query)
            if emb is None:
                return {"metadatas":[],"documents":[],"distances":[],"ids":[]}
            where_clause = {"domain": domain} if domain else None
        
            results = self.clip_collection.query(
                    query_embeddings=[emb],
                    n_results=k,
                    where=where_clause 
            )
            return results
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {"metadatas":[],"documents":[],"distances":[],"ids":[]}


# ### 資料處理模組

# In[251]:


class DataProcessor:
    def __init__(self, embedding_processor: 'ClipEmbeddingProcessor'):
        self.embedding_processor = embedding_processor
        
    def process_csv_with_images(self, csv_path: str) -> Tuple[List[Dict], List[str]]:
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


    def process_pdf(self, pdf_path: str) -> List[Dict]:
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
                    # elif is_acu:
                    #     paragraphs = self.split_acu_blocks(text)
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
                                    'type': 'professional',
                                    'domain':self.detect_domain(pdf_name),
                                    'source_file': pdf_name,  # 添加文件名
                                    'page': str(page_num + 1),
                                    'content_length': str(len(c))
                                }
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
                social_qa_pairs, images = self.process_csv_with_images(csv_path)
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
                pdf_qa_pairs = self.process_pdf(pdf_path)
                all_professional_pairs.extend(pdf_qa_pairs)
                logger.info(f"\nProcessed {Path(pdf_path).name}:")
                logger.info(f"- Extracted paragraphs: {len(pdf_qa_pairs)}")
            
            # 3. 合并 => all_qa_pairs
            all_qa_pairs = social_qa_pairs + all_professional_pairs
            
            # 4. 準備 texts + metadatas => 你就能一次或多次呼叫 add_data
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
            
            # 输出处理结果
            logger.info(f"\nFinal processing summary:")
            logger.info(f"- Total questions: {len(questions)}")
            logger.info(f"- Total answers: {len(answers)}")
            logger.info(f"- Valid images: {len(valid_images)}")
            logger.info(f"- Social content: {len(social_qa_pairs)} QA pairs")
            logger.info(f"- Professional content: {len(all_professional_pairs)} paragraphs")
            
            # 5. 全部寫進clip_collection
            
            
            # (C) professional paragraphs
            prof_texts  = [qa["answers"][0] for qa in all_professional_pairs]
            prof_metas  = [qa["metadata"]   for qa in all_professional_pairs]

            self.embedding_processor.add_data(texts=prof_texts,
                                            metadatas=prof_metas)
            
            # (A) 先加所有 question
            self.embedding_processor.add_data(
                texts = questions,
                metadatas = question_metas
            )

            # (B) 再加所有 answers
            # self.embedding_processor.add_data(
            #     texts = answers,
            #     metadatas = answer_metas
            # )
            
            # (D) 再加 images
            # 沒有對應metadata？可以簡單做
            # [{"type":"image","source":"facebook"} ...] 或
            # 想知道它屬於哪個QApair? 就要自己對應
            if valid_images:
                meta_for_imgs = []
                for img_name in valid_images:
                    meta_for_imgs.append({
                        "type":"image",
                        "source":"facebook",
                        "filename": img_name
                    })

                self.embedding_processor.add_data(
                    images=valid_images,
                    metadatas=meta_for_imgs
                )

            logger.info("All data added to clip_collection.")
            return len(questions), len(valid_images)
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise


# ### QA系統模組

# ##### code

# In[252]:


from deep_translator import GoogleTranslator
class QASystem:
    def __init__(self, embedding_processor: 'ClipEmbeddingProcessor',
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
                # 你也可以放 'documents':[], 'relevance':[]... 視需要
            },
            "professional": {
                "metadata": [],
                "content": [],
                "documents":[]
                # ...
            },
            "images": {
                "metadata": [],
                "paths": [],
                "relevance":[]
            }
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

                # 判斷 metadata 是屬於哪個來源
                # 例如 meta.get("type") == "social_qa" => 放到 social
                #     meta.get("type") == "professional" => 放到 professional
                #     meta.get("type") == "image" => 放到 images
                src_type = meta.get("type","")

                if src_type in ["social_qa"]:
                    # 純文字 chunk => 放social
                    structured["social"]["metadata"].append(meta)
                    structured["social"]["documents"].append(doc_text)
                    # 若 meta 裡有 link => structured["social"]["link"].append(meta["link"])
                    link_str = meta.get("link","")
                    if link_str:
                        structured["social"]["link"].append(link_str)
                    # content or question
                    if "post_content" in meta:
                        structured["social"]["content"].append(meta["post_content"])

                elif src_type in ["professional"]:
                    structured["professional"]["metadata"].append(meta)
                    structured["professional"]["documents"].append(doc_text)
                    # 可能把段落文字塞到 "content"
                    # (這需要你當初 add_data 時有在 documents/metadata 寫 para)
                    # 這裡只是示例
                    # ...
                
                elif src_type == "image":
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


    def get_prompt_by_type(self,query: str, context: str, question_type: str, references_str: str = "") -> str:
        # print('[DEBUG]:query in get_prompt_by_type',query)
        # logger.info("prompt type:",question_type)
        
        # role
        base_system = (
            "您是一名專業獸醫，擅長：1.犬認知功能障礙綜合症（CCD）的診斷和護理 2.豐富的寵物中醫知識 3.常見問題診斷及改善建議" 
        )

        prompts = {
            "multiple_choice": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                                    {base_system}
                                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                        
                                    這是一題選擇題：{query}

                                    請以以下格式回答：
                                    1. 第一行只能回答 A/B/C/D (請勿帶任何標點、文字)
                                    2. 第二行才是說明，以 2~3 句話簡述理由
                                    3. 若同時出現多個選項，請只選一個最適合的
                                    4. 請在答案最後顯示你參考的來源連結或論文名稱，
                                    如果來源中包含「(經驗) some_link」，請在回答中以 [Experience: some_link] 形式標示；
                                    若包含「(文獻) some.pdf」，就 [reference: some.pdf]。
                                    5.  若有相關圖片，在回答中自然地說明圖片內容跟回答的關係

                                    限制：
                                    - 請將整體回答限制在 200 字以內
                                    - 直接切入重點
                                    - 保持客觀和專業

                                    參考資料：
                                    {context}

                                    來源：
                                    {references_str}

                                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                    """,
            "true_false": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                                {base_system}
                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                問題：{query}

                                參考資料：
                                {context}
                                來源：
                                    {references_str}

                                請按照下列格式回答是非題：
                                Step‑by‑step：
                                1. 先列出判斷依據（可條列）
                                2. 請在答案最後顯示你參考的來源連結或論文名稱，
                                    如果來源中包含「(經驗) some_link」，請在回答中以 [Experience: some_link] 形式標示；
                                    若包含「(文獻) some.pdf」，就 [reference: some.pdf]。
                                3.  若有相關圖片，在回答中自然地說明圖片內容跟回答的關係
                                4. 最後再給出結論，只能寫「True」或「False」


                                限制：
                                - 請將整體回答限制在 200 字以內
                                - 直接切入重點
                                - 保持客觀和專業

                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                """,
            "qa": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                                {base_system}
                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                問題：{query}

                                參考資料：
                                {context}
                                來源：
                                    {references_str}

                                請依以下格式回答：
                                1. 針對問題提供具體答案
                                2. 若遇到無法確定或證據不足的情況可以補充說明研究不足
                                3. 提供實用的建議或解釋
                                4. 請在答案最後顯示你參考的來源連結或論文名稱，
                                    如果來源中包含「(經驗) some_link」，請在回答中以 [Experience: some_link] 形式標示；
                                    若包含「(文獻) some.pdf」，就 [reference: some.pdf]。
                                5. 若有相關圖片，在回答中說明圖片內容

                                限制：
                                - 請將整體回答限制在 400 字以內
                                - 使用平易近人的語言
                                - 避免過度技術性術語

                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                """
        }

        # 若 question_type 未包含在定義的 prompts 中，就預設使用 "qa"。
        return prompts.get(question_type, prompts["qa"])


    def translate_en_to_zh(self,chinese_text: str) -> str:
        try:
            # 指定原文語言為 'zh'（中文），目標語言為 'en'（英文）
            translator = GoogleTranslator(source='en', target='zh-TW')
            result = translator.translate(chinese_text)
            return result
        except Exception as e:
            print(f"翻譯錯誤：{e} - 對應中文問題：{chinese_text}")
            return chinese_text  # 若翻譯失敗，返回原文

    def generate_response(self, query: str,question_type: Optional[str] = None) -> Tuple[str, List[str]]:
        try:
            TARGET_DOMAIN = "中醫方劑" 
            raw_result = self.embedding_processor.search(
                query,
                k=25,
                domain=TARGET_DOMAIN)  
            print(raw_result["metadatas"])
            # raw_result 是 clip_collection 的資料
            # 用後處理
            # search_results = self.embedding_processor.search(raw_result)
            search_results = self._classify_collection_results(raw_result)
            logger.info("SEARCH RESULT(structured): %s",search_results)

            context = self.format_context(search_results)
            references_str = self.gather_references(search_results)
            # link應該用傳參數的會成功 可能用context.link之類的抓題目的reference
            
            zh_query = self.translate_en_to_zh(query)
            

            # --- ① 題型 --------------------------------------------------------
            if question_type:          # 呼叫端已經給我，就直接用
                q_type = question_type
            else:                      # 否則退回舊邏輯自動判斷
                q_type = self.determine_question_type(query)
            # question_type = self.determine_question_type(zh_query)
            prompt = self.get_prompt_by_type(query, context, q_type, references_str)
        
            
            message = {
                'role':'user',
                'content': prompt
            }

            # 處理圖片
            image_paths = []
            # 2) 從 social metadata 把圖片撈出
            for md in search_results["social"]["metadata"]:
                if md.get("images"):  # e.g. "image12.jpg,image02.jpg"
                    for img_name in md["images"].split(","):
                        img_name = img_name.strip()
                        if img_name:
                            full_path = self.embedding_processor.image_dir / img_name
                            if full_path.exists():
                                image_paths.append(str(full_path.resolve()))
            # 3) OLlama 只允許一張, 你可取 image_paths[:1] => message["images"] = ...
            if image_paths:
                print("We found images: ", image_paths)
                # 你可以先隨便取一張
                # or 全部 inject to prompt
            else:
                logger.info("No images to display")

            # 生成响应
            response = ollama.chat(
                model=self.model_name,
                messages=[message]
            )
            return response['message']['content'], image_paths

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
                    response_text, image_paths = self.generate_response(query,question_type)
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


# ### 系統初始化和資料處理

# #### 正式embedding(之後刪)

# In[57]:


from pathlib import Path

# 初始化 embedding processor
embedding_processor = ClipEmbeddingProcessor(
    image_size=(224, 224)  # 設定圖片處理的目標尺寸
)

# 初始化數據處理器
data_processor = DataProcessor(embedding_processor)

# 自動抓取 RAG_data 資料夾中的所有 PDF 檔案
rag_data_dir = Path("RAG_data")
pdf_paths = list(rag_data_dir.glob("*.pdf"))

# 輸出找到的檔案清單（可選）
print(f"找到以下 PDF 檔案：")
for pdf in pdf_paths:
    print(f"  - {pdf}")

# 處理資料
num_texts, num_images = data_processor.process_all(
    csv_path="post_response_filtered.xlsx",#"post_response_v2.csv"
    pdf_paths=pdf_paths
)


# ### 系統測試

# In[64]:


qa_system = QASystem(
    embedding_processor=embedding_processor,
    model_name='llama3.2-vision'
)


# #### 個別題目測試

# In[112]:


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



# #### test questions 

# In[160]:


import string
import re
import pandas as pd
from deep_translator import GoogleTranslator

def translate_zh_to_en(chinese_text: str) -> str:
    try:
        # 指定原文語言為 'zh'（中文），目標語言為 'en'（英文）
        translator = GoogleTranslator(source='zh-TW', target='en')
        result = translator.translate(chinese_text)
        return result
    except Exception as e:
        print(f"翻譯錯誤：{e} - 對應中文問題：{chinese_text}")
        return chinese_text  # 若翻譯失敗，返回原文



def parse_llm_answer(llm_response: str, q_type: str) -> str:
    """
    根據題型 (選擇 or 是非)，從 LLM 的回覆字串中解析出可能的最終答案。
    """

    # 把回覆都轉小寫，以便搜尋
    q_type = q_type.strip().lower()        # 保險起見
    cleaned = llm_response.strip()
    
    
    if q_type == "multiple_choice":
       
        # 1) 先去除可能的全形/半形混雜、移除多餘符號等（可選）
        #    下面先做個最基本的 strip() 處理
        cleaned = llm_response.strip()

        # 2) 建立 Regex：
        #    - `^[ \t]*(A|B|C|D)[ \t]*$`：代表這一行(含前後空白)只有 A/B/C/D
        #    - (?m) 代表 MULTILINE 模式，使 ^ 和 $ 可以匹配每一行的開頭與結尾
        pattern = re.compile(r'^[ \t]*(A|B|C|D)[ \t]*$', re.MULTILINE)

        # 3) 搜尋
        match = pattern.search(cleaned)
        if match:
            # group(1) 會是 'A' or 'B' or 'C' or 'D'
            return match.group(1)
        else:
            return "UNKNOWN"
    
    elif q_type == "true_false":

        for line in reversed(llm_response.splitlines()):
            line = line.strip().lower()
            if line.startswith(("結論", "答案")):
                if "true" in line or "是" in line:
                    return "True"
                if "false" in line or "否" in line or "不" in line:
                    return "False"
                
        negative_phrases = [
            "不是", "否", "不對", "false", "no", "不可以",
            "不能", "不行", "never", "cannot"
        ]
        positive_phrases = [
            "是", "對", "true", "yes", "可以",
            "能", "行", "可以的", "沒問題"
        ]
       # 去掉標點
        text_nopunct = re.sub(f"[{re.escape(string.punctuation)}]", " ", cleaned)

        for phrase in negative_phrases:
            if phrase in text_nopunct:
                return "False"
        for phrase in positive_phrases:
            if phrase in text_nopunct:
                return "True"
        return "UNKNOWN"
    
    else:
        return "UNKNOWN"


def main():
    # 讀取題目資料
    df = pd.read_excel("test_questions.xlsx")
    
    # 篩選 type = multiple_choice 或 true_false 或 qa
    # test_df = df[df["type"].isin(["multiple_choice","true_false"])].copy()
    test_df = df.loc[
        (df["domain"] == "中醫") &
        (df["type"].isin(["multiple_choice", "true_false"]))
    ].copy()
    # test_df = df[df["type"].isin(["true_false"])].copy()
    # test_df = test_df.head(4)
    
    # 新增欄位來存儲系統的回覆 & 預測答案
    test_df["llm_response"] = ""
    test_df["predicted"] = ""
    test_df["is_correct"] = 0
    
    for idx, row in test_df.iterrows():
        q = row["question"]
        en_q = translate_zh_to_en(q)
        q_type = row["type"]
        correct_ans = str(row["answers"]).strip()
        
        # llm_resp = qa_system.display_response(q)
        try:
            response_text, _ = qa_system.display_response(en_q,q_type)

        except Exception as e:
            print(f"Error with query {en_q}: {e}")
            response_text = "No response"
        
        # 解析出預測答案
        # pred_ans = parse_llm_answer(llm_resp, q_type)
        pred_ans = parse_llm_answer(response_text, q_type)
        
        # 比對正確答案
        # 為保險，正確答案也 upper 或 lower 下來比較
        is_correct = 1 if pred_ans.upper() == correct_ans.upper() else 0
        
        # 寫回 DataFrame
        test_df.at[idx, "llm_response"] = response_text
        test_df.at[idx, "predicted"] = pred_ans
        test_df.at[idx, "is_correct"] = is_correct
    
    # 計算 Accuracy
    total = len(test_df)
    correct_count = test_df["is_correct"].sum()
    accuracy = correct_count / total if total>0 else 0.0
    
    print("=== 測試結果 ===")
    print(test_df[["id","type","answers","predicted","is_correct"]])
    print(f"\n共 {total} 題，正確 {correct_count} 題，Accuracy = {accuracy:.2f}")
    
    # 若需要將回覆結果輸出 CSV 
    test_df.to_csv("test_result.csv", index=False, encoding='utf-8')
    print("結果已儲存 test_result.csv")

if __name__ == "__main__":
    main()


# #### test data

# In[253]:


from pathlib import Path
TEST_MODE = True                           # ← 切換開關
COLLECTION_NAME = "clip_collection_test_v3"   # 測試用向量庫

# 1) 初始化 embedding_processor，傳入新的 collection_name
embedding_processor = ClipEmbeddingProcessor(
    image_size=(224, 224) ,
    collection_name=COLLECTION_NAME    # ★若 __init__ 沒這參數，改下方註解方法
)

# 2) 初始化資料處理器
data_processor = DataProcessor(embedding_processor)

# 3) 指定測試或正式資料夾
rag_data_dir = Path("RAG_data_test" if TEST_MODE else "RAG_data")
pdf_paths = list(rag_data_dir.glob("*.pdf"))


print("找到以下 PDF：")
for p in pdf_paths: print(" -", p.name)

# 4) 處理資料 （CSV 你可以傳 None 代表不處理社群資料）
_ = data_processor.process_all(
    csv_path=None,           # 只測 PDF，可先不管社群
    pdf_paths=pdf_paths
)


# In[224]:


sample = embedding_processor.clip_collection.get(include=["documents","metadatas"], limit=1)
print("DOC 前 120 字:", sample["documents"][0][:120])


# In[254]:


# 建立 QA 系統，沿用同一個 embedding_processor
qa_system = QASystem(
    embedding_processor=embedding_processor,
    model_name='llama3.2-vision'
)
TARGET_DOMAIN = "中醫方劑"   # 想測哪個就填哪個
qa_system.TARGET_DOMAIN = TARGET_DOMAIN   # 若你寫成屬性


# In[257]:


# 1. 讀檔 + 題型篩選
df = pd.read_excel("test_questions.xlsx")
# test_df = df[df["type"].isin(["multiple_choice", "true_false"])].copy()
test_df = df[df["type"].isin(["multiple_choice"])].copy()

# 2. ★ 建立欄位（一定要在後面的篩選前先加）
test_df["llm_response"] = ""
test_df["predicted"]    = ""
test_df["is_correct"]   = 0

# 3. 再依 domain 篩子集合
test_df = test_df[test_df["domain"] == "中醫"].copy()
test_df=test_df.head(3)

# 4. 迴圈計分
for idx, row in test_df.iterrows():
    q  = row["question"]
    q_type = row["type"]
    gt = str(row["answers"]).strip()

    resp, _ = qa_system.display_response(q, q_type)
    pred = parse_llm_answer(resp, q_type)

    test_df.at[idx, "llm_response"] = resp
    test_df.at[idx, "predicted"]    = pred
    test_df.at[idx, "is_correct"]   = int(pred.upper() == gt.upper())

# 5. 計算 Accuracy
accuracy = test_df["is_correct"].mean()
print(test_df[["id","type","answers","predicted","is_correct"]])
print(f"[{TARGET_DOMAIN}] Accuracy = {accuracy:.2%}")

