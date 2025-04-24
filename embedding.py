from image_processor import ImageProcessor
import chromadb   
from config import CHROMA_DIR, SIGLIP_NAME, DEVICE, EMBED_DIM,PERSIST_DIR
from logger import logger
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image as PILImage 
from typing import Union   

import numpy as np

from transformers import AutoProcessor, AutoModel
import torch


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
