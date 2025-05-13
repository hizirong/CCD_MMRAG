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
                all_ids.append(str(uuid.uuid4())) #(f"text_{idx}")
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
            src_meta.pop("type", None) 

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
            # q = query.lower()
            # if re.search(r"(st|cv|gv|bl|pc)-\d{1,2}|穴位", q):
            #     weight = {"herb": 0.3, "ccd": 0.3}     # acupoint = 1.0
            # elif any(w in q for w in ["柴胡", "黃芩", "清熱"]):
            #     weight = {"acupoint": 0.3, "ccd": 0.3}
            # elif any(w in q for w in ["認知", "nlrp3", "發炎"]):
            #     weight = {"herb": 0.3, "acupoint": 0.3}
            # else:
            #     weight = {}

            # metas = results["metadatas"][0]
            # dists = results["distances"][0]
            # docs  = results["documents"][0]

            # scored = []
            # for i, (m, d) in enumerate(zip(metas, dists)):
            #     w = weight.get(m.get("type", ""), 1.0)
            #     scored.append((d * w, i))          # 距離愈小愈好
            # scored.sort(key=lambda x: x[0])

            # idxs = [i for _, i in scored][:k]       # 取前 k

            #-----------不降權用的-----------------
            metas = results["metadatas"][0]
            idxs  = list(range(len(metas)))[:k]
            #-------------------------------------
            
            for key in ["metadatas", "distances", "documents"]:
                results[key][0] = [results[key][0][i] for i in idxs]

                return results
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {"metadatas":[],"documents":[],"distances":[]}

    # ------------------------------------------------------------
    #  新增：similarity_search_with_images()
    # ------------------------------------------------------------
    def similarity_search_with_images(
        self,
        query: str,
        k_mix: int = 40,
        img_threshold: float = 0.35,
        max_images: int = 2,
    ):
        """
        先取前 k_mix 名混合結果，再把距離 < img_threshold 的
        image / caption 補到結果中（最多 max_images 張）。
        回傳值格式與 similarity_search 相同。
        """
        emb = self.encode_text_to_vec(query)
        if emb is None:
            return {"metadatas": [], "documents": [], "distances": []}

        # ① 混合前 k_mix
        raw = self.clip_collection.query(
            query_embeddings=[emb],
            n_results=k_mix,
            include=["metadatas", "documents", "distances"],
        )

        # ② 把圖片候選抓出來（距離小於門檻）
        metas    = raw["metadatas"][0]
        dists    = raw["distances"][0]
        docs     = raw["documents"][0]
        good_idx = [
            i for i, (m, d) in enumerate(zip(metas, dists))
            if m.get("type") in ("image", "images", "caption") and d < img_threshold
        ]

        # ③ 若不足 max_images，再專搜圖片補足
        if len(good_idx) < max_images:
            need   = max_images - len(good_idx)
            imgraw = self.clip_collection.query(
                query_embeddings=[emb],
                n_results=need,
                where={"type": {"$in": ["image", "images", "caption"]}},
                include=["metadatas", "documents", "distances"],
            )
            for key in ["metadatas", "documents", "distances"]:
                raw[key][0].extend(imgraw[key][0])

        return raw

    
