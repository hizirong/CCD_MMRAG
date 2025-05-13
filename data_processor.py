from embedding import EmbeddingProcessor
from logger import logger
from typing import List, Dict, Tuple
import PyPDF2
from pathlib import Path
import pandas as pd
import re

TYPE_MAP = {
    "acupoint"    : ["針灸", "acupuncture"],
    "herb"        : ["herbology", "herbal", "方劑"],
    "ccd"         : ["ccd", "認知", "cognition"],
    "social"      : [],                   # csv 直接指定
    "professional": [],
    "image":[]                                       # 其他未分類
}


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
