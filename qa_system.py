from embedding import EmbeddingProcessor
from logger import logger
from reranker import rerank
from typing import List, Dict, Optional, Tuple
import re
import ollama
from PIL import Image as PILImage 
from IPython.display import display
from IPython.display import Image as IPyImage 

from deep_translator import GoogleTranslator


class QASystem:
    def __init__(self, embedding_processor: 'EmbeddingProcessor',
                 model_name: str = 'llama3.2-vision'):
        self.embedding_processor = embedding_processor
        self.model_name = model_name
        logger.info(f"Initialized QA System with Ollama model: {model_name}")

    def _classify_collection_results(self, raw_result) -> Dict:

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
                "relevance":[],
                "description":[]
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
                elif src_type in ("image", "images", "caption"):
                    structured["images"]["metadata"].append(meta)
                    path = meta.get("path") or meta.get("ref_image", "") or ""
                    structured["images"]["paths"].append(path)
                    structured["images"]["relevance"].append(dist)
                    doc_text = meta.get("content", "")
                    structured["images"]["description"].append(doc_text)

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
            # raw_result = self.embedding_processor.similarity_search(
            #     query,
            #     k=25) 
            raw_result = self.embedding_processor.similarity_search_with_images(
                 query, k_mix=40, img_threshold=0.35, max_images=2)
 

            # print(raw_result["metadatas"])
            # logger.info("raw RESULT(structured): %s",raw_result)
            
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
            # 取前 2 張圖的 caption
            caption_snips = [
                m.get("content", "")[:120]
                for m in search_results["images"]["metadata"][:1]
                if m.get("content")
            ]
            caption_block = "\n".join(caption_snips)

            user_prompt = self.build_user_prompt(
                query=query,
                context=caption_block + "\n" + context[:1500],
                references_str=references_str
)

            # user_prompt = self.build_user_prompt(
            #     query=query,
            #     context=context[:1500],
            #     references_str=references_str
            # )

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
                    "針對問題提供具體答案並詳細說明 \n"
                )

            #"您是一名專業獸醫，1.擅長犬認知功能障礙綜合症（CCD）的診斷和護理 2.擁有豐富的寵物中醫知識 3.常見問題診斷及改善建議\n" "請在答案最後顯示你參考的來源連結或論文名稱，如果來源中包含「(經驗) some_link」，請在回答中以 [Experience: some_link] 形式標示；若包含「(文獻) some.pdf」，就 [reference: some.pdf]\n""如檢索結果仍無相關資訊，請以[NoRef]標示並根據你的常識回答。\n"
            system_prompt = (
                """你是資深獸醫，擅長犬認知功能障礙綜合症（CCD）的診斷和護理並擁有豐富的寵物中醫知識，必須遵守以下規則回答問題：
                    1. 先根據【檢索結果】內容作答;若資訊不足，才可依照你的常識回答。
                    2. 若檢索結果中含有圖片，在回答中自然的帶入圖片說明
                    3. 若需補充一般臨床常識，請將該句放在段落最後並標註［常識］。
                    4. 每一句結尾必須標註引用來源編號，如［1］或［1,3］。
                    5. 並在最後面整理列出每個編號的source_file，如[1] ...pdf 或 [2] Chinese Veterinary Materia Medica """
                + format_rules
            )
        

            # 處理圖片
            image_paths = []
            # for meta in search_results.get("images", {}).get("metadata", []):
            #     if meta.get("type") in ("image", "images", "caption", "herb_img"):
            #         p = meta.get("path") or meta.get("ref_image")
            #         if p:
            #             logger.info("got image ib a!", p)
            #             full = self.embedding_processor.image_dir / p
            #             if full.exists():
            #                 image_paths.append(str(full.resolve()))

            # -- (b) 保留舊的 social.images 規則 (若還需要) --
            for md in search_results.get("social", {}).get("metadata", []):
                if md.get("images"):
                    for img_name in md["images"].split(","):
                        full_path = self.embedding_processor.image_dir / img_name.strip()
                        if full_path.exists():
                            image_paths.append(str(full_path.resolve()))
            for mm in search_results.get("images", {}).get("metadata", []):#["images"]["metadata"]:
                p = mm.get("path") or mm.get("ref_image")
                if p and (self.embedding_processor.image_dir / p).exists():
                    image_paths.append(str((self.embedding_processor.image_dir / p).resolve()))
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