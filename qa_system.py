from embedding import ClipEmbeddingProcessor
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