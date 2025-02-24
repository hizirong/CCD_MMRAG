from typing import List
import requests
import json

class LlamaQASystem:
    def __init__(self, model_name: str = "llama3.2-vision", api_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
        
    def get_response(self, query: str) -> str:
        # 準備請求的資料
        data = {
            "model": self.model_name,
            "prompt": query,
            "stream": False  # 設為 False 以取得完整回應
        }
        
        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()  # 檢查是否有錯誤狀態碼
            result = response.json()
            return result["response"]
            
        except requests.exceptions.RequestException as e:
            return f"發生錯誤: {str(e)}"
    
    def display_response(self, query: str) -> None:
        print(f"\n問題: {query}")
        # print("-" * 50)
        response = self.get_response(query)
        print(f"回答: {response}")
        print("-" * 50)

def main():
    # 初始化 QA 系統
    qa_system = LlamaQASystem()
    
    # 測試查詢
    test_queries = [
        "CCD(Canine Cognitive Dysfunction) 是否與神經發炎相關？有無特定細胞因子（cytokines）或發炎路徑（例如NLRP3 inflammasome）參與？",
"CCD(Canine Cognitive Dysfunction) 是否與腸道微生物群變化有關？是否有特定細菌群落會影響大腦健康？",
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

# "問題一：根據資料中對犬認知功能障礙（CCD）神經發炎機制的探討，NLRP3炎症小體在分子層面上如何參與CCD進程？該過程涉及哪些關鍵細胞因子與調控機制？",
# "問題二：資料提到腸道微生物群與CCD之間可能存在聯繫，請問文中如何闡述腸道菌群失衡影響神經傳導與免疫反應的分子機制？哪些特定細菌群落的變化被認為與CCD進展相關？",
# "問題三：在探討CCD的診斷策略中，該資料對於利用影像學技術（如MRI與CT）區分CCD與其他神經退行性疾病的應用提出了哪些見解？這些技術的優勢與局限性分別是什麼？",
# "問題四：資料中對失智犬松果體退化與褪黑激素分泌減少之間的關聯有詳細論述，請問該研究如何描述這一生理變化的分子機制以及其對犬隻睡眠-覺醒週期的影響？",
# "問題五：針對CCD的治療策略，資料中提出了哪些基於分子機制的治療方法？請分析這些方法在臨床應用上的現狀、潛在優勢及未來研究中亟待解決的挑戰。",

# "哪種犬容易失智？",
# "大中小型狗的失智照顧方式有什麼不同？"
    ]
    
    # 執行測試
    for query in test_queries:
        qa_system.display_response(query)

if __name__ == "__main__":
    main()