from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
import json
from datetime import datetime
import urllib.request
from selenium.webdriver.chrome.options import Options

class FacebookScraper:
    def __init__(self, email=None, password=None, cookies_file=None):
        self.email = email
        self.password = password
        
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-notifications')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        # 如果提供了cookies文件，則優先使用cookies
        if cookies_file and os.path.exists(cookies_file):
            self.load_cookies(cookies_file)
        elif email and password:
            self.login()
        else:
            print("警告：未提供cookies或登入資訊")
    
    def login(self):
        """登入Facebook"""
        print("正在進行登入...")
        try:
            self.driver.get("https://www.facebook.com")
            
            # 等待登入表單元素出現
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form[action*='/login/']"))
            )
            
            # 找到email和password輸入框
            email_field = self.driver.find_element(By.CSS_SELECTOR, "input[name='email']")
            password_field = self.driver.find_element(By.CSS_SELECTOR, "input[name='pass']")
            
            # 清空輸入框
            email_field.clear()
            password_field.clear()
            
            # 輸入帳號密碼
            print("輸入帳號...")
            email_field.send_keys(self.email)
            time.sleep(1)  # 稍微等待
            
            print("輸入密碼...")
            password_field.send_keys(self.password)
            time.sleep(1)  # 稍微等待
            
            # 找到並點擊登入按鈕
            login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], [role='button']")
            print("點擊登入按鈕...")
            login_button.click()
            
            # 等待登入完成
            time.sleep(5)
            
            # 檢查是否登入成功
            if "login" not in self.driver.current_url:
                print("登入成功！")
                return True
            else:
                print("登入可能失敗，請檢查帳號密碼是否正確")
                return False
                
        except Exception as e:
            print(f"登入過程中發生錯誤: {str(e)}")
            return False
    
    def save_cookies(self, file_path="facebook_cookies.json"):
        """保存目前的cookies到文件"""
        with open(file_path, 'w') as file:
            json.dump(self.driver.get_cookies(), file)
        print(f"Cookies已保存到 {file_path}")
    
    def load_cookies(self, file_path):
        """從文件載入cookies"""
        print("正在載入cookies...")
        self.driver.get("https://www.facebook.com")
        time.sleep(3)
        
        try:
            with open(file_path, 'r') as file:
                cookies = json.load(file)
                for cookie in cookies:
                    try:
                        self.driver.add_cookie(cookie)
                    except:
                        pass
            print("Cookies載入完成")
            
            # 重新整理頁面以應用cookies
            self.driver.refresh()
            time.sleep(3)
            
            # 檢查是否需要重新登入
            if "login" in self.driver.current_url and self.email and self.password:
                print("Cookies已過期，嘗試重新登入...")
                return self.login()
            
            return True
        except Exception as e:
            print(f"載入cookies時發生錯誤: {str(e)}")
            if self.email and self.password:
                print("嘗試使用帳號密碼登入...")
                return self.login()
            return False
    
    def scrape_group(self, group_url, num_posts=10):
        """爬取社團內容"""
        try:
            print("正在載入頁面...")
            self.driver.get(group_url)
            
            # 檢查是否需要登入
            if "login" in self.driver.current_url:
                if self.email and self.password:
                    print("需要重新登入...")
                    if not self.login():
                        return []
                    self.driver.get(group_url)
                else:
                    print("無法訪問頁面，需要登入憑證")
                    return []
            
            print("頁面載入完成")
            
            # 等待頁面載入
            time.sleep(5)
            posts_data = []
            
            # 滾動頁面載入更多貼文
            print(f"準備載入 {num_posts} 篇貼文...")
            for i in range(num_posts // 5):
                print(f"正在滾動頁面... ({i+1}/{num_posts//5})")
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
            
            # 找到所有貼文
            print("開始擷取貼文...")
            posts = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="post_container"]')
            
            for i, post in enumerate(posts[:num_posts], 1):
                try:
                    print(f"正在處理第 {i}/{num_posts} 篇貼文")
                    
                    # 取得貼文時間
                    timestamp = post.find_element(By.CSS_SELECTOR, "a[href*='/groups/'][role='link']").get_attribute("aria-label")
                    
                    # 取得貼文內容
                    try:
                        content = post.find_element(By.CSS_SELECTOR, '[data-ad-preview="message"]').text
                    except:
                        content = "無法擷取貼文內容"
                    
                    # 取得分享連結
                    try:
                        share_button = post.find_element(By.CSS_SELECTOR, '[aria-label="分享"]')
                        share_button.click()
                        time.sleep(1)
                        share_link = self.driver.find_element(By.CSS_SELECTOR, '[aria-label="複製連結"]').get_attribute("href")
                        # 關閉分享選單
                        self.driver.find_element(By.CSS_SELECTOR, '[aria-label="關閉"]').click()
                    except:
                        share_link = "無法取得分享連結"
                    
                    # 取得圖片(如果有的話)
                    images = []
                    try:
                        img_elements = post.find_elements(By.CSS_SELECTOR, 'img[src*="https"]')
                        for img in img_elements:
                            img_url = img.get_attribute("src")
                            if img_url and "https" in img_url and not "emoji" in img_url.lower():
                                images.append(img_url)
                    except:
                        pass
                    
                    # 取得留言
                    comments = []
                    try:
                        comment_elements = post.find_elements(By.CSS_SELECTOR, '[data-testid="comment"]')
                        for comment in comment_elements:
                            comment_text = comment.text
                            if comment_text:
                                comments.append(comment_text)
                    except:
                        pass
                    
                    posts_data.append({
                        "時間": timestamp,
                        "內容": content,
                        "分享連結": share_link,
                        "圖片網址": "|".join(images),
                        "留言": "|".join(comments)
                    })
                    
                except Exception as e:
                    print(f"處理貼文時發生錯誤: {str(e)}")
                    continue
            
            return posts_data
            
        except Exception as e:
            print(f"爬取過程中發生錯誤: {str(e)}")
            return []
    
    def save_to_excel(self, posts_data, output_file="fb_group_data.xlsx"):
        """將資料儲存至Excel"""
        if not posts_data:
            print("沒有資料可以儲存")
            return
            
        print("正在儲存資料到Excel...")
        df = pd.DataFrame(posts_data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"資料已儲存至 {output_file}")
        
    def download_images(self, posts_data, image_folder="images"):
        """下載貼文中的圖片"""
        if not posts_data:
            print("沒有圖片可以下載")
            return
            
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
            
        print("開始下載圖片...")
        for i, post in enumerate(posts_data):
            if post["圖片網址"]:
                for j, img_url in enumerate(post["圖片網址"].split("|")):
                    try:
                        img_name = f"post_{i}_img_{j}.jpg"
                        print(f"正在下載圖片: {img_name}")
                        urllib.request.urlretrieve(img_url, os.path.join(image_folder, img_name))
                    except:
                        print(f"下載圖片失敗: {img_url}")
        print("圖片下載完成")
    
    def close(self):
        """關閉瀏覽器"""
        print("關閉瀏覽器...")
        self.driver.quit()

def main():
    # 設定Facebook帳號密碼（可選）
    email = "0988245274"
    password = "Abcdefgh88125"
    
    # cookies檔案路徑（可選）
    cookies_file = "facebook_cookies.json"
    
    # FB社團網址
    group_url = "https://www.facebook.com/groups/423579508162626/"
    
    # 初始化爬蟲（可以選擇是否提供帳密和cookies）
    scraper = FacebookScraper(
        email=email, 
        password=password,
        cookies_file=cookies_file
    )
    
    try:
        # 如果登入成功，可以保存cookies供下次使用
        scraper.save_cookies()
        
        posts_data = scraper.scrape_group(group_url, num_posts=20)
        scraper.save_to_excel(posts_data)
        scraper.download_images(posts_data)
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main()