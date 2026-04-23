import requests
import bs4

def get(url):
        # 1.设置请求头，请求网页，并打印状态码
        try:
            headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(url,headers=headers, timeout=10)
            response.encoding = 'utf-8'
            print(f'请求：{url}')
            print(f'响应状态码{response.status_code}')

            if response.status_code != 200:
                print(f'请求：{url}失败')
                return []
            #2.使用BeautifulSoup解析HTML
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            #3.获取所有新闻标题及对应链接，即返回统一的字典列表，包含text和url两个键
            news_list = []
#soup.find_all返回的是bs tag列表，需要转换成字典，先创建空字典方便后续返回数据的储存
            news_links = soup.find_all('a', href=True)#href定义超链接的目标地址

            for news in news_links:
                href = news['href']#直接索引，无属性时报错，使用href = tag.get('href', '')无属性返回默认值
                text = news.get_text(strip=True)
                
                if text and len(text) > 5 and len(text) < 100:#设置过滤条件

                    if href.startswith('/'):
                        full_url = 'https://www.northnews.cn' + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue  # 跳过无效链接
                
                    # 避免重复
                    if not any(news['url'] == full_url for news in news_list):
                        news_list.append({'text': text,'url': full_url})
                        print(f'标题：{text}，链接：{href}')

            #4.返回前10条新闻标题及链接
            return news_list[:10]
        except Exception as e:
            print(f'请求：{url}失败')
            return []

            #5.提取的数据保存到JSON文件 
import json
import os

# 打印的json文件将保存到桌面
os.chdir(os.path.expanduser("~"))
print(f"工作目录已切换至: {os.getcwd()}")#保存到用户主目录，保证可写

def save_to_json(data,filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:#写入
            json.dump(data, f, ensure_ascii=False, indent=2)
        print('数据保存成功')
        return True
    except Exception as e:
        print(f'数据保存失败:{e}')#打印具体错误信息
        return False
    
url = "https://www.northnews.cn/"
print(f"\n目标网站：{url}")
print("\n开始提取新闻列表...")
news_list = get(url)

if news_list:
    print(f"\n成功提取 {len(news_list)} 条新闻：")
    
    for i, news in enumerate(news_list, 1):
        print(f"{i}. {news['text']}")
        print(f"   链接：{news['url']}")
    
    # 保存到JSON文件
    print("正在保存数据...")
    save_to_json(news_list, 'northnews_news.json')
    
    # 验证保存的文件
    print("\n验证保存的文件：")
    try:
        with open('northnews_news.json', 'r', encoding='utf-8') as f:#读取
            loaded_data = json.load(f)
        print(f"成功读取文件，包含 {len(loaded_data)} 条新闻")
    except Exception as e:
        print(f"读取文件失败：{e}")
else:
    print("\n未能提取到新闻列表")