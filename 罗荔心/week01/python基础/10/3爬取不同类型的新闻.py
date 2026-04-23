import requests
import bs4
import time
import json
import os

def get_category(url,catagory_name):#增加分类参数
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
            
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            news_list = []

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
                        #以上服用作业2，都需要获取url和text
                        news_list.append({'category': category_name,'url': full_url, 'title': text})
                    print(f"找到 {len(news_list)} 条新闻")
                    return news_list[:20]
        except Exception as e:
            print(f"爬取分类时出错，错误信息：{e}")
            return []
        
#每个分类分别保存到不同的json文件中
os.chdir(os.path.expanduser("~"))
print(f"工作目录已切换至: {os.getcwd()}")#保存到用户主目录，保证可写
def save_category_data(news_list, category_name):
#保存分类数据到JSON文件
    if not news_list:
        print(f"分类 {category_name} 没有数据可保存")
        return False
    
    filename = f'northnews_{category_name}.json'
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        print(f"分类 {category_name} 的数据已保存到 {filename}")
        return True
    except Exception as e:
        print(f"保存分类 {category_name} 数据失败：{type(e).__name__}: {e}")
        return False
    
#主函数
categories = {
    '首页': 'https://www.northnews.cn/',
    '国内': 'https://www.northnews.cn/news/guonei/',
    '国际': 'https://www.northnews.cn/news/guoji/',
}

print(f"\n准备爬取 {len(categories)} 个分类")

all_results = {}

for category_name, url in categories.items():
    # 爬取分类
    news_list = get_category(url, category_name)
    
    if news_list:
        all_results[category_name] = news_list
        # 保存每个分类的数据
        save_category_data(news_list, category_name)
    
    # 添加延时，避免请求过快
    if category_name != list(categories.keys())[-1]:  # 最后一个不需要延时
        print("等待1秒后继续...")
        time.sleep(1)

total_news = 0
for category_name, news_list in all_results.items():
    count = len(news_list)
    total_news += count
    print(f"{category_name}：{count} 条")

print(f"\n总计：{total_news} 条新闻")

# 保存所有数据到一个文件
all_data_file = 'northnews_all_categories.json'
try:
    with open(all_data_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n所有分类数据已保存到 {all_data_file}")
except Exception as e:
    print(f"保存总数据失败：{type(e).__name__}: {e}")
