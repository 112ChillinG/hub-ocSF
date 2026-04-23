import requests
import bs4
import json
import os
import time

class NewsCrawler:
    def __init__(self, base_url = 'https://www.northnews.cn'):
        self.base_url = base_url
        self.session = requests.Session()
        #self.session.headers.update后续的请求都自动携带这个请求头，无需手动传递
        self.session.headers.update = ({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
        
    def get_category_urls(self):
        categories = {
            '首页': f'{self.base_url}/',
            '内蒙古': f'{self.base_url}/news/neimenggu/',
            '国内': f'{self.base_url}/news/guonei/',
            '国际': f'{self.base_url}/news/guoji/',
            '财经': f'{self.base_url}/news/caijing/',
            '体育': f'{self.base_url}/news/tiyu/',
            '文化': f'{self.base_url}/news/wenhua/',
        }
        return categories

    def get_news_list(self, category_url):
        news_links = []
        
        try:
            response = self.session.get(category_url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"请求失败，状态码：{response.status_code}")
                return news_links
            
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # 过滤条件
                if text and len(text) > 5 and len(text) < 100:
                    # 处理链接
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # 避免重复
                    if not any(item['url'] == full_url for item in news_links):
                        news_links.append({
                            'url': full_url,
                            'title': text
                        })
        
        except Exception as e:
            print(f"获取新闻列表时出错：{type(e).__name__}: {e}")
        
        return news_links

    def get_news_content(self, title, news_url):
        try:
            response = self.session.get(news_url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                return None
            
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            content = None

            content_selectors = [
                ('div', {'class': 'content'}),
                ('div', {'class': 'article-content'}),
                ('div', {'class': 'news-content'}),
                ('div', {'id': 'content'}),
                ('article', {}),
            ]
            
            for tag, attrs in content_selectors:
                content_elem = soup.find(tag, attrs)
                if content_elem:
                    content = content_elem.get_text(strip=True)
                    break
            
            # 如果没有找到特定容器，尝试提取所有段落
            if not content:
                paragraphs = soup.find_all('p')
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # 清理内容
            #if content:
            #    content = self._clean_content(content)
            
            return {
                'title': title,
                'content': content if content else "无法获取内容",
                'url': news_url
            }
        
        except Exception as e:
            print(f"获取新闻内容出错 {news_url}: {type(e).__name__}: {e}")
            return None

    def crawl_category(self, category_name, category_url, max_news=10):
        print(f"\n开始爬取分类: {category_name}")

        news_links = self.get_news_list(category_url)[:max_news]
        print(f"找到 {len(news_links)} 条新闻链接，将爬取前 {max_news} 条")
        
        news_with_content = []
        for idx, link_info in enumerate(news_links, 1):
            print(f"  正在爬取 [{idx}/{len(news_links)}]: {link_info['title']}")
            content_data = self.get_news_content(link_info['title'], link_info['url'])
            if content_data:
                news_with_content.append(content_data)
            time.sleep(1)  # 礼貌爬取，避免过快

        print(f"成功获取 {len(news_with_content)} 篇新闻内容")
        return news_with_content

    def crawl_all_categories(self, max_news_per_category=5):
        categories = self.get_category_urls()
        all_data = {}#字典

        for category_name, category_url in categories.items():
            news_data = self.crawl_category(category_name, category_url, max_news_per_category)
            if news_data:#非空才存入
                all_data[category_name] = news_data
            
            # 分类之间稍作停顿
            if category_name != list(categories.keys())[-1]:
                time.sleep(2)

        return all_data


    def save_data(self, data, filename='northnews_data.json'):
        home_dir = os.path.expanduser("~")
        full_path = os.path.join(home_dir, filename)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n数据已保存到 {full_path}")
            return True
        except Exception as e:
            print(f"保存数据失败：{type(e).__name__}: {e}")
            return False

# 主程序
if __name__ == '__main__':
    # 创建爬虫实例
    crawler = NewsCrawler()
    
    # 方式1：爬取所有分类（每个分类限制数量）
    print("\n方式1：爬取所有分类的新闻")
    all_news = crawler.crawl_all_categories(max_news_per_category=3)
    
    # 保存数据
    if all_news:
        crawler.save_data(all_news, 'northnews_all_data.json')

        total_news = 0
        for category, news_list in all_news.items():
            count = len(news_list)
            total_news += count
            print(f"{category}：{count} 篇")
        
        print(f"\n总共爬取了 {total_news} 篇新闻")
        print(f"分类数量：{len(all_news)}")
    
    # 方式2：爬取单个分类
    print("\n\n方式2：爬取单个分类的新闻")
    categories = crawler.get_category_urls()
    test_category = list(categories.keys())[0]  # 选择第一个分类作为示例
    test_url = categories[test_category]
    
    single_category_news = crawler.crawl_category(test_category, test_url, max_news=5)
    if single_category_news:
        crawler.save_data({test_category: single_category_news}, f'northnews_{test_category}.json')