import requests
import json
import os
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor

class OpenDiggerURLFetcher:
    def __init__(self):
        self.metrics_list = [
            "openrank", "activity", "attention", "active_dates_and_times",
            "stars", "technical_fork", "participants", "new_contributors",
            "inactive_contributors", "bus_factor", "issues_new", "issues_closed",
            "issue_comments", "pr_new", "pr_merged", "pr_reviews",
            "merged_code_addition", "merged_code_deletion", "line_of_code_changed"
        ]

    def parse_github_url(self, url):
        """
        从 GitHub 网址中提取 org 和 repo
        支持格式: 
        - https://github.com/org/repo
        - https://github.com/org/repo/tree/main
        - org/repo
        """
        url = url.strip()
        # 处理完整的 URL
        if "github.com/" in url:
            # 使用正则匹配域名后的前两个路径段
            match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
            if match:
                org = match.group(1)
                repo = match.group(2).replace(".git", "") # 去掉可能存在的 .git 后缀
                return org, repo
        # 处理直接输入的 org/repo 格式
        elif "/" in url:
            parts = url.split('/')
            return parts[0], parts[1]
        
        return None, None

    def fetch_single_metric(self, org, repo, metric):
        """获取单个指标的 JSON 内容"""
        base_url = f"https://oss.open-digger.cn/github/{org}/{repo}/"
        url = f"{base_url}{metric}.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return metric, response.json()
        except:
            pass
        return metric, None

    def get_all_metrics(self, org, repo):
        """使用多线程加速获取"""
        print(f"\n🚀 正在为仓库 [{org}/{repo}] 检索 OpenDigger 指标...")
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 传递 org 和 repo 参数给抓取函数
            task_results = list(executor.map(lambda m: self.fetch_single_metric(org, repo, m), self.metrics_list))
            
        for metric, data in task_results:
            if data:
                results[metric] = data
                print(f" ✅ {metric}")
        return results

    def export_data(self, org, repo, data):
        """保存为 JSON 和 CSV"""
        if not data:
            print("❌ 未能获取到任何有效指标。")
            return

        # 1. 保存原始 JSON
        json_name = f"{org}_{repo}_raw.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # 2. 转换为月度汇总 CSV
        all_series = {}
        for metric, content in data.items():
            # 仅保留 YYYY-MM 格式的月度数据
            monthly_values = {k: v for k, v in content.items() if re.match(r'^\d{4}-\d{2}$', str(k))}
            if monthly_values:
                all_series[metric] = pd.Series(monthly_values)
        
        if all_series:
            df = pd.DataFrame(all_series).sort_index()
            csv_name = f"{org}_{repo}_summary.csv"
            df.to_csv(csv_name, encoding='utf-8-sig')
            print(f"\n💾 数据已保存：\n - 原始数据: {json_name}\n - 汇总报表: {csv_name}")
            print("\n--- 最近 3 个月数据预览 ---")
            print(df.tail(3))

def main():
    fetcher = OpenDiggerURLFetcher()
    user_input = input("请输入 GitHub 项目网址 (例如 https://github.com/pingcap/tidb): ")
    
    org, repo = fetcher.parse_github_url(user_input)
    
    if org and repo:
        full_data = fetcher.get_all_metrics(org, repo)
        fetcher.export_data(org, repo, full_data)
    else:
        print("❌ 无法解析网址。请确保输入正确的 GitHub 地址，如 https://github.com/组织/仓库")

if __name__ == "__main__":
    main()