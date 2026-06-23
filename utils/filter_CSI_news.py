"""
筛选节点数据中在CSI100指数列表的股票
"""

import pandas as pd
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
import os
import pandas as pd
import ast
#### 从data_processed的数据中筛选新闻中包含CSI100指数中的股票，保存至data_processed/CSI100/nodes

###筛选CSI100中的股票
def filter_news_by_constituents(news_file, constituents_file, output_file=None):
    """
    筛选新闻文件中 ts_code 属于成分股代码的行
    
    Args:
        news_file (str): 新闻文件路径（如 "新闻_20220328.csv"）
        constituents_file (str): 成分股文件路径（如 "CSIA100_normalized.csv"）
        output_file (str, optional): 输出文件路径。如果为 None，则不保存
    
    Returns:
        pd.DataFrame: 筛选后的新闻数据
    """
    # 读取新闻数据
    news_df = pd.read_csv(news_file)
    
    # 读取成分股数据
    constituents_df = pd.read_csv(constituents_file)
    constituent_codes = set(constituents_df['成份券代码Constituent Code'])
    
    # 筛选新闻数据
    filtered_news = news_df[news_df['ts_code'].isin(constituent_codes)]
    #对结果根据ts_code进行去重
    filtered_news=filtered_news.drop_duplicates(subset=['ts_code'])
    
    # 可选：保存结果
    if output_file:
        filtered_news.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"筛选结果已保存到: {output_file}")

    
    return filtered_news

#filter_news_by_constituents("../data_processed/新闻_20220328.csv", '../CSIA100_normalized.csv', output_file="../data_processed/CSI100/news0328.csv")

input_dir="../CSI100/node"
output_dir="../CSI100/node"
for filename in os.listdir(input_dir):
    
        # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        output_filename = filename
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"处理中: {filename} → {output_filename}")
        
        try:
            

            filter_news_by_constituents(input_path, '../CSIA100_normalized.csv', output_path)
            
        except Exception as e:
            print(f"处理文件{filename}时出错: {str(e)}")