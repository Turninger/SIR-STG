"""
数据处理第一步
"""

### 构建边数据集，包含被边连接的两个节点，边的特征


import pandas as pd
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
import ast
import os
from itertools import combinations

#对股票代码进行格式统一
class EXCHANGE():
    XSHG = 'XSHG'
    SSE = 'XSHG'
    SH = 'SH'
    XSHE = 'XSHE'
    SZ = 'SZ'
    SZE = 'XSHE'

def normalize_code(symbol, pre_close=None):
    """
    归一化证券代码
    
    :param symbol: 如 '1' 或 '000001'
    :param pre_close: 前收盘价（用于推断指数）
    :return: 证券代码的全称 如 '000001.SZ'
    """
    if not isinstance(symbol, str):
        return symbol

    # 处理带交易所前缀的代码（如 sz000001）
    if symbol.startswith('sz') and len(symbol) == 8:
        normalized = symbol[2:8].zfill(6)
        return f"{normalized}.{EXCHANGE.SZ}"
    elif symbol.startswith('sh') and len(symbol) == 8:
        normalized = symbol[2:8].zfill(6)
        return f"{normalized}.{EXCHANGE.SH}"

    # 尝试补全为6位代码（如 '1' → '000001'）
    if len(symbol) < 6:
        symbol = symbol.zfill(6)  # 左侧补零到6位

    # 仅处理6位代码
    if len(symbol) != 6:
        return symbol  # 非6位代码直接返回

    # 规则匹配
    if symbol.startswith('00'):
        if pre_close and pre_close > 2000:  # 推断为上证指数
            return f"{symbol}.{EXCHANGE.SH}"
        else:
            return f"{symbol}.{EXCHANGE.SZ}"
    elif symbol.startswith(('399', '159', '150', '16', '184801', '201872')):
        return f"{symbol}.{EXCHANGE.SZ}"
    elif symbol.startswith(('50', '51', '60', '688', '900')) or symbol == '751038':
        return f"{symbol}.{EXCHANGE.SH}"
    elif symbol[:3] in ['000', '001', '002', '200', '300']:
        return f"{symbol}.{EXCHANGE.SZ}"
    else:
        return f"{symbol}.{EXCHANGE.SZ}"  # 默认归属深交所（根据需求调整）

def generate_pairs(stock_list):
    codes = [d['stocks_code'] for d in stock_list]
    return list(combinations(codes, 2)) if len(codes) >= 2 else []

#构建边数据集
def construct_edge(news_file,output_file=None):

    news_df=pd.read_csv(news_file)
    news_df["stocks"] = news_df["stocks"].apply(ast.literal_eval)
    
    news_df.reset_index(drop=True, inplace=True)
    news_df_filter=news_df[news_df["stocks"].apply(len) > 1]
    df=news_df_filter

    df['pairs'] = df['stocks'].apply(generate_pairs)
    df = df.explode('pairs')

    # 拆分组合为两列
    df[['stock_1', 'stock_2']] = pd.DataFrame(df['pairs'].tolist(), index=df.index)

    # 重置索引并清理
    result = df.drop(columns=['stocks', 'pairs']).reset_index(drop=True)


    result["stock_1"] = result["stock_1"].astype(str).apply(normalize_code)
    result["stock_2"] = result["stock_2"].astype(str).apply(normalize_code)


    # 读取成分股数据
    constituents_df = pd.read_csv("./CSIA100_normalized.csv")
    constituent_codes = set(constituents_df['成份券代码Constituent Code'])
    
    # 筛选同时包含两个成分股的行
    filtered_result = result[
        result['stock_1'].isin(constituent_codes) &  result['stock_2'].isin(constituent_codes)
    ]

    #保存结果
    if output_file:
        filtered_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"筛选结果已保存到: {output_file}")
    return result


input_dir="./data_raw/2022"
output_dir="./data_processed/all/2022"

for filename in os.listdir(input_dir):
    if filename.startswith("新闻_") and filename.endswith(".csv"):
        # 构建输入输出路径
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace("新闻_", "news_")
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"处理中: {filename} → {output_filename}")
            
            try:
                construct_edge(input_path, output_path)
                
            except Exception as e:
                print(f"处理文件{filename}时出错: {str(e)}")

#数据引入

