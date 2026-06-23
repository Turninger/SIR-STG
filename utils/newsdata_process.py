### 构建边数据集，包含被边连接的两个节点，边的特征
import pandas as pd
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
import ast
import os
from itertools import combinations
import tushare as ts

ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')
pro = ts.pro_api()
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



def process_data(news_file, output_file=None):
    # 读取CSV文件
    news_df = pd.read_csv(news_file)
    
    # 将字符串形式的列表转换为实际列表
    news_df["stocks"] = news_df["stocks"].apply(ast.literal_eval)
    
    # 展开stocks列，每支股票创建一行
    df = news_df.explode('stocks')
    
    # 从字典中提取股票代码
    df['stock_code'] = df['stocks'].apply(lambda x: x['stocks_code'])
    
    # 删除原始stocks列
    df.drop(columns=['stocks'], inplace=True)
    
    # 重置索引
    result = df.reset_index(drop=True)
    result['stock_code']=result['stock_code'].astype(str).apply(normalize_code)


    # 提取唯一的股票代码并转换为tushare需要的格式
    stock_codes = result['stock_code'].dropna().unique()
    ts_codes = [code.split('.')[0] for code in stock_codes]  # 去掉后缀

    batch_size = 1000
    stock_name_map = {}

    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i+batch_size]
        data = pro.query('stock_basic', exchange='', list_status='L', 
                        fields='symbol,name')
        
        # 创建映射字典：symbol -> name
        batch_map = dict(zip(data['symbol'], data['name']))
        stock_name_map.update(batch_map)

    # 添加新列
    def get_stock_name(row):
        if pd.isna(row['stock_code']):
            return None
        symbol = row['stock_code'].split('.')[0]  # 提取纯数字代码
        return stock_name_map.get(symbol, None)

    result['stock_name'] = result.apply(get_stock_name, axis=1)


    # 保存结果
    if output_file:
        result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"转换结果已保存到: {output_file}")
    
    return result

# 示例调用
# result = process_data('input.csv', 'output.csv')

input_dir="../data_processed/all/2022/buquan"
output_dir="../data_processed/all/2022/buquan"

for filename in os.listdir(input_dir):
    if filename.startswith("新闻_") and filename.endswith(".csv"):
        # 构建输入输出路径
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace("新闻_", "news_")
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"处理中: {filename} → {output_filename}")
            
            try:
                process_data(input_path, output_path)
                
            except Exception as e:
                print(f"处理文件{filename}时出错: {str(e)}")

#数据引入

