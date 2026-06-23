import torch
import os
import pandas as pd
from torch_geometric.data import HeteroData
from datetime import datetime
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用详细错误报告
os.environ['TORCH_USE_CUDA_DSA'] = '1'   # 启用设备端断言

# 定义数据集保存路径
DATASET_PATH = "hetero_graph_dataset.pt"


# 1. 获取所有日期列表
node_dir = "./data_processed/all/nodes"
all_dates = sorted(
    [f.split('_')[1].split('.')[0] for f in os.listdir(node_dir) 
    if f.startswith('news_') and f.endswith('.csv')
])
num_days = len(all_dates)
split_idx = int(num_days * 4 / 5)
train_dates = all_dates[:split_idx]
test_dates = all_dates[split_idx:]

print(f"总天数: {num_days}")
print(f"训练集日期: {train_dates[0]} 至 {train_dates[-1]} ({len(train_dates)}天)")
print(f"测试集日期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)}天)")



# 2. 为每个日期创建独立的异构图
def create_daily_graph(date):
    """为指定日期创建异构图"""
    data = HeteroData()
    
    # 读取三个文件
    node_file = f"./data_processed/all/nodes/news_{date}.csv"
    ss_edge_file = f"./data_processed/all/SS_edge/news_{date}.csv"
    ns_is_edge_file = f"./data_processed/all/NS_IS_edge/news_{date}_with_MR_with_SA.csv"
    
    if not all(os.path.exists(f) for f in [node_file, ss_edge_file, ns_is_edge_file]):
        print(f"跳过不完整日期: {date}")
        return None
    
    # 读取数据
    nodes_df = pd.read_csv(node_file)
    ss_edge_df = pd.read_csv(ss_edge_file)
    ns_is_edge_df = pd.read_csv(ns_is_edge_file)
    
    # 创建本地映射
    stocks= nodes_df["ts_code"].unique()
    #stocks = pd.concat([ss_edge_df['stock_1'], ss_edge_df['stock_2']]).unique()
    industries = ns_is_edge_df['industry_code'].unique()
    news_ids = ns_is_edge_df['id'].unique()
    
    stock_map = {code: idx for idx, code in enumerate(stocks)}
    industry_map = {code: i for i, code in enumerate(industries)}
    news_map = {id: i for i, id in enumerate(news_ids)}
    
    # 添加股票节点
    stock_features = []
    stock_labels = []
    for _, row in nodes_df.iterrows():
        if row['ts_code'] in stock_map:
            stock_idx = stock_map[row['ts_code']]
            # 确保特征维度一致
            features = row[2:10].values.astype(np.float32)
            if len(features) < 8:
                features = np.pad(features, (0, 8 - len(features)), 'constant')
            stock_features.append(features)
            stock_labels.append(row['label'])
    
    if stock_features:
        data['stock'].x = torch.tensor(np.array(stock_features), dtype=torch.float)
        data['stock'].y = torch.tensor(stock_labels, dtype=torch.long)
    
    # 添加行业节点（无特征）
    data['industry'].x = torch.ones(len(industry_map), 1, dtype=torch.float)  # 占位特征
    data['industry'].num_nodes = len(industry_map)
    
    # 添加新闻节点（无特征）
    data['news'].x = torch.ones(len(news_map), 1, dtype=torch.float)  # 占位特征
    data['news'].num_nodes = len(news_map)
    
    # 添加S-S边
    ss_edge_src = []
    ss_edge_dst = []
    ss_edge_attr = []
    for _, row in ss_edge_df.iterrows():
        if row['stock_1'] in stock_map and row['stock_2'] in stock_map:
            src = stock_map[row['stock_1']]
            dst = stock_map[row['stock_2']]
            ss_edge_src.append(src)
            ss_edge_dst.append(dst)
            ss_edge_attr.append(row['linkage_value'])
    
    if ss_edge_src:
        data['stock', 'link', 'stock'].edge_index = torch.tensor([ss_edge_src, ss_edge_dst], dtype=torch.long)
        data['stock', 'link', 'stock'].edge_attr = torch.tensor(ss_edge_attr, dtype=torch.float)
    
    # 添加N-S边和I-S边
    ns_edge_src = []
    ns_edge_dst = []
    ns_edge_attr = []
    is_edge_src = []
    is_edge_dst = []
    is_edge_attr = []
    
    for _, row in ns_is_edge_df.iterrows():
        stock_code = row['stock_code']
        industry_code = row['industry_code']
        news_id = row['id']
        
        if stock_code in stock_map and industry_code in industry_map and news_id in news_map:
            stock_idx = stock_map[stock_code]
            industry_idx = industry_map[industry_code]
            news_idx = news_map[news_id]
            
            # N-S边
            ns_edge_src.append(news_idx)
            ns_edge_dst.append(stock_idx)
            ns_edge_attr.append(row['sentiment_confidence'])
            
            # I-S边
            is_edge_src.append(industry_idx)
            is_edge_dst.append(stock_idx)
            is_edge_attr.append(row['market_ratio'])
    
    if ns_edge_src:
        data['news', 'sentiment', 'stock'].edge_index = torch.tensor([ns_edge_src, ns_edge_dst], dtype=torch.long)
        data['news', 'sentiment', 'stock'].edge_attr = torch.tensor(ns_edge_attr, dtype=torch.float)
    
    if is_edge_src:
        data['industry', 'marketratio', 'stock'].edge_index = torch.tensor([is_edge_src, is_edge_dst], dtype=torch.long)
        data['industry', 'marketratio', 'stock'].edge_attr = torch.tensor(is_edge_attr, dtype=torch.float)
    
    print(f"日期 {date}: 股票节点={len(stocks)} 新闻节点={len(news_ids)} 行业节点={len(industries)}")
    print(f"    SS边={len(ss_edge_src)} NS边={len(ns_edge_src)} IS边={len(is_edge_src)}")
    
    return data