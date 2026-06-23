"""
构建220328-220703的所有

"""
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
from datetime import datetime, timedelta
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用详细错误报告
os.environ['TORCH_USE_CUDA_DSA'] = '1'   # 启用设备端断言

# 定义数据集保存路径
DATASET_PATH = "5_day_100_hetero_graph_dataset.pt"

# 1. 获取所有日期列表
node_dir = "./data_processed/CSI100/7_day_nodes"
all_dates = sorted(
    [f.split('_')[1].split('.')[0] for f in os.listdir(node_dir) 
    if f.startswith('news_') and f.endswith('.csv')
])
num_days = len(all_dates)
split_idx = int(num_days * 2 / 3)
train_dates = all_dates[:split_idx]
test_dates = all_dates[split_idx:]

print(f"总天数: {num_days}")
print(f"训练集日期: {train_dates[0]} 至 {train_dates[-1]} ({len(train_dates)}天)")
print(f"测试集日期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)}天)")

# 2. 为每个日期创建独立的异构图
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

def create_daily_graph(date, n_days=7):
    """为指定日期创建异构图，合并前n_days的边数据"""
    data = HeteroData()
    
    # 节点文件路径（只使用当天）
    node_file = f"./data_processed/all/2022/5_day_nodes/news_{date}.csv"
    if not os.path.exists(node_file):
        print(f"跳过: 节点文件不存在 - {date}")
        return None
    
    # 生成前n_days的日期列表（包括当天）
    base_date = datetime.strptime(date, "%Y%m%d")
    date_list = [(base_date - timedelta(days=i)).strftime("%Y%m%d") 
                 for i in range(n_days)]
    
    # 合并SS_edge文件
    ss_edge_dfs = []
    for d in date_list:
        ss_file = f"./data_processed/all/2022/SS_edge/news_{d}.csv"
        if os.path.exists(ss_file):
            df = pd.read_csv(ss_file)
            df['date'] = d  # 添加日期列用于去重
            ss_edge_dfs.append(df)
    
    # 合并并去重（保留最近日期的记录）
    ss_edge_df = pd.concat(ss_edge_dfs, ignore_index=True)
    if not ss_edge_df.empty:
        ss_edge_df = ss_edge_df.sort_values('date').drop_duplicates(
            subset=['stock_1', 'stock_2'], keep='last')
    
    # 合并NS_IS_edge文件
    ns_is_edge_dfs = []
    for d in date_list:
        ns_file = f"./data_processed/all/2022/NS_IS_edge/news_{d}_with_MR_with_SA.csv"
        if os.path.exists(ns_file):
            df = pd.read_csv(ns_file)
            df['date'] = d  # 添加日期列用于去重
            ns_is_edge_dfs.append(df)
    
    # 合并并去重（保留最近日期的记录）
    ns_is_edge_df = pd.concat(ns_is_edge_dfs, ignore_index=True)
    if not ns_is_edge_df.empty:
        ns_is_edge_df = ns_is_edge_df.sort_values('date').drop_duplicates(
            subset=['id', 'stock_code', 'industry_code'], keep='last')
    
    # 读取节点数据
    nodes_df = pd.read_csv(node_file)
    
    # 创建本地映射
    stocks = nodes_df["ts_code"].unique()
    industries = ns_is_edge_df['industry_code'].unique() if not ns_is_edge_df.empty else []
    news_ids = ns_is_edge_df['id'].unique() if not ns_is_edge_df.empty else []
    
    stock_map = {code: idx for idx, code in enumerate(stocks)}
    industry_map = {code: i for i, code in enumerate(industries)}
    news_map = {id: i for i, id in enumerate(news_ids)}
    
    # 添加股票节点（特征和标签）
    stock_features = []
    stock_labels = []
    for _, row in nodes_df.iterrows():
        if row['ts_code'] in stock_map:
            stock_idx = stock_map[row['ts_code']]
            features = row[2:10].values.astype(np.float32)
            if len(features) < 8:
                features = np.pad(features, (0, 8 - len(features)), 'constant')
            stock_features.append(features)
            stock_labels.append(row['label'])
    
    if stock_features:
        data['stock'].x = torch.tensor(np.array(stock_features), dtype=torch.float)
        data['stock'].y = torch.tensor(stock_labels, dtype=torch.long)
    
    # 添加行业节点（占位特征）
    data['industry'].x = torch.ones(len(industry_map), 1, dtype=torch.float)
    data['industry'].num_nodes = len(industry_map)
    
    # 添加新闻节点（占位特征）
    data['news'].x = torch.ones(len(news_map), 1, dtype=torch.float)
    data['news'].num_nodes = len(news_map)
    
    # 添加S-S边（如果存在数据）
    ss_edge_src, ss_edge_dst, ss_edge_attr = [], [], []
    if not ss_edge_df.empty:
        for _, row in ss_edge_df.iterrows():
            if row['stock_1'] in stock_map and row['stock_2'] in stock_map:
                ss_edge_src.append(stock_map[row['stock_1']])
                ss_edge_dst.append(stock_map[row['stock_2']])
                ss_edge_attr.append(row['linkage_value'])
    
    if ss_edge_src:
        data['stock', 'link', 'stock'].edge_index = torch.tensor([ss_edge_src, ss_edge_dst], dtype=torch.long)
        data['stock', 'link', 'stock'].edge_attr = torch.tensor(ss_edge_attr, dtype=torch.float)
    
    # 添加N-S边和I-S边（如果存在数据）
    ns_edge_src, ns_edge_dst, ns_edge_attr = [], [], []
    is_edge_src, is_edge_dst, is_edge_attr = [], [], []
    
    if not ns_is_edge_df.empty:
        for _, row in ns_is_edge_df.iterrows():
            stock_code = row['stock_code']
            industry_code = row['industry_code']
            news_id = row['id']
            
            if stock_code in stock_map:
                stock_idx = stock_map[stock_code]
                
                # N-S边
                if news_id in news_map:
                    ns_edge_src.append(news_map[news_id])
                    ns_edge_dst.append(stock_idx)
                    ns_edge_attr.append(row['sentiment_confidence'])
                
                # I-S边
                if industry_code in industry_map:
                    is_edge_src.append(industry_map[industry_code])
                    is_edge_dst.append(stock_idx)
                    is_edge_attr.append(row['market_ratio'])
    
    if ns_edge_src:
        data['news', 'sentiment', 'stock'].edge_index = torch.tensor([ns_edge_src, ns_edge_dst], dtype=torch.long)
        data['news', 'sentiment', 'stock'].edge_attr = torch.tensor(ns_edge_attr, dtype=torch.float)
    
    if is_edge_src:
        data['industry', 'marketratio', 'stock'].edge_index = torch.tensor([is_edge_src, is_edge_dst], dtype=torch.long)
        data['industry', 'marketratio', 'stock'].edge_attr = torch.tensor(is_edge_attr, dtype=torch.float)
    
    # 打印统计信息
    print(f"日期 {date} (合并{n_days}天数据):")
    print(f"  股票节点={len(stocks)} 新闻节点={len(news_ids)} 行业节点={len(industries)}")
    print(f"  SS边={len(ss_edge_src)} NS边={len(ns_edge_src)} IS边={len(is_edge_src)}")
    
    return data
# 3. 构建每日图数据集或从文件加载
def build_or_load_dataset():
    """构建或加载数据集"""
    if os.path.exists(DATASET_PATH):
        print(f"从文件加载数据集: {DATASET_PATH}")
        dataset = torch.load(DATASET_PATH,weights_only=False)
        train_graphs = dataset['train']
        test_graphs = dataset['test']
        print(f"加载完成: {len(train_graphs)}个训练图, {len(test_graphs)}个测试图")
        return train_graphs, test_graphs
    
    print("构建图数据集...")
    train_graphs = []
    test_graphs = []
    
    for date in all_dates:
        graph = create_daily_graph(date)
        if graph is not None:
            if date in train_dates:
                train_graphs.append(graph)
            else:
                test_graphs.append(graph)
    
    print(f"构建完成: {len(train_graphs)}个训练图, {len(test_graphs)}个测试图")
    
    # 保存数据集
    dataset = {
        'train': train_graphs,
        'test': test_graphs
    }
    torch.save(dataset, DATASET_PATH)
    print(f"数据集已保存到: {DATASET_PATH}")
    
    return train_graphs, test_graphs

build_or_load_dataset()
