import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import pandas as pd
import numpy as np

# 1. 数据引入
nodes_df = pd.read_csv("./data_processed/all/nodes/news_20220328.csv")
ss_edge_df = pd.read_csv("./data_processed/all/SS_edge/news_20220328.csv")
ns_is_edge_df = pd.read_csv("./data_processed/all/NS_IS_edge/news_20220328_with_MR_with_SA.csv")

# 2. 创建股票代码到节点索引的映射（保留所有股票，包括节点数据中不存在的）
all_stocks = pd.concat([ss_edge_df['stock_1'], ss_edge_df['stock_2']]).unique()
industry_codes = ns_is_edge_df["industry_code"].drop_duplicates().tolist()
news_ids = ns_is_edge_df["id"].drop_duplicates().tolist()

# 对节点进行 map 转换
code_to_index = {code: idx for idx, code in enumerate(all_stocks)}
industry_id_map = {code: i for i, code in enumerate(industry_codes)}
news_id_map = {id: i for i, id in enumerate(news_ids)}

# 3. 将 S-S 边数据中的股票代码转换为节点索引
ss_edge_df["stock_1_idx"] = ss_edge_df["stock_1"].map(code_to_index)
ss_edge_df["stock_2_idx"] = ss_edge_df["stock_2"].map(code_to_index)
ss_edge_df = ss_edge_df.dropna(subset=["stock_1_idx", "stock_2_idx"])

ns_is_edge_df["stock_code_idx"] = ns_is_edge_df["stock_code"].map(code_to_index)
ns_is_edge_df = ns_is_edge_df.dropna(subset=["stock_code_idx"])
ns_is_edge_df["industry_code_idx"] = ns_is_edge_df["industry_code"].map(industry_id_map)
ns_is_edge_df = ns_is_edge_df.dropna(subset=["industry_code_idx"])
ns_is_edge_df["news_id_idx"] = ns_is_edge_df["id"].map(news_id_map)
ns_is_edge_df = ns_is_edge_df.dropna(subset=["news_id_idx"])




# 修改新闻边索引构建部分
# 确保新闻ID映射正确
news_ids = ns_is_edge_df["id"].drop_duplicates().tolist()
news_id_map = {id: i for i, id in enumerate(news_ids)}

# 在构建边索引前，确保映射一致性
valid_news_indices = []
valid_stock_indices = []
valid_confidence = []

for _, row in ns_is_edge_df.iterrows():
    news_id = row["id"]
    stock_code = row["stock_code"]
    
    # 确保新闻ID和股票代码都有有效映射
    if news_id in news_id_map and stock_code in code_to_index:
        news_idx = news_id_map[news_id]
        stock_idx = code_to_index[stock_code]
        
        # 确保索引在有效范围内
        if 0 <= news_idx < len(news_ids) and 0 <= stock_idx < len(all_stocks):
            valid_news_indices.append(news_idx)
            valid_stock_indices.append(stock_idx)
            valid_confidence.append(row["sentiment_confidence"])

# 构建边索引
ns_edge_index = torch.tensor([valid_news_indices, valid_stock_indices], dtype=torch.long)
ns_edge_attr = torch.tensor(valid_confidence, dtype=torch.float)

# 4. 构建 SS 边索引（不过滤无效边，直接使用原始边数据）
ss_edge_index = torch.tensor(
    [ss_edge_df["stock_1_idx"].tolist(), ss_edge_df["stock_2_idx"].tolist()],
    dtype=torch.long
)



# 修改行业-股票边构建部分
valid_industry_indices = []
valid_stock_indices_is = []
valid_market_ratios = []

for _, row in ns_is_edge_df.iterrows():
    industry_code = row["industry_code"]
    stock_code = row["stock_code"]
    
    # 确保行业代码和股票代码都有有效映射
    if industry_code in industry_id_map and stock_code in code_to_index:
        industry_idx = industry_id_map[industry_code]
        stock_idx = code_to_index[stock_code]
        
        # 确保索引在有效范围内
        if 0 <= industry_idx < len(industry_codes) and 0 <= stock_idx < len(all_stocks):
            valid_industry_indices.append(industry_idx)
            valid_stock_indices_is.append(stock_idx)
            valid_market_ratios.append(row["market_ratio"])

# 构建边索引
is_edge_index = torch.tensor([valid_industry_indices, valid_stock_indices_is], dtype=torch.long)
is_edge_attr = torch.tensor(valid_market_ratios, dtype=torch.float)


# 5. 创建节点特征（使用前 10 列作为特征）
node_features = nodes_df[nodes_df.columns[2:10]]

# 添加股票节点（使用所有在边中出现的股票）
all_stocks_df = pd.DataFrame({"ts_code": all_stocks})

# 8. 创建异构图数据（保留原始代码中的异构图逻辑）
data = HeteroData()

# 初始化节点特征
data['stock'].x = torch.zeros(len(all_stocks), node_features.shape[1], dtype=torch.float)

data['industry'].num_nodes = len(industry_codes)  # 显式设置节点数量
# 新闻节点（无特征，但设置节点数量）
data['news'].num_nodes = len(news_ids)  # 显式设置节点数量
# data['industry'].x = torch.tensor(list(industry_id_map.values()), dtype=torch.long)
# data['news'].x = torch.tensor(list(news_id_map.values()), dtype=torch.long)

# 为存在于节点数据中的股票填充特征
for idx, code in enumerate(nodes_df["ts_code"]):
    if code in code_to_index:
        stock_idx = code_to_index[code]
        data['stock'].x[stock_idx] = torch.tensor(node_features.iloc[idx].values, dtype=torch.float)

data['stock'].y = torch.zeros(len(all_stocks), dtype=torch.long)
for idx, code in enumerate(nodes_df["ts_code"]):
    if code in code_to_index:
        stock_idx = code_to_index[code]
        data['stock'].y[stock_idx] = nodes_df.iloc[idx]["label"]

# 构建 S-S 边（使用原始边数据）
data['stock', 'link', 'stock'].edge_index = ss_edge_index
data['stock', 'link', 'stock'].edge_attr = torch.tensor(ss_edge_df["linkage_value"].values, dtype=torch.float)

# 构建 N-S 边
data['news','sentiment','stock'].edge_index = ns_edge_index
data['news','sentiment','stock'].edge_attr = torch.tensor(ns_is_edge_df["sentiment_confidence"].values, dtype=torch.float)
# 构建 I-S 边
data['industry','marketratio','stock'].edge_index = is_edge_index
data['industry','marketratio','stock'].edge_attr = torch.tensor(ns_is_edge_df["market_ratio"].values, dtype=torch.float)

# 打印异构图构建结果
print(f"\n异构图构建结果：")
print(f"股票节点数: {data['stock'].num_nodes}")
print(f"SS 边数: {data['stock', 'link', 'stock'].num_edges}")
print(f"节点特征维度: {data['stock'].x.shape[1]}")

# 查看概览信息
def summarize_hetero_data(data):
    """打印异构图的统计信息"""
    print("=" * 40)
    print("异构图数据概览:")
    print("=" * 40)
    
    # 节点信息
    print("\n节点类型和数量:")
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        print(f"- {node_type}: {num_nodes} 个节点")
    
    # 边信息
    print("\n边类型和数量:")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].num_edges
        print(f"- {edge_type}: {num_edges} 条边")
    
    # 孤立节点检测
    print("\n孤立节点检测:")
    for node_type in data.node_types:
        # 收集所有涉及该节点类型的边
        related_edges = []
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            if src_type == node_type or dst_type == node_type:
                related_edges.append(edge_type)
        
        # 如果没有与该节点类型相关的边，则所有节点都是孤立的
        if not related_edges:
            print(f"- {node_type}: 所有节点都是孤立的 ({data[node_type].num_nodes}个)")
            continue
        
        # 否则，检查哪些节点没有出现在任何边中
        all_nodes = set(range(data[node_type].num_nodes))
        connected_nodes = set()
        
        for edge_type in related_edges:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index
            
            if src_type == node_type:
                connected_nodes.update(edge_index[0].tolist())
            if dst_type == node_type:
                connected_nodes.update(edge_index[1].tolist())
        
        isolated_nodes = all_nodes - connected_nodes
        print(f"- {node_type}: {len(isolated_nodes)} 个孤立节点")
        if isolated_nodes:
            print(f"  孤立节点 ID: {sorted(isolated_nodes)}")
    
    # 节点特征维度
    print("\n节点特征维度:")
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            print(f"- {node_type}: {data[node_type].x.shape[0]} 维特征")
        else:
            print(f"- {node_type}: 无节点特征")
    
    # 边特征维度
    print("\n边特征维度:")
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_attr'):
            print(f"- {edge_type}: {data[edge_type].edge_attr.shape[1]} 维特征")
        else:
            print(f"- {edge_type}: 无边特征")
    
    print("=" * 40)

summarize_hetero_data(data)

