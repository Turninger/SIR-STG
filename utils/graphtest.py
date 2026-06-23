import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import pandas as pd

# 定义编码和解码函数，增强鲁棒性
exchange_to_id = {"SH": 0, "SZ": 1, "SI": 2}
id_to_exchange = {0: "SH", 1: "SZ", 2: "SI"}

def stock_code_encode(code):
    """将股票代码转换为数值ID，增强错误处理"""
    if not isinstance(code, str):
        print(f"警告：非字符串类型的股票代码 {code}")
        return None
    
    parts = code.split('.')
    if len(parts) != 2:
        # 尝试其他可能的分隔符
        if '_' in code:
            parts = code.split('_')
        elif '-' in code:
            parts = code.split('-')
        else:
            print(f"无效股票代码格式: {code}")
            return None
    
    number_part = parts[0]
    exchange_part = parts[1]
    
    # 确保数字部分可以转换为整数
    try:
        number = int(number_part)
    except ValueError:
        print(f"无法将股票代码的数字部分转换为整数: {code}")
        return None
    
    # 处理交易所代码（不区分大小写）
    exchange_id = exchange_to_id.get(exchange_part.upper(), -1)
    if exchange_id == -1:
        print(f"未知交易所代码: {exchange_part}，股票代码: {code}")
        return None
    
    # 组合为唯一ID
    combined = (number << 8) | exchange_id
    return combined

def stock_code_decode(combined_id):
    """将数值ID解码为股票代码"""
    number_part = combined_id >> 8
    exchange_part = combined_id & 0xFF
    exchange_code = id_to_exchange.get(exchange_part, "未知")
    return f"{number_part}.{exchange_code}"

# 1. 数据引入
nodes_df = pd.read_csv("../CSI100/node/news_20220328.csv")
edge_df = pd.read_csv("../CSI100/NS_IS_edge/news_20220328_with_MR_with_SA.csv")
ss_edge_df = pd.read_csv("../CSI100/SS_edge/news_20220328.csv")

# 2. 创建股票代码到节点索引的映射（改进错误处理）
# 对nodes_df中的ts_code进行编码
stock_id_map = {}
invalid_codes = []

for code in nodes_df["ts_code"].unique():
    encoded = stock_code_encode(code)
    if encoded is not None:
        stock_id_map[code] = encoded
    else:
        invalid_codes.append(code)

if invalid_codes:
    print(f"警告：{len(invalid_codes)}个股票代码无法编码")
    print(f"无效代码示例: {invalid_codes[:5]}")

# 创建股票ID到节点索引的映射
id_to_index = {stock_id: idx for idx, stock_id in enumerate(nodes_df["ts_code"].map(stock_id_map).dropna())}

# 3. 将边数据中的股票代码转换为节点索引（改进错误处理）
ss_edge_df["stock_1_id"] = ss_edge_df["stock_1"].map(stock_id_map)
ss_edge_df["stock_2_id"] = ss_edge_df["stock_2"].map(stock_id_map)

# 统计无法映射的边
unmapped_edges = ss_edge_df[ss_edge_df["stock_1_id"].isna() | ss_edge_df["stock_2_id"].isna()]
if not unmapped_edges.empty:
    print(f"警告：{len(unmapped_edges)}条边包含无法映射的股票代码")
    print(f"示例无法映射的股票代码: {unmapped_edges[['stock_1', 'stock_2']].head()}")

# 过滤掉无效的边
valid_edges = ss_edge_df.dropna(subset=["stock_1_id", "stock_2_id"]).copy()

# 将股票ID转换为节点索引
valid_edges["stock_1_idx"] = valid_edges["stock_1_id"].map(id_to_index)
valid_edges["stock_2_idx"] = valid_edges["stock_2_id"].map(id_to_index)

# 再次过滤可能存在的无效索引
valid_edges = valid_edges.dropna(subset=["stock_1_idx", "stock_2_idx"])

print(f"有效边数量: {len(valid_edges)}")

# 4. 创建异构图数据
data = HeteroData()

# 添加股票节点
feature_columns = [col for col in nodes_df.columns if col not in ["ts_code", "label"]]
data['stock'].x = torch.tensor(nodes_df[feature_columns].values, dtype=torch.float)
data['stock'].y = torch.tensor(nodes_df["label"].values, dtype=torch.long)

# 添加行业节点
industry_codes = edge_df["industry_code"].drop_duplicates().tolist()
industry_id_map = {code: i for i, code in enumerate(industry_codes)}
data['industry'].x = torch.tensor(list(industry_id_map.values()), dtype=torch.long)

# 添加新闻节点
news_ids = edge_df["id"].drop_duplicates().tolist()
news_id_map = {id: i for i, id in enumerate(news_ids)}
data['news'].x = torch.tensor(list(news_id_map.values()), dtype=torch.long)

# 5. 构建S-S边
ss_forward_edges = valid_edges[['stock_1_idx', 'stock_2_idx']].values.T
ss_reverse_edges = valid_edges[['stock_2_idx', 'stock_1_idx']].values.T

# 合并正向和反向边
ss_edge_index = torch.tensor(
    [ss_forward_edges[0].tolist() + ss_reverse_edges[0].tolist(),
     ss_forward_edges[1].tolist() + ss_reverse_edges[1].tolist()],
    dtype=torch.long
)

# 添加S-S邻边矩阵
data['stock', 'link', 'stock'].edge_index = ss_edge_index

# 添加S-S边的权重
edge_weights = valid_edges["linkage_value"].tolist()
data['stock', 'link', 'stock'].edge_attr = torch.tensor(
    edge_weights + edge_weights,  # 正向边权重 + 反向边权重
    dtype=torch.float
)

# 打印构建结果
print(f"股票节点数: {data['stock'].num_nodes}")
print(f"SS边数: {data['stock', 'link', 'stock'].num_edges}")
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
            print(f"  孤立节点ID: {sorted(isolated_nodes)}")
    
    # 节点特征维度
    print("\n节点特征维度:")
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            print(f"- {node_type}: {data[node_type].x.shape[1]} 维特征")
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