import pandas as pd
import numpy as np
import os

def filter_edges_by_constituents(edge_file, constituents_file, output_file=None):
    """
    筛选边文件中 stock_1 和 stock_2 都属于成分股代码的行
    
    Args:
        edge_file (str): 边文件路径（包含stock_1和stock_2列）
        constituents_file (str): 成分股文件路径
        output_file (str, optional): 输出文件路径。如果为 None，则不保存
    
    Returns:
        pd.DataFrame: 筛选后的边数据
    """
    # 读取边数据
    edge_df = pd.read_csv(edge_file)
    
    # 读取成分股数据
    constituents_df = pd.read_csv(constituents_file)
    constituent_codes = set(constituents_df['成份券代码Constituent Code'])
    
    # 筛选同时包含两个成分股的行
    filtered_edges = edge_df[
        edge_df['stock_1'].isin(constituent_codes) &  edge_df['stock_2'].isin(constituent_codes)
    ]
    
    # 可选：保存结果
    if output_file:
        filtered_edges.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"筛选结果已保存到: {output_file}")
    
    return filtered_edges

# 主处理流程
input_dir = "../data_processed/all/2022/SS_edge"
output_dir = "../data_processed/CSI100/SS_edge"
constituents_file = '../CSIA100_normalized.csv'  # 成分股文件路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"处理中: {filename}")
        try:
            filter_edges_by_constituents(
                edge_file=input_path,
                constituents_file=constituents_file,
                output_file=output_path
            )
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")