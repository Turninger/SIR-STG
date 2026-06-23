"""
获取节点数据  从tushare获取特征数据
"""

import os
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

def get_node_data(input_file, output_file):
    """
    处理股票数据并生成带标签的节点数据
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    """
    # 设置Tushare token (需提前注册获取)
    ts.set_token('e5991012344cb5807859d974da3d1a08a98f5c404bf8dace4e7e4ebe')  # 替换为你的实际token
    pro = ts.pro_api()

    # 1. 读取数据并处理股票代码
    df = pd.read_csv(input_file)
    
    # 合并两列股票代码并去重
    stocks = pd.concat([df['stock_1'], df['stock_2']]).unique().tolist()
    
    # 2. 获取日期列并转换格式
    df['trade_date'] = pd.to_datetime(df['new_date_column']).dt.strftime('%Y%m%d')
    unique_dates = df['trade_date'].unique().tolist()

    ts_code=','.join(stocks)
    # 3. 获取股票数据
    all_data = []
    # 获取基础行情数据
    daily_df = pro.daily(ts_code=ts_code, 
                        start_date=min(unique_dates), 
                        end_date=max(unique_dates))
    

    
    # 合并数据
    merged_df = daily_df
    all_data.append(merged_df)
    
    if not all_data:
        raise ValueError("未获取到任何股票数据")
    
    full_data = pd.concat(all_data)
    
    # 4. 准备下一日收盘价数据
    full_data['next_close'] = full_data.groupby('ts_code')['close'].shift(-1)
    
    # 5. 计算标签
    full_data['label'] = (full_data['next_close'] > full_data['close']).astype(int)
    
    # 6. 筛选所需列并保存
    #result = full_data[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'vol', 'label']]
    result = full_data[['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount','label']]
    result.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存至: {output_file}")

# 使用示例
input_dir="../data_processed/all/SS_edge"
output_dir="../data_processed/all/nodes"

for filename in os.listdir(input_dir):
    # 构建输入输出路径
    input_path = os.path.join(input_dir, filename)
    output_filename = filename
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"处理中: {filename} → {output_filename}")
    
    try:
        get_node_data(input_path, output_path)
        
    except Exception as e:
        print(f"处理文件{filename}时出错: {str(e)}")
        

#数据引入


# input_file="../data_processed/CSI100/SS_edge/news_20220328.csv"
# output_file="../CSI100/node/news_20220328.csv"
# if __name__ == "__main__":
#     get_node_data(input_file, output_file)