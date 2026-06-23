"""
获取节点数据  从tushare获取特征数据
"""
import time
import os
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

def get_node_data(input_files, output_file):
    """
    处理股票数据并生成带标签的节点数据

    参数:
    input_files (list): 输入CSV文件路径列表
    output_file (str): 输出CSV文件路径
    """
    # 设置Tushare token (需提前注册获取)
    ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')  # 替换为你的实际token
    pro = ts.pro_api()

    all_dfs = []
    all_stocks = set()
    all_dates = set()

    # 1. 读取数据并处理股票代码和日期
    for input_file in input_files:
        df = pd.read_csv(input_file)
        all_dfs.append(df)
        stocks = pd.concat([df['stock_1'], df['stock_2']]).unique().tolist()
        all_stocks.update(stocks)
        df['trade_date'] = pd.to_datetime(df['new_date_column']).dt.strftime('%Y%m%d')
        unique_dates = df['trade_date'].unique().tolist()
        all_dates.update(unique_dates)
    time.sleep(0.1)
    ts_code = ','.join(all_stocks)
    # 获取最后一天的日期
    last_date = max(all_dates)
    #若该日为非交易日，获取最近的一个交易日
    trade_cal = pro.trade_cal(exchange='SSE', start_date='20211231', end_date=last_date)
    trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()

    time.sleep(0.2)
    trade_date=trade_dates[0]
    # 计算下一个交易日
    # trade_cal = pro.trade_cal(exchange='SSE', start_date=last_date, end_date='20251231')
    # trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
    if len(trade_dates) > 1:
        next_trade_date = trade_dates[1]
    else:
        raise ValueError(f"无法获取{last_date}的下一个交易日")

    # 3. 获取股票数据 - 只获取最后一天
    all_data = []
    # 获取基础行情数据
    daily_df = pro.daily(ts_code=ts_code,
                         trade_date=trade_date)

    # 4. 获取下一个交易日的收盘价
    next_day_df = pro.daily(ts_code=ts_code,
                           trade_date=next_trade_date)
    next_day_close = next_day_df[['ts_code', 'close']].rename(columns={'close': 'next_close'})
    time.sleep(0.3)
    # 5. 合并数据并计算标签
    merged_df = pd.merge(daily_df, next_day_close, on='ts_code', how='left')
    merged_df['label'] = (merged_df['next_close'] > merged_df['close']).astype(int)

    if merged_df.empty:
        raise ValueError("未获取到任何股票数据")

    # 6. 筛选所需列并保存
    result = merged_df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
    result.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存至: {output_file}")

def process_n_days(input_dir, output_dir, n):
    # 获取所有文件并按日期排序
    files = sorted([f for f in os.listdir(input_dir) if f.startswith('news_') and f.endswith('.csv')])
    for i in range(n - 1, len(files)):
        current_date = datetime.strptime(files[i][5:13], '%Y%m%d')
        start_index = i - n + 1
        input_files = [os.path.join(input_dir, files[j]) for j in range(start_index, i + 1)]
        output_filename = files[i]
        output_path = os.path.join(output_dir, output_filename)

        print(f"处理中: {','.join([files[j] for j in range(start_index, i + 1)])} → {output_filename}")

        try:
            get_node_data(input_files, output_path)
        except Exception as e:
            print(f"处理文件{output_filename}时出错: {str(e)}")

# 使用示例
input_dir = "../data_processed/CSI100/SS_edge"
output_dir = "../data_processed/CSI100/7_day_nodes"
n = 7 # 可以修改为你需要的天数

process_n_days(input_dir, output_dir, n)