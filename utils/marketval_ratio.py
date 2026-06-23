import tushare as ts
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')
pro = ts.pro_api()

# 读取原始CSV文件
# df = pd.read_csv("../data_processed/CSI100/NS_edge_industry/news_20220328.csv") 




# 转换日期格式为YYYYMMDD
def convert_date_format(date_str):
    try:
        start_date=pd.to_datetime(date_str).strftime('%Y%m%d')
        #end_date= start_date+pd.Timedelta(days=1)
        return start_date
    except:
        return np.nan



# 定义函数获取行业市值占比
def get_industry_market_ratio(stock_code, trade_date):
    """
    计算指定股票在指定日期的行业市值占比
    
    参数:
    stock_code (str): 股票代码 (如 '000001.SZ')
    trade_date (str): 交易日期 (格式 'YYYYMMDD')
    
    返回:
    float: 行业市值占比
    """
     # 1. 获取股票所属行业
    stock_info = pro.stock_basic(ts_code=stock_code, fields='industry')
    if stock_info.empty:
        return np.nan
    
    industry = stock_info['industry'].values[0]
    
    # 2. 获取当日全市场股票市值数据
    daily_data = pro.daily_basic(trade_date=trade_date, 
                                fields='ts_code,total_mv')
    if daily_data.empty:
        return np.nan
    
    # 3. 获取全市场股票行业信息（批量获取）
    if not hasattr(get_industry_market_ratio, 'industry_cache'):
        all_stocks = pro.stock_basic(exchange='', list_status='L', 
                                    fields='ts_code,industry')
        get_industry_market_ratio.industry_cache = all_stocks.set_index('ts_code')['industry']
    
    # 4. 合并行业信息到市值数据
    daily_data['industry'] = daily_data['ts_code'].map(
        get_industry_market_ratio.industry_cache
    )
    
    # 5. 计算行业总市值和市场总市值
    industry_mask = daily_data['industry'] == industry
    industry_mv = daily_data.loc[industry_mask, 'total_mv'].sum()
    total_mv = daily_data['total_mv'].sum()
    
    # 6. 计算占比
    if total_mv > 0 and industry_mv > 0:
        return industry_mv / total_mv
    return np.nan
    # try:
       
        
    # except Exception as e:
    #     print(f"Error processing {stock_code} on {trade_date}: {e}")
    #     return np.nan

def add_market_ratio(input_file,output_file):

    # 读取原始CSV文件
    df = pd.read_csv(input_file) 
    # 确保日期列是字符串类型
    df['new_date_column'] = df['new_date_column'].astype(str)
    df["trade_date"]= df['new_date_column'].apply(convert_date_format)

    # 确保stock_code列是字符串类型
    df['stock_code'] = df['stock_code'].astype(str)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        stock_code = row['stock_code']

    trade_date=df['trade_date'][0]
    trade_cal = pro.trade_cal(exchange='SSE', start_date='20211231', end_date=trade_date)
    trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
    trade_date=trade_dates[0]
    print(trade_date)

    # 应用函数获取每行的行业市值占比
    market_ratios = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        stock_code = row['stock_code']
        #trade_date = row['trade_date']
        #若该日为非交易日，获取最近的一个交易日
        # trade_cal = pro.trade_cal(exchange='SSE', start_date='20211231', end_date=trade_date)
        # trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()

        #trade_date=trade_dates[0]


        #end_date= row['end_date']
        
        # 获取市值占比
        market_ratio = get_industry_market_ratio(stock_code, trade_date)
        #print(market_ratio)
        market_ratios.append(market_ratio)
        
        # 避免API请求过于频繁
        time.sleep(0.2)

    # 添加市值占比列到DataFrame
    df['market_ratio'] = market_ratios

    # 保存结果到新的CSV文件
    #output_file = "../data_processed/CSI100/NS_edge_industry/news_20220328_with_market_ratio.csv"
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

#add_market_ratio("../data_processed/all/2022/NS_edge/news_20220104.csv","../data_processed/all/2022/NS_IS_edge/news_20220104.csv")