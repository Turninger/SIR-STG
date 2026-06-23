"""
构建S-S边数据集，保存至SS_edge文件夹内
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import tushare as ts
import time
import os
from datetime import datetime, timedelta
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 设置tushare token
ts.set_token('e5991012344cb5807859d974da3d1a08a98f5c404bf8dace4e7e4ebe')
pro = ts.pro_api()

class StockLinkageCalculator:
    def __init__(self, base_date, days=20, alpha1=0.6, alpha2=0.4):
        self.base_date = base_date
        self.days = days
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.trade_dates = self._get_trade_dates()

    #获取交易日，前20日
    def _get_trade_dates(self):
        start_date = (datetime.strptime(self.base_date, '%Y%m%d') - 
                     timedelta(days=self.days * 2)).strftime('%Y%m%d')
        
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=self.base_date)
        trade_dates = df[df['is_open'] == 1]['cal_date'].tolist()
        return trade_dates[-self.days:]
    

    ###获取股票数据
    def _get_stock_data(self, ts_code):
        start_date = str(self.trade_dates[-1])
        end_date = str(self.trade_dates[0])
        
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            time.sleep(0.1)
            
            # 检查返回的DataFrame是否包含所需列
            if df.empty or not all(col in df.columns for col in ['open', 'close', 'high', 'low']):
                print(f"警告: {ts_code} 返回数据不完整或为空")
                return np.full((self.days, 4), np.nan)
                
            # 转换为多维数组
            data = df[['open', 'close', 'high', 'low']].values
            
            if len(data) < self.days:
                padding = np.full((self.days - len(data), 4), np.nan)
                data = np.vstack([padding, data])
                
            return data
        except Exception as e:
            print(f"获取 {ts_code} 数据时出错: {str(e)}")
            return np.full((self.days, 4), np.nan)
        
    ###获取市场行情数据
    def _get_csi100_data(self):
        start_date = str(self.trade_dates[-1])
        end_date = str(self.trade_dates[0])
        
        try:
            df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date)
            
            if df.empty or not all(col in df.columns for col in ['open', 'close', 'high', 'low']):
                print("警告: CSI100指数数据不完整或为空")
                return np.full((self.days, 4), np.nan)
                
            data = df[['open', 'close', 'high', 'low']].values
            
            if len(data) < self.days:
                padding = np.full((self.days - len(data), 4), np.nan)
                data = np.vstack([padding, data])
                
            return data
        except Exception as e:
            print(f"获取CSI100指数数据时出错: {str(e)}")
            return np.full((self.days, 4), np.nan)
    
    def calculate_pearson_partial(self, stock_a, stock_b, market_index):
        # 转换为浮点数确保数值类型正确
        stock_a = stock_a.astype(float)
        stock_b = stock_b.astype(float)
        market_index = market_index.astype(float)
        
        # 使用pd.isna替代np.isnan
        mask = (
            ~pd.isna(stock_a).any(axis=1) & 
            ~pd.isna(stock_b).any(axis=1) & 
            ~pd.isna(market_index).any(axis=1)
        )
        stock_a = stock_a[mask]
        stock_b = stock_b[mask]
        market_index = market_index[mask]
        
        if len(stock_a) < 5:
            return np.nan
        
        stock_a_flat = stock_a.flatten()
        stock_b_flat = stock_b.flatten()
        market_flat = market_index.flatten()
        
        gamma_ij, _ = pearsonr(stock_a_flat, stock_b_flat)
        gamma_ih, _ = pearsonr(stock_a_flat, market_flat)
        gamma_jh, _ = pearsonr(stock_b_flat, market_flat)
        
        numerator = gamma_ij - gamma_ih * gamma_jh
        denominator = np.sqrt(1 - gamma_ih**2) * np.sqrt(1 - gamma_jh**2)
        
        if denominator < 1e-10:
            return np.nan
        
        partial_corr = numerator / denominator
        return partial_corr
    
    def time_weighted_dtw(self, series1, series2, smoothing_alpha=0.98):
        # 转换为浮点数确保数值类型正确
        series1 = series1.astype(float)
        series2 = series2.astype(float)
        
        # 使用pd.isna替代np.isnan
        mask = ~pd.isna(series1).any(axis=1) & ~pd.isna(series2).any(axis=1)
        series1 = series1[mask]
        series2 = series2[mask]
        
        if len(series1) < 5 or len(series2) < 5:
            return np.nan
        
        try:
            distance, _ = fastdtw(series1, series2, dist=euclidean)
            return distance
        except Exception as e:
            print(f"计算DTW时出错: {str(e)}")
            return np.nan
    
    def calculate_linkage(self, stock_a_code, stock_b_code):
        stock_a_data = self._get_stock_data(stock_a_code)
        stock_b_data = self._get_stock_data(stock_b_code)
        csi100_data = self._get_csi100_data()
        
        partial_corr = self.calculate_pearson_partial(stock_a_data, stock_b_data, csi100_data)
        dtw_distance = self.time_weighted_dtw(stock_a_data, stock_b_data)
        
        if np.isnan(partial_corr) or np.isnan(dtw_distance):
            return np.nan
        
        dtw_similarity = 1 / (1 + dtw_distance)
        linkage_value = self.alpha1 * dtw_similarity + self.alpha2 * partial_corr
        
        return linkage_value

def calculate_weight(input_file,output_file):
    # input_file = "../data_processed/CSI100/SSedge/news_20220328.csv"
    # output_file = "../data_processed/CSI100/SSedge/news_20220328_with_linkage.csv"
    
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在!")
        return
    
    df = pd.read_csv(input_file)
    
    # 去重并重置索引
    df = df.drop_duplicates(subset=['stock_1', 'stock_2']).reset_index(drop=True)
    
    calculator = StockLinkageCalculator(base_date='20220327', days=20)
    
    linkage_values = []
    total_rows = len(df)
    
    print(f"开始计算 {total_rows} 对股票的联动性...")
    for i, row in df.iterrows():
        stock1 = row['stock_1']
        stock2 = row['stock_2']
        
        linkage = calculator.calculate_linkage(stock1, stock2)
        linkage_values.append(linkage)
        
        if (i + 1) % 10 == 0 or i == total_rows - 1:
            print(f"进度: {i+1}/{total_rows} - 当前: {stock1}-{stock2} = {linkage:.4f}")
        
        time.sleep(0.1)
    
    df['linkage_value'] = linkage_values
    
    # 去除无效值并保存
    df = df.dropna(subset=['linkage_value']).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"计算完成! 结果已保存到 {output_file}")



input_dir = "../data_processed/all/2022/SSedge"
output_dir = "../data_processed/all/2022/SS_edge"

#calculate_SA(input_dir,output_dir)

for filename in os.listdir(input_dir):
    
  # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        # 分离文件名和扩展名（如.txt）
        name, ext = os.path.splitext(filename)
        # 在原始文件名后添加 "_with_MR"
        #output_filename = f"{name}_linkage_value{ext}"
        output_filename=f"{name}{ext}"
        output_path = os.path.join(output_dir, output_filename)
                
        print(f"处理中: {filename} → {output_filename}")
        
        try:
            calculate_weight(input_path, output_path)
            
        except Exception as e:
            print(f"处理文件{filename}时出错: {str(e)}")
