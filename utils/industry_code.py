#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import tushare as ts
from tqdm import tqdm
import os

"""
补全行业信息，csv中的“行业”列,获取行业代码

"""

# 设置Tushare Token
ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')
pro = ts.pro_api()

def complete_industry_info(input_file, output_file=None):
    """
    根据股票代码补全行业信息（假设股票代码已符合Tushare格式）
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出文件路径(可选)
    """
    # 读取输入文件
    df = pd.read_csv(input_file)
    
    # 获取唯一股票代码
    stock_codes = df['stock_code'].unique().tolist()

    print(f"发现 {len(stock_codes)} 个唯一股票代码")
    
    # 获取行业信息映射
    industry_mapping = {}
    industry_code_mapping = {}
    for code in tqdm(stock_codes, desc="获取行业信息"):
        try:
            # 直接使用原始股票代码查询
            data = pro.index_member_all(ts_code=code, fields="ts_code,l1_name,l1_code")
            if not data.empty:
                industry_mapping[code] = data.iloc[0]['l1_name']
                industry_code_mapping[code] = data.iloc[0]['l1_code']
        except Exception as e:
            print(f"股票 {code} 查询失败: {str(e)}")
            industry_mapping[code] = None
    
    # 补全行业信息
    df['行业'] = df['stock_code'].map(industry_mapping)
    df['industry_code'] = df['stock_code'].map(industry_code_mapping)
    
    # 处理未获取到的行业
    unknown_count = df['行业'].isna().sum()
    if unknown_count > 0:
        print(f"警告: {unknown_count} 个股票未获取到行业信息")
        df['行业'] = df['行业'].fillna('未知行业')
    
    # 保存结果
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"结果已保存至: {output_file}")
    
    return df

if __name__ == "__main__":

    input_dir="../data_processed/all/2022/MR"
    output_dir="../data_processed/all/2022/MR"

    for filename in os.listdir(input_dir):
        
        # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        output_filename = filename
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"处理中: {filename} → {output_filename}")
        
        try:
            complete_industry_info(input_path, output_path)
            
        except Exception as e:
            print(f"处理文件{filename}时出错: {str(e)}")

#数据引入




# # 文件路径配置
# input_file="../data_processed/CSI100/NS_IS_edge/news_20220328_with_MR_with_SA.csv"
# output_file="../data_processed/CSI100/NS_IS_edge/news_20220328_with_MR_with_SA_industry.csv"

# # 执行行业信息补全
# result_df = complete_industry_info(input_file, output_file)

# # 显示行业分布
# print("\n行业分布统计:")
# print(result_df['行业'].value_counts())

# # 显示样本结果
# print("\n样本数据预览:")
# print(result_df[['stock_code', '行业']].head())