#!/usr/bin/env python
# -*- coding: utf-8 -*-
#情感分析，构建N-S边特征值
import pandas as pd
import numpy as np
import ast
import os
from itertools import combinations
import tushare as ts
from tqdm import tqdm
ts.set_token('e5991012344cb5807859d974da3d1a08a98f5c404bf8dace4e7e4ebe')
pro = ts.pro_api()

#N-S边构建并存储至 /CSI100/N_S_edge
from sentiment_analysis import analyze_sentiment_toward_entities,calculate_sentiment_weight


def calculate_SA(news_file, output_file=None):

    # 读取CSV文件
    news_df = pd.read_csv(news_file)
    # news_text=news_df["content"]
    # companies=news_df["stock_name"]

    news_df['sentiment_confidence'] = None
    for idx, row in tqdm(news_df.iterrows(), total=len(news_df)):
        try:
            # 获取当前行的文本和公司名称
            text = str(row['content']).strip()
            company = str(row['stock_name']).strip()  # 确保字符串格式并去除空格
            
            if not company:  # 跳过空公司名
                continue
                
            # 调用情感分析函数
            sentiment_results = analyze_sentiment_toward_entities(text, [company])

            for company, result in sentiment_results.items():
                label = result["sentiment"]
                confidence = result["score"]
                context = result["context"]
                weight = calculate_sentiment_weight(label, confidence, context)
                # print(f"公司: {company}")
                # print(f"上下文: {result.get('context', '无')}")
                # print(f"情感倾向: {result['sentiment']}")
                # print(f"置信度: {result['score']:.4f}")
                # print(f"权重: {weight}")
                # print("-" * 60)
            
            # 提取当前公司的置信度（假设返回格式：{公司: 置信度}）
            news_df.at[idx, 'sentiment_confidence'] = weight
        
        except Exception as e:
            print(f"处理第{idx}行时出错: {str(e)}")
            news_df.at[idx, 'sentiment_confidence'] = None  # 显式标记错误

    news_df.to_csv(output_file, index=False)
    return news_df





input_dir="../data_processed/all/2022/MR"
output_dir="../data_processed/all/2022/MR"

#calculate_SA(input_dir,output_dir)

for filename in os.listdir(input_dir):
    
  # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        # 分离文件名和扩展名（如.txt）
        name, ext = os.path.splitext(filename)
        # 在原始文件名后添加 "_with_MR"
        output_filename = f"{name}_with_SA{ext}"
        output_path = os.path.join(output_dir, output_filename)
                
        print(f"处理中: {filename} → {output_filename}")
        
        try:
            calculate_SA(input_path, output_path)
            
        except Exception as e:
            print(f"处理文件{filename}时出错: {str(e)}")


