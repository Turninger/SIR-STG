"""
构建行业——股票邻边，IS边，权重为该股票占所属行业的市值比率（market ratio）

"""

import tushare as ts
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os

from marketval_ratio import get_industry_market_ratio,add_market_ratio 

input_dir="../data_processed/all/2022/buquan"
output_dir="../data_processed/all/2022/buquan"

for filename in os.listdir(input_dir):
    
  # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        # 分离文件名和扩展名（如.txt）
        name, ext = os.path.splitext(filename)
        # 在原始文件名后添加 "_with_MR"
        output_filename = f"{name}_with_MR{ext}"
        # output_filename = f"{name}{ext}"
        output_path = os.path.join(output_dir, output_filename)
                
        print(f"处理中: {filename} → {output_filename}")
        
        try:
            add_market_ratio(input_path, output_path)
            
        except Exception as e:
            print(f"处理文件{filename}时出错: {str(e)}")
