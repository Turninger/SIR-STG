"""
将CSI指数.csv中的股票代码进行归一化并保存,获取CSI100指数列表

"""
import pandas as pd
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
import os
import pandas as pd
import ast

class EXCHANGE():
    XSHG = 'XSHG'
    SSE = 'XSHG'
    SH = 'SH'
    XSHE = 'XSHE'
    SZ = 'SZ'
    SZE = 'XSHE'
    
def normalize_code(symbol, pre_close=None):
    """
    归一化证券代码
    
    :param symbol: 如 '1' 或 '000001'
    :param pre_close: 前收盘价（用于推断指数）
    :return: 证券代码的全称 如 '000001.SZ'
    """
    if not isinstance(symbol, str):
        return symbol

    # 处理带交易所前缀的代码（如 sz000001）
    if symbol.startswith('sz') and len(symbol) == 8:
        normalized = symbol[2:8].zfill(6)
        return f"{normalized}.{EXCHANGE.SZ}"
    elif symbol.startswith('sh') and len(symbol) == 8:
        normalized = symbol[2:8].zfill(6)
        return f"{normalized}.{EXCHANGE.SH}"

    # 尝试补全为6位代码（如 '1' → '000001'）
    if len(symbol) < 6:
        symbol = symbol.zfill(6)  # 左侧补零到6位

    # 仅处理6位代码
    if len(symbol) != 6:
        return symbol  # 非6位代码直接返回

    # 规则匹配
    if symbol.startswith('00'):
        if pre_close and pre_close > 2000:  # 推断为上证指数
            return f"{symbol}.{EXCHANGE.SH}"
        else:
            return f"{symbol}.{EXCHANGE.SZ}"
    elif symbol.startswith(('399', '159', '150', '16', '184801', '201872')):
        return f"{symbol}.{EXCHANGE.SZ}"
    elif symbol.startswith(('50', '51', '60', '688', '900')) or symbol == '751038':
        return f"{symbol}.{EXCHANGE.SH}"
    elif symbol[:3] in ['000', '001', '002', '200', '300']:
        return f"{symbol}.{EXCHANGE.SZ}"
    else:
        return f"{symbol}.{EXCHANGE.SZ}"  # 默认归属深交所（根据需求调整）
    

###输入要修改的CSI指数
csi100=pd.read_csv("../CSIA100.csv")
csi100["成份券代码Constituent Code"] = csi100["成份券代码Constituent Code"].astype(str).apply(normalize_code)
# 保存结果
output_path = '../CSIA100_normalized.csv'
csi100.to_csv(output_path, index=False, encoding='utf-8-sig')