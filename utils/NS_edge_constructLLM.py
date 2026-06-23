#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 增强版金融情感分析：多级分类+动态权重
import pandas as pd
import json
import time
from tqdm import tqdm
import openai
from openai import OpenAI
# 升级方舟 SDK 到最新版本 pip install -U 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark


import hashlib

# 初始化API客户端
client = OpenAI(api_key="bab6b861-266d-43c7-970d-04a0692e93f2")

# 情感分类体系（7类）
SENTIMENT_CATEGORIES = {
    "强烈正面": {
        "description": "明确利好，如业绩大增、重大创新、政策支持",
        "weight_factor": 1.2,
        "score_threshold": 0.9
    },
    "一般正面": {
        "description": "普通积极信息，如小幅增长、行业向好",
        "weight_factor": 1.0,
        "score_threshold": 0.7
    },
    "轻微正面": {
        "description": "微弱积极信号，如中性偏好的表述",
        "weight_factor": 0.6,
        "score_threshold": 0.55
    },
    "中性": {
        "description": "事实陈述无倾向，或正负因素抵消",
        "weight_factor": 0.2,
        "score_threshold": None
    },
    "轻微负面": {
        "description": "微弱消极信号，如中性偏负的表述",
        "weight_factor": -0.5,
        "score_threshold": 0.55
    },
    "一般负面": {
        "description": "普通负面信息，如业绩下滑、行业风险",
        "weight_factor": -1.0,
        "score_threshold": 0.7
    },
    "强烈负面": {
        "description": "重大利空，如财务造假、监管处罚、重大亏损",
        "weight_factor": -1.5,
        "score_threshold": 0.9
    }
}

# 动态权重计算（考虑情感类别、置信度和上下文强度）
def calculate_dynamic_weight(sentiment_data):
    """
    参数:
        sentiment_data: {
            "sentiment": 情感类别,
            "score": 置信度(0-1),
            "context": 分析理由,
            "intensity": 上下文强度(1-3)
        }
    返回:
        加权后的情感值
    """
    category = sentiment_data["sentiment"]
    base_factor = SENTIMENT_CATEGORIES[category]["weight_factor"]
    confidence = sentiment_data["score"]
    
    # 强度调整 (1:弱, 2:中, 3:强)
    intensity = sentiment_data.get("intensity", 2)
    intensity_multiplier = 0.8 + 0.2 * intensity
    
    # 非线性置信度调整 (高置信度更敏感)
    confidence_adjusted = confidence ** 1.5
    
    # 金融领域特殊调整
    financial_adjustment = 1.0
    context = sentiment_data["context"].lower()
    
    # 对关键金融术语的额外加权
    financial_terms = {
        "财报": 1.1, "盈利": 1.1, "亏损": 1.3, 
        "并购": 1.2, "诉讼": 1.3, "监管": 1.4
    }
    
    for term, factor in financial_terms.items():
        if term in context:
            financial_adjustment *= factor
            break
    
    return base_factor * confidence_adjusted * intensity_multiplier * financial_adjustment

# 增强版OpenAI情感分析
def enhanced_sentiment_analysis(text, company):
    """
    返回结构:
    {
        "company": {
            "sentiment": 情感类别,
            "score": 置信度,
            "context": 分析理由,
            "intensity": 强度等级,
            "weight": 计算后的权重,
            "triggers": [触发情感的关键词列表]
        }
    }
    """
    prompt = f"""作为资深金融分析师，请对以下内容进行情感分析：
    
分析对象：{company}
分析要求：
1. 情感分类（严格选择其一）：
   - 强烈正面：明确重大利好
   - 一般正面：普通积极信息  
   - 轻微正面：微弱积极信号
   - 中性：无明确倾向
   - 轻微负面：微弱消极信号
   - 一般负面：普通负面信息
   - 强烈负面：重大利空事件

2. 评估置信度(0-1)
3. 判断情感强度(1-3级)
4. 提取3个触发情感的关键词
5. 用20字内说明理由

返回JSON格式示例：
{{
    "sentiment": "强烈正面",
    "score": 0.95,
    "intensity": 3,
    "triggers": ["业绩大增", "创新突破", "市场份额"],
    "context": "财报显示利润同比增长200%"
}}

新闻内容：
{text}
"""
    
    for _ in range(3):  # 重试机制
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "你是有10年经验的金融分析师，擅长从新闻中识别细微情感倾向。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 验证结果格式
            required_keys = ["sentiment", "score", "context", "intensity", "triggers"]
            if all(k in result for k in required_keys):
                # 确保情感类别有效
                if result["sentiment"] not in SENTIMENT_CATEGORIES:
                    raise ValueError("无效情感类别")
                
                # 计算最终权重
                result["weight"] = calculate_dynamic_weight(result)
                return {company: result}
            
            raise ValueError("返回数据不完整")
            
        except Exception as e:
            print(f"分析出错: {str(e)}，重试中...")
            time.sleep(2)
    
    # 默认返回中性结果
    return {company: {
        "sentiment": "中性",
        "score": 0.5,
        "intensity": 1,
        "triggers": [],
        "context": "分析失败默认值",
        "weight": 0
    }}

# 带缓存的分析处理
def analyze_with_cache(text, company, cache_dict):
    cache_key = f"{company}_{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key not in cache_dict:
        cache_dict[cache_key] = enhanced_sentiment_analysis(text, company)
    return cache_dict[cache_key]

# 主处理函数
def calculate_enhanced_SA(news_file, output_file):
    df = pd.read_csv(news_file)
    
    # 结果列
    result_cols = [
        'sentiment_category', 'sentiment_score',
        'sentiment_intensity', 'sentiment_triggers',
        'sentiment_context', 'sentiment_weight'
    ]
    
    for col in result_cols:
        df[col] = None
    
    cache = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = str(row['content']).strip()
            company = str(row['stock_name']).strip()
            
            if not company or not text:
                continue
                
            result = analyze_with_cache(text, company, cache)
            data = result[company]
            
            # 填充结果
            df.at[idx, 'sentiment_category'] = data["sentiment"]
            df.at[idx, 'sentiment_score'] = data["score"]
            df.at[idx, 'sentiment_intensity'] = data["intensity"]
            df.at[idx, 'sentiment_triggers'] = ", ".join(data["triggers"])
            df.at[idx, 'sentiment_context'] = data["context"]
            df.at[idx, 'sentiment_weight'] = data["weight"]
            
            # 调试输出
            if idx % 10 == 0:
                print(f"\n样本{idx}分析结果:")
                print(f"公司: {company}")
                print(f"分类: {data['sentiment']}(强度{data['intensity']})")
                print(f"关键词: {data['triggers']}")
                print(f"权重: {data['weight']:.2f}")
                print("-"*50)
                
        except Exception as e:
            print(f"行{idx}处理失败: {str(e)}")
    
    df.to_csv(output_file, index=False)
    print(f"分析完成！结果保存至: {output_file}")
    print(f"情感分布统计:\n{df['sentiment_category'].value_counts()}")
    return df

# 示例执行
if __name__ == "__main__":
    input_path = "../data_processed/CSI100/NS_edge/news_20220328.csv"
    output_path = "../data_processed/CSI100/NS_edge/news_20220328_enhanced_SA.csv"
    
    calculate_enhanced_SA(input_path, output_path)