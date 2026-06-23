#!/usr/bin/env python
# -*- coding: utf-8 -*-
#情感分析，构建N-S边特征值
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import re

def analyze_sentiment_toward_entities(text, entities, model_name="yiyanghkust/finbert-tone-chinese"):
    """
    分析文本对特定实体的情感倾向
    
    参数:
        text: 待分析文本
        entities: 实体列表，如["通威股份", "特变电工", "大全能源"]
        model_name: 使用的BERT模型
    
    返回:
        dict: 各实体的情感分析结果
    """
    # 初始化模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # 存储结果
    results = {}
    
    for entity in entities:
        # 提取包含实体的句子或上下文
        context = extract_entity_context(text, entity)
        
        if context:
            # 分析情感
            try:
                sentiment_result = nlp(context)[0]
                results[entity] = {
                    "sentiment": sentiment_result["label"],
                    "score": sentiment_result["score"],
                    "context": context
                }
            except:
                results[entity] = {
                    "sentiment": "neutral",
                    "score": 0.5,
                    "context": context,
                    "error": "分析失败"
                }
        else:
            results[entity] = {
                "sentiment": "neutral",
                "score": 0.5,
                "context": None,
                "note": "文本中未找到该实体"
            }
    
    return results

def extract_entity_context(text, entity, window_size=100):
    """
    提取实体周围的上下文
    
    参数:
        text: 完整文本
        entity: 目标实体
        window_size: 上下文窗口大小(字符数)
    
    返回:
        str: 包含实体的上下文文本
    """
    # 找到实体所有出现位置
    matches = [m.start() for m in re.finditer(re.escape(entity), text)]
    
    if not matches:
        return None
    
    # 收集所有上下文片段
    contexts = []
    for pos in matches:
        start = max(0, pos - window_size)
        end = min(len(text), pos + len(entity) + window_size)
        contexts.append(text[start:end])
    
    # 合并所有上下文片段
    return " ".join(contexts)

def calculate_sentiment_weight(label, confidence):
    """
    计算情感量化权重
    
    参数:
        label: 情感标签 (positive/negative/neutral)
        confidence: 置信度分数 (0-1)
    
    返回:
        float: 量化权重值 (-1 到 +1)
    """
    # 情感标签映射
    label = label.lower()
    
    if label == "positive":
        # 积极情感: +1 × 置信度
        return +1.0 * confidence
    elif label == "negative":
        # 消极情感: -1 × 置信度
        return -1.0 * confidence
    else:
        # 中性情感: 0
        # 可考虑添加细微调整 (见下文)
        return 0.0

# 使用示例
if __name__ == "__main__":
    # 示例文本
    news_text = """
    通威股份暴涨", "特变电工暴跌", "大全能源稳定上涨
    """
   
    # 要分析的公司
    companies = ["通威股份", "特变电工", "大全能源"]
    
    # 进行分析
    sentiment_results = analyze_sentiment_toward_entities(news_text, companies)
    #label = sentiment_results.result['sentiment']
    #confidence = sentiment_results["score"]
                
    # 计算量化权重
    #weight = calculate_sentiment_weight(label, confidence)
    
    # 打印结果
    for company, result in sentiment_results.items():
        print(f"公司: {company}")
        print(f"上下文: {result.get('context', '无')}")
        print(f"情感倾向: {result['sentiment']}")
        print(f"置信度: {result['score']:.4f}")
        #print(f"权重：{weight}")
        print("-" * 60)