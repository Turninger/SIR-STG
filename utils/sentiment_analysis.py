#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 情感分析，构建N-S边特征值
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# 关键词权重调整系数
positive_keywords = {
    "暴涨": 1.5,
    "大幅增长": 1.3,
    "显著提升": 1.2
}

negative_keywords = {
    "暴跌": 1.5,
    "大幅下降": 1.3,
    "显著下滑": 1.2
}

def analyze_sentiment_toward_entities(text, entities, model_name="yiyanghkust/finbert-tone-chinese"):
    """
    分析文本对特定实体的情感倾向
    """

    # 从本地加载分词器和模型
    save_directory = "./finbert_tone_chinese"
    #设置截断 防止字符过长
    tokenizer = AutoTokenizer.from_pretrained(save_directory,truncation=True, max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,truncation=True, max_length=512)
    results = {}
    for entity in entities:
        context = extract_entity_context(text, entity)
        if context:
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

def extract_entity_context(text, entity, window_size=200):
    """
    提取实体周围的上下文
    """
    window_size=200
    matches = [m.start() for m in re.finditer(re.escape(entity), text)]
    if not matches:
        return None
    contexts = []
    for pos in matches:
        start = max(0, pos - window_size)
        end = min(len(text), pos + len(entity) + window_size)
        contexts.append(text[start:end])
    return " ".join(contexts)


"""
    计算情感量化权重，基于关键词的权重调整
    """
def calculate_sentiment_weight(label, confidence, context):
    
    label = label.lower()
    weight = 0.0
    if label == "positive":
        weight = +1.0 * confidence
        for keyword, factor in positive_keywords.items():
            if re.search(keyword, context):
                weight *= factor
                break
    elif label == "negative":
        weight = -1.0 * confidence
        for keyword, factor in negative_keywords.items():
            if re.search(keyword, context):
                weight *= factor
                break
    return weight

# 使用示例
if __name__ == "__main__":
    news_text = """
    万泽股份：2021年度报告净利润9529.24万元，同比暴涨24.20%

    """
    companies = ["万泽股份"]
    sentiment_results = analyze_sentiment_toward_entities(news_text, companies)
    for company, result in sentiment_results.items():
        label = result["sentiment"]
        confidence = result["score"]
        context = result["context"]
        weight = calculate_sentiment_weight(label, confidence, context)
        print(f"公司: {company}")
        print(f"上下文: {result.get('context', '无')}")
        print(f"情感倾向: {result['sentiment']}")
        print(f"置信度: {result['score']:.4f}")
        print(f"权重: {weight}")
        print("-" * 60)
        