import os
import pandas as pd
import numpy as np
from volcenginesdkarkruntime import Ark
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import jieba
import jieba.posseg as pseg
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class IndustrySentimentAnalyzer:
    def __init__(self):
        # 初始化豆包客户端
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="bab6b861-266d-43c7-970d-04a0692e93f2"
        )
        
        # 行业关键词词典
        self.industry_keywords = {}
        
        # 初始化FinBERT
        self.setup_finbert()
        
        # 金融实体词典
        #self.finance_entities = self.load_finance_entities()
        
    def setup_finbert(self):
        """初始化FinBERT模型"""

        # 从本地加载分词器和模型
        save_directory = "./finbert_tone_chinese"
        #设置截断 防止字符过长
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory,truncation=True, max_length=512)
        self.model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        self.finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )
        print("FinBERT模型加载成功")
    
    def load_finance_entities(self):
        """加载金融实体词典"""
        finance_terms = [
            '上证指数', '深证成指', '创业板指', 'A股', '券商', '证券', '白酒', '消费板块',
            '房地产', '建材', '旅游', '酒店', '商贸零售', '食品饮料', '农林牧渔',
            '银行', '保险', '基金', '信托', '期货', '期权', '股票', '债券'
        ]
        return set(finance_terms)
    
    def generate_industry_keywords(self, industry, max_retries=3):
        """使用豆包模型生成行业关键词"""
        prompt = f"""
        请为【{industry}】行业生成一个全面的关键词列表。
        要求：
        1. 包含行业核心术语、相关公司、产品服务、政策概念等
        2. 包含直接相关和间接相关的词汇
        3. 包含产业链上下游相关概念
        4. 每个关键词用中文逗号分隔
        5. 只返回关键词，不要额外解释
        
        示例格式：传媒,媒体,广告,影视,出版,新媒体,短视频,内容制作,IP运营
        """
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="doubao-seed-1-6-vision-250815",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                
                response = completion.choices[0].message.content.strip()
                keywords = [kw.strip() for kw in response.split('，') if kw.strip()]
                keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
                
                if keywords:
                    print(f"为行业【{industry}】生成 {len(keywords)} 个关键词")
                    return keywords
                    
            except Exception as e:
                print(f"第{attempt+1}次尝试失败: {e}")
                continue
                
        # 如果API调用失败，返回基础关键词
        base_keywords = self.get_base_keywords(industry)
        print(f"使用基础关键词 for {industry}: {len(base_keywords)} 个")
        return base_keywords
    
    def get_base_keywords(self, industry):
        """获取行业基础关键词（备用）"""
        base_keyword_map = {
            '传媒': ['传媒', '媒体', '广告', '影视', '出版', '新媒体', '短视频', '内容', 'IP'],
            '金融': ['金融', '银行', '证券', '保险', '基金', '信托', '投资', '信贷', '理财'],
            '科技': ['科技', '技术', '创新', '研发', '软件', '硬件', '互联网', '数字化'],
            '消费': ['消费', '零售', '电商', '品牌', '商品', '服务', '用户', '市场'],
            '医药': ['医药', '医疗', '药品', '医院', '健康', '生物', '制药', '器械']
        }
        return base_keyword_map.get(industry, [industry])
    
    def build_industry_keywords(self, industries):
        """构建行业关键词词典"""
        print("开始构建行业关键词词典...")
        
        for industry in set(industries):
            if industry not in self.industry_keywords:
                self.industry_keywords[industry] = self.generate_industry_keywords(industry)
        
        print(f"关键词词典构建完成，共 {len(self.industry_keywords)} 个行业")
        return self.industry_keywords
    
    def preprocess_text(self, text):
        """文本预处理"""
        if not isinstance(text, str):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除特殊字符和多余空格
        text = re.sub(r'[^\w\u4e00-\u9fff，。！？；：、]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_industry_relevance(self, text, industry):
        """计算文本与行业的相关性"""
        if industry not in self.industry_keywords:
            return 0.0
        
        keywords = self.industry_keywords[industry]
        text_lower = text.lower()
        
        # 关键词匹配计数
        match_count = 0
        for keyword in keywords:
            if keyword in text_lower:
                match_count += 1
        
        # 计算相关性分数 (0-1)
        relevance = min(match_count / len(keywords) * 2, 1.0)  # 乘以2是为了放大相关性
        
        return relevance
    
    def extract_industry_segments(self, text, industry):
        """提取与行业相关的文本片段"""
        if industry not in self.industry_keywords:
            return []
        
        sentences = re.split(r'[。！？；]', text)
        relevant_sentences = []
        keywords = self.industry_keywords[industry]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword in sentence_lower:
                    relevant_sentences.append(sentence.strip())
                    break
        
        return relevant_sentences
    
    def analyze_sentiment_with_finbert(self, text):
        """使用FinBERT分析情感"""
        if not text or len(text) < 10:  # 文本太短可能分析不准确
            return {'label': 'neutral', 'score': 0.5, 'raw_scores': [0.33, 0.33, 0.34]}
        
        try:
            results = self.finbert_pipeline(text, truncation=True, max_length=512)
            
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                label = result['label'].lower()
                score = result['score']
                
                # 根据标签返回对应的情感分数
                if 'positive' in label:
                    return {'label': 'positive', 'score': score, 'raw_scores': [score, 0, 1-score]}
                elif 'negative' in label:
                    return {'label': 'negative', 'score': score, 'raw_scores': [0, score, 1-score]}
                else:
                    return {'label': 'neutral', 'score': score, 'raw_scores': [0, 0, score]}
            else:
                return {'label': 'neutral', 'score': 0.5, 'raw_scores': [0.33, 0.33, 0.34]}
                
        except Exception as e:
            print(f"情感分析错误: {e}")
            return {'label': 'neutral', 'score': 0.5, 'raw_scores': [0.33, 0.33, 0.34]}
    
    def apply_sentiment_rules(self, text, sentiment_result):
        """应用情感修正规则"""
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        # 否定词检测
        negation_words = ['不', '没', '未', '无', '非', '难以', '不会', '不能', '没有']
        negation_patterns = ['不看好', '难以增长', '不会改善', '没有改善', '未达预期']
        
        for pattern in negation_patterns:
            if pattern in text:
                if label == 'positive':
                    label = 'negative'
                    score = 1 - score  # 反转置信度
                elif label == 'negative':
                    label = 'positive'
                    score = 1 - score
        
        # 强度词检测
        intensity_words = {
            '大幅': 1.3, '显著': 1.2, '明显': 1.1, '略微': 0.8, '稍微': 0.8, '轻微': 0.7
        }
        
        for word, multiplier in intensity_words.items():
            if word in text:
                if label != 'neutral':
                    score = min(score * multiplier, 0.95)  # 限制最大置信度
        
        return {'label': label, 'score': score}
    
    def calculate_final_sentiment_score(self, text, industry):
        """计算最终的情感得分 (-1 到 1)"""
        # 预处理文本
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return 0.0
        
        # 计算行业相关性
        relevance = self.calculate_industry_relevance(cleaned_text, industry)
        
        if relevance < 0.1:  # 相关性太低，返回中性
            return 0.0
        
        # 提取行业相关片段
        relevant_segments = self.extract_industry_segments(cleaned_text, industry)
        
        if not relevant_segments:  # 没有相关片段，使用全文分析但降低权重
            full_analysis = self.analyze_sentiment_with_finbert(cleaned_text)
            sentiment_result = self.apply_sentiment_rules(cleaned_text, full_analysis)
            sentiment_score = self.sentiment_to_numeric(sentiment_result)
            return sentiment_score * relevance * 0.5  # 降低权重
        
        # 分析相关片段
        segment_scores = []
        for segment in relevant_segments:
            if len(segment) > 10:  # 只分析有内容的片段
                sentiment_result = self.analyze_sentiment_with_finbert(segment)
                sentiment_result = self.apply_sentiment_rules(segment, sentiment_result)
                segment_score = self.sentiment_to_numeric(sentiment_result)
                segment_scores.append(segment_score)
        
        if not segment_scores:
            return 0.0
        
        # 计算加权平均得分
        avg_score = np.mean(segment_scores)
        final_score = avg_score * relevance
        
        # 值域压缩到 (-1, 1)
        final_score = max(min(final_score, 0.99), -0.99)
        
        return final_score
    
    def sentiment_to_numeric(self, sentiment_result):
        """将情感标签转换为数值 (-1 到 1)"""
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        if label == 'positive':
            return score  # 0 到 1
        elif label == 'negative':
            return -score  # -1 到 0
        else:  # neutral
            return 0.0
    
    def process_csv(self, csv_path, output_path=None):
        """处理CSV文件"""
        print(f"开始处理CSV文件: {csv_path}")
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        # 检查必要的列
        if 'content' not in df.columns or '行业' not in df.columns:
            raise ValueError("CSV文件必须包含'content'和'行业'列")
        
        # 构建行业关键词词典
        industries = df['行业'].dropna().unique()
        self.build_industry_keywords(industries)
        
        # 计算情感得分
        ni_edge_scores = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{total_rows}")
            
            content = row['content']
            industry = row['行业']
            
            score = self.calculate_final_sentiment_score(content, industry)
            ni_edge_scores.append(score)
        
        # 添加新列
        df['NI_edge'] = ni_edge_scores
        
        # 保存结果
        if output_path is None:
            output_path = csv_path.replace('.csv', '_with_NI.csv')
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"处理完成！结果保存至: {output_path}")
        
        # 打印统计信息
        self.print_statistics(df)
        
        return df
    
    def print_statistics(self, df):
        """打印统计信息"""
        print("\n=== 情感分析统计 ===")
        print(f"总记录数: {len(df)}")
        print(f"正面情感 (NI_edge > 0): {len(df[df['NI_edge'] > 0])}")
        print(f"负面情感 (NI_edge < 0): {len(df[df['NI_edge'] < 0])}")
        print(f"中性情感 (NI_edge = 0): {len(df[df['NI_edge'] == 0])}")
        print(f"平均情感得分: {df['NI_edge'].mean():.4f}")
        print(f"得分标准差: {df['NI_edge'].std():.4f}")
        print(f"得分范围: [{df['NI_edge'].min():.4f}, {df['NI_edge'].max():.4f}]")

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = IndustrySentimentAnalyzer()
    
    # 处理CSV文件
    csv_file_path = "../data_processed/all/2022/NS_IS_edge/news_20220101_with_MR_with_SA.csv"  # 替换为您的CSV文件路径
    try:
        result_df = analyzer.process_csv(csv_file_path)
        print("情感分析完成！")
    except Exception as e:
        print(f"处理失败: {e}")