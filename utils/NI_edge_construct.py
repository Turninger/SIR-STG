import pandas as pd
import numpy as np
import json
import os
import re
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self, keywords_json_path="industry_keywords.json"):
        # 加载行业关键词词典
        self.industry_keywords = self.load_keywords(keywords_json_path)
        
        # 初始化FinBERT
        self.setup_finbert()
        
        # 情感分析缓存
        self.sentiment_cache = {}
        
        # 调试模式
        self.debug_mode = True
    
    def load_keywords(self, json_path):
        """加载行业关键词词典"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            print(f"已加载 {len(keywords)} 个行业的关键词词典")
            return keywords
        except FileNotFoundError:
            print(f"警告: 关键词文件 {json_path} 不存在")
            return {}
        except Exception as e:
            print(f"加载关键词文件失败: {e}")
            return {}
    
    def setup_finbert(self):
        """初始化FinBERT模型"""
        try:
            # 从本地加载分词器和模型
            save_directory = "./finbert_tone_chinese"
            self.tokenizer = AutoTokenizer.from_pretrained(save_directory, truncation=True, max_length=512)
            self.model = AutoModelForSequenceClassification.from_pretrained(save_directory)
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print("FinBERT模型加载成功")
        except Exception as e:
            print(f"FinBERT模型加载失败: {e}")
            self.finbert_pipeline = None
    
    def preprocess_text(self, text):
        """文本预处理"""
        if not isinstance(text, str):
            return ""
        
        # 去除HTML标签但保留文本内容
        text = re.sub(r'<[^>]+>', '', text)
        # 保留中文字符、标点和基本符号
        text = re.sub(r'[^\w\u4e00-\u9fff，。！？；：、\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_industry_relevance(self, text, industry):
        """计算文本与行业的相关性 - 改进版本"""
        if industry not in self.industry_keywords:
            if self.debug_mode:
                print(f"行业 '{industry}' 不在关键词词典中")
            return 0.0, []
        
        keywords = self.industry_keywords[industry]
        text_lower = text
        
        # 改进的关键词匹配：考虑部分匹配和权重
        match_count = 0
        matched_keywords = []
        total_weight = 0
        
        for keyword in keywords:
            # 检查完整匹配
            if keyword in text_lower:
                match_count += 1
                matched_keywords.append(keyword)
                total_weight += 1.0
            # 检查部分匹配（关键词长度大于2时）
            elif len(keyword) > 2 and any(keyword in word for word in text_lower.split()):
                match_count += 0.5
                matched_keywords.append(f"{keyword}(部分)")
                total_weight += 0.5
        
        # 改进的相关性计算
        if len(keywords) > 0:
            # 基础相关性 + 匹配密度加权
            base_relevance = total_weight / len(keywords)
            density_bonus = min(total_weight / (len(text_lower) / 100), 0.3)  # 文本长度归一化
            relevance = min(base_relevance + density_bonus, 1.0)
        else:
            relevance = 0.0
        
        if self.debug_mode and len(matched_keywords) > 0:
            print(f"行业 '{industry}' 匹配到关键词: {matched_keywords[:3]}...")
        
        return relevance, matched_keywords
    
    def extract_industry_segments(self, text, industry, context_sentences=1):
        """提取与行业相关的文本片段 - 改进版本"""
        if industry not in self.industry_keywords:
            return []
        
        # 使用更细致的句子分割
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # 过滤过短句子
        
        relevant_sentences = []
        keywords = self.industry_keywords[industry]
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence
            keyword_found = False
            
            for keyword in keywords:
                if keyword in sentence_lower:
                    keyword_found = True
                    # 添加上下文句子
                    start_idx = max(0, i - context_sentences)
                    end_idx = min(len(sentences), i + context_sentences + 1)
                    context = "".join(sentences[start_idx:end_idx])
                    relevant_sentences.append(context)
                    break
            
            # 如果找到关键词，跳过接下来的上下文句子避免重复
            if keyword_found:
                i += context_sentences
        
        return list(set(relevant_sentences))  # 去重
    
    def analyze_sentiment_with_finbert(self, text):
        """使用FinBERT分析情感 - 增加调试信息"""
        if self.finbert_pipeline is None:
            if self.debug_mode:
                print("FinBERT管道未初始化")
            return {'label': 'neutral', 'score': 0.5}
            
        if not text or len(text) < 5:
            return {'label': 'neutral', 'score': 0.5}
        
        # 检查缓存
        cache_key = text[:100]
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        try:
            # 限制文本长度，避免过长文本
            if len(text) > 500:
                text = text[:500]
                
            results = self.finbert_pipeline(text, truncation=True, max_length=512)
            
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                label = result['label'].lower()
                score = result['score']
                
                if self.debug_mode:
                    print(f"情感分析: '{text[:50]}...' -> {label}({score:.3f})")
                
                sentiment_result = {'label': label, 'score': score}
                self.sentiment_cache[cache_key] = sentiment_result
                return sentiment_result
            else:
                if self.debug_mode:
                    print(f"情感分析无结果: '{text[:50]}...'")
                return {'label': 'neutral', 'score': 0.5}
                
        except Exception as e:
            if self.debug_mode:
                print(f"情感分析错误: {e}, 文本: '{text[:50]}...'")
            return {'label': 'neutral', 'score': 0.5}
    
    def apply_sentiment_rules(self, text, sentiment_result):
        """应用情感修正规则 - 改进版本"""
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        # 扩展否定词检测
        negation_patterns = [
            '不看好', '难以增长', '不会改善', '没有改善', '未达预期', 
            '不及预期', '低于预期', '表现不佳', '面临压力', '存在风险'
        ]
        
        positive_negation = False
        for pattern in negation_patterns:
            if pattern in text:
                if 'positive' in label:
                    label = 'negative'
                    score = max(0.1, 1 - score)  # 确保有足够的负面强度
                    positive_negation = True
                elif 'negative' in label:
                    score = min(0.9, score * 1.2)  # 加强负面置信度
        
        # 强度词检测
        intensity_words = {
            '大幅': 1.3, '显著': 1.2, '明显': 1.15, '强劲': 1.25,
            '略微': 0.8, '稍微': 0.8, '轻微': 0.7, '略有': 0.85
        }
        
        for word, multiplier in intensity_words.items():
            if word in text:
                if label != 'neutral' and not positive_negation:
                    new_score = score * multiplier
                    score = min(new_score, 0.95) if new_score > score else max(new_score, 0.05)
        
        return {'label': label, 'score': score}
    
    def sentiment_to_numeric(self, sentiment_result):
        """将情感标签转换为数值 (-1 到 1)"""
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        if 'positive' in label:
            return score  # 0 到 1
        elif 'negative' in label:
            return -score  # -1 到 0
        else:
            return 0.0
    
    def calculate_ni_edge_score(self, text, industry):
        """计算NI_edge情感得分 (-1 到 1) - 完全重写"""
        # 预处理文本
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text or len(cleaned_text) < 20:
            if self.debug_mode:
                print("文本过短或为空")
            return 0.0, 0.0, []
        
        # 计算行业相关性
        relevance, matched_keywords = self.calculate_industry_relevance(cleaned_text, industry)
        
        # 大幅降低相关性阈值
        if relevance < 0.01:  # 从0.1降到0.01
            if self.debug_mode:
                print(f"相关性过低: {relevance:.4f}")
            return 0.0, relevance, matched_keywords
        
        if self.debug_mode:
            print(f"行业 '{industry}' 相关性: {relevance:.4f}, 匹配关键词: {len(matched_keywords)}")
        
        # 策略1: 先分析全文情感作为基准
        full_analysis = self.analyze_sentiment_with_finbert(cleaned_text)
        full_sentiment = self.sentiment_to_numeric(full_analysis)
        
        # 策略2: 提取并分析行业相关片段
        relevant_segments = self.extract_industry_segments(cleaned_text, industry)
        
        if not relevant_segments:
            if self.debug_mode:
                print("未找到相关片段，使用全文情感")
            # 没有相关片段，但相关性足够，使用全文情感
            sentiment_result = self.apply_sentiment_rules(cleaned_text, full_analysis)
            sentiment_score = self.sentiment_to_numeric(sentiment_result)
            # 使用相关性调整但不要过度惩罚
            final_score = sentiment_score * min(relevance * 3, 1.0)
            return final_score, relevance, matched_keywords
        
        if self.debug_mode:
            print(f"找到 {len(relevant_segments)} 个相关片段")
        
        # 分析相关片段
        segment_scores = []
        for i, segment in enumerate(relevant_segments):
            if len(segment) > 10:
                sentiment_result = self.analyze_sentiment_with_finbert(segment)
                sentiment_result = self.apply_sentiment_rules(segment, sentiment_result)
                segment_score = self.sentiment_to_numeric(sentiment_result)
                segment_scores.append(segment_score)
                
                if self.debug_mode and i < 3:  # 只打印前3个片段
                    print(f"片段{i+1} 情感: {segment_score:.3f}")
        
        if not segment_scores:
            if self.debug_mode:
                print("相关片段分析无结果")
            return full_sentiment * min(relevance * 2, 1.0), relevance, matched_keywords
        
        # 策略3: 组合全文情感和片段情感
        avg_segment_score = np.mean(segment_scores)
        
        # 如果片段情感与全文情感方向一致，加强信号
        if (avg_segment_score > 0 and full_sentiment > 0) or (avg_segment_score < 0 and full_sentiment < 0):
            combined_score = (avg_segment_score * 0.7 + full_sentiment * 0.3)
        else:
            # 方向不一致时，优先考虑片段情感
            combined_score = avg_segment_score
        
        # 使用相关性调整最终得分，但避免过度压缩
        if abs(combined_score) > 0.1:  # 如果有明显情感倾向
            final_score = combined_score * min(relevance * 1.5, 1.0)
        else:
            final_score = combined_score * relevance
        
        # 确保值域在 (-1, 1)
        final_score = max(min(final_score, 0.99), -0.99)
        
        if self.debug_mode:
            print(f"最终得分: {final_score:.4f} (全文: {full_sentiment:.4f}, 片段平均: {avg_segment_score:.4f})")
        
        return final_score, relevance, matched_keywords
    
    def process_csv(self, csv_path, output_path=None):
        """处理CSV文件"""
        print(f"开始处理CSV文件: {csv_path}")
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        # 检查必要的列
        if 'content' not in df.columns or '行业' not in df.columns:
            raise ValueError("CSV文件必须包含'content'和'行业'列")
        
        # 计算情感得分
        ni_edge_scores = []
        relevance_scores = []
        matched_keywords_list = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:  # 更频繁的进度报告
                print(f"处理进度: {idx}/{total_rows}")
            
            content = row['content']
            industry = row['行业']
            
            # 对前10条数据开启调试模式
            self.debug_mode = (idx < 10)
            
            ni_score, relevance, matched_keywords = self.calculate_ni_edge_score(content, industry)
            ni_edge_scores.append(ni_score)
            relevance_scores.append(relevance)
            matched_keywords_list.append(",".join(matched_keywords[:3]))  # 只保留前3个关键词
            
            # 打印前几条的详细结果用于调试
            if idx < 5:
                print(f"样例 {idx+1}: NI_edge = {ni_score:.4f}, 相关性 = {relevance:.4f}")
        
        # 关闭调试模式
        self.debug_mode = False
        
        # 添加新列
        df['NI_edge'] = ni_edge_scores
        #df['relevance_score'] = relevance_scores
        #df['matched_keywords'] = matched_keywords_list
        
        # 保存结果
        if output_path is None:
            output_path = csv_path.replace('.csv', '_NI.csv')
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"处理完成！结果保存至: {output_path}")
        
        # 打印统计信息
        self.print_statistics(df)
        
        return df
    
    def print_statistics(self, df):
        """打印统计信息"""
        print("\n=== 情感分析统计 ===")
        print(f"总记录数: {len(df)}")
        
        ni_edge = df['NI_edge']
        positive_count = len(ni_edge[ni_edge > 0.01])
        negative_count = len(ni_edge[ni_edge < -0.01])
        neutral_count = len(ni_edge[(ni_edge >= -0.01) & (ni_edge <= 0.01)])
        
        print(f"正面情感 (NI_edge > 0.01): {positive_count}")
        print(f"负面情感 (NI_edge < -0.01): {negative_count}")
        print(f"中性情感 (|NI_edge| <= 0.01): {neutral_count}")
        print(f"平均情感得分: {ni_edge.mean():.4f}")
        print(f"得分标准差: {ni_edge.std():.4f}")
        print(f"得分范围: [{ni_edge.min():.4f}, {ni_edge.max():.4f}]")
        
        if 'relevance_score' in df.columns:
            print(f"平均相关性得分: {df['relevance_score'].mean():.4f}")

# 使用示例
# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = SentimentAnalyzer("industry_keywords.json")
    
    # 设置输入和输出目录
    input_dir = "../data_processed/all/2022/NS_IS_edge"
    output_dir = "../data_processed/all/2022/NS_IS_edge"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入目录下所有CSV文件
    import glob
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    print(f"在目录 {input_dir} 中找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        print(f"\n处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # 设置输出文件路径
            filename = os.path.basename(csv_file)
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}_NI.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # 处理CSV文件
            result_df = analyzer.process_csv(csv_file, output_path)
            print(f"成功处理: {filename}")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            continue
    
    print(f"\n所有文件处理完成！")