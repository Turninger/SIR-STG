import re
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from tqdm import tqdm

class IndustrySentimentAnalyzer:
    def __init__(self, target_industry, device=None):
        """
        初始化行业情感分析器
        
        参数:
            target_industry: str - 目标行业名称 (如 "金融", "科技")
            device: str - 指定计算设备 ("cuda" 或 "cpu")
        """
        self.target_industry = target_industry
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "？", "！", ";", "；", "，", "、"],
            length_function=self._count_chinese_chars
        )
        # 从本地加载分词器和模型
        save_directory = "./finbert_tone_chinese"
        # 初始化情感分析模型 (中文金融情感分析)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(save_directory)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model=self.sentiment_model,
            tokenizer=self.sentiment_tokenizer,
            device=self.device
        )
        
        # 初始化语义相似度模型
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 行业关键词库 (可根据需要扩展)
        self.industry_keywords = {
            "金融": ["银行", "证券", "保险", "信贷", "投资", "股市", "基金", "债券", "金融", "理财"],
            "科技": ["科技", "半导体", "芯片", "人工智能", "AI", "云计算", "大数据", "5G", "物联网", "创新"],
            "医疗": ["医疗", "医药", "医院", "疫苗", "生物", "健康", "诊断", "治疗", "药品"],
            "能源": ["石油", "煤炭", "天然气", "电力", "新能源", "太阳能", "风能", "电池", "储能"],
            "消费": ["零售", "电商", "消费", "品牌", "市场", "需求", "用户", "产品", "销售"],
            '电力设备':[],
            '建筑材料':[],
            '商贸零售':[],
            '基础化工':[],
            '交通运输':[], 
            '有色金属':[], 
            '机械设备':[], 
            '美容护理':[],
            '电子':[], 
            '计算机':[], 
            '国防军工':[], 
            '纺织服饰':[], 
            '医药生物':[], 
            '非银金融':[], 
            '汽车':[], 
            '轻工制造':[], 
            '公用事业':[],
            '石油石化':[], 
            '房地产':[], 
            '传媒':[], 
            '家用电器':[], 
            '食品饮料':[], 
            '通信':[], 
            '农林牧渔':[], 
            '钢铁':[], 
            '建筑装饰':[],
            '综合':[], 
            '环保':[], 
            '煤炭':[], 
            '银行':[], 
            '社会服务':[]
        }
    
    def _count_chinese_chars(self, text):
        """计算中文字符数量（更准确的中文长度估计）"""
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars)
    
    def _enhance_text_with_context(self, text):
        """增强文本上下文，添加行业提示"""
        return f"本文讨论[{self.target_industry}]行业：" + text
    
    def _calculate_keyword_density(self, text):
        """计算行业关键词密度"""
        if self.target_industry not in self.industry_keywords:
            return 1.0  # 默认权重
        
        keywords = self.industry_keywords[self.target_industry]
        total_words = len(re.findall(r'\w+', text))
        if total_words == 0:
            return 1.0
            
        keyword_count = sum(1 for word in keywords if word in text)
        density = keyword_count / total_words
        # 密度转换为权重 (0.5-1.5范围)
        return 0.5 + min(density * 10, 1.0)
    
    def _analyze_chunk_sentiment(self, chunk):
        """分析单个文本块的情感"""
        try:
            result = self.sentiment_analyzer(
                chunk,
                truncation=True,
                max_length=512,
                padding="max_length"
            )[0]
            
            # 将情感标签转换为数值
            label_map = {
                "Positive": 1,
                "Neutral": 0,
                "Negative": -1
            }
            
            return {
                "sentiment_score": label_map.get(result['label'], 0),
                "confidence": result['score'],
                "text": chunk
            }
        except Exception as e:
            print(f"情感分析出错: {e}")
            return {
                "sentiment_score": 0,
                "confidence": 0.5,
                "text": chunk
            }
    
    def analyze(self, text, verbose=False):
        """
        分析长文本对目标行业的情感
        
        参数:
            text: str - 输入的长文本
            verbose: bool - 是否输出详细分析过程
            
        返回:
            dict - 包含分析结果
        """
        # 1. 文本增强和分割
        enhanced_text = self._enhance_text_with_context(text)
        chunks = self.text_splitter.split_text(enhanced_text)
        
        if verbose:
            print(f"分割为 {len(chunks)} 个文本块")
            for i, chunk in enumerate(chunks):
                print(f"\n文本块 #{i+1} (长度: {len(chunk)}):")
                print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        # 2. 分块情感分析
        results = []
        for chunk in tqdm(chunks, desc="分析文本块情感", disable=not verbose):
            chunk_result = self._analyze_chunk_sentiment(chunk)
            results.append(chunk_result)
        
        # 3. 计算语义相关性权重
        industry_query = f"{self.target_industry}行业的市场表现和发展前景"
        query_embedding = self.similarity_model.encode([industry_query])[0]
        chunk_embeddings = self.similarity_model.encode([res['text'] for res in results])
        
        similarities = []
        for emb in chunk_embeddings:
            cos_sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(cos_sim)
        
        # 4. 计算关键词密度权重
        keyword_weights = [self._calculate_keyword_density(res['text']) for res in results]
        
        # 5. 加权聚合情感分数
        weighted_scores = []
        for res, sim_weight, kw_weight in zip(results, similarities, keyword_weights):
            # 综合权重 = 语义相关性权重 * 关键词密度权重
            combined_weight = sim_weight * kw_weight
            
            # 情感分数 = 情感方向 * 置信度 * 综合权重
            chunk_score = res['sentiment_score'] * res['confidence'] * combined_weight
            weighted_scores.append(chunk_score)
            
            if verbose:
                sentiment_label = "积极" if res['sentiment_score'] > 0 else "消极" if res['sentiment_score'] < 0 else "中性"
                print(f"\n文本块: {res['text'][:100]}...")
                print(f"情感: {sentiment_label}, 置信度: {res['confidence']:.2f}, "
                      f"语义相关性: {sim_weight:.2f}, 关键词密度权重: {kw_weight:.2f}, "
                      f"贡献分数: {chunk_score:.4f}")
        
        # 6. 计算最终情感分数 (-1到1之间)
        final_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        
        # 7. 确定情感倾向
        if final_score > 0.2:
            sentiment = "积极"
        elif final_score < -0.2:
            sentiment = "消极"
        else:
            sentiment = "中性"
        
        # 8. 识别关键段落
        top_indices = np.argsort(similarities)[-3:][::-1]
        key_paragraphs = [{
            "text": results[i]['text'],
            "sentiment": "积极" if results[i]['sentiment_score'] > 0 else "消极" if results[i]['sentiment_score'] < 0 else "中性",
            "relevance": float(similarities[i]),
            "contribution": float(weighted_scores[i])
        } for i in top_indices]
        
        # 9. 构建结果
        return {
            "industry": self.target_industry,
            "overall_sentiment": sentiment,
            "sentiment_score": float(final_score),
            "key_paragraphs": key_paragraphs,
            "chunk_details": [{
                "text": r['text'],
                "sentiment_score": r['sentiment_score'],
                "confidence": r['confidence'],
                "semantic_relevance": float(s),
                "keyword_weight": float(kw),
                "weighted_score": float(ws)
            } for r, s, kw, ws in zip(results, similarities, keyword_weights, weighted_scores)]
        }


# ======================== 使用示例 ========================
if __name__ == "__main__":
    # 示例长文本 (实际应用中可替换为任意长文本)
    long_text = """
    近年来，全球经济格局发生深刻变化，各行业面临前所未有的挑战与机遇。在金融领域，数字化转型加速推进，
    传统银行业务模式受到金融科技的强烈冲击。移动支付、区块链技术、人工智能在风险控制中的应用等创新不断涌现，
    为金融服务带来革命性变革。与此同时，监管环境日趋严格，反洗钱和合规要求不断提高，增加了金融机构的运营成本。
    
    在科技创新方面，人工智能和大数据技术正在重塑产业生态。半导体行业作为科技创新的基础，面临着供应链安全的挑战。
    全球芯片短缺问题凸显了产业链布局的重要性，各国纷纷加大在半导体制造领域的投入。与此同时，5G技术的商用化进程加速，
    为物联网、自动驾驶等新兴应用场景提供了基础设施支持。
    
    能源转型成为全球共识，新能源产业迎来爆发式增长。太阳能和风能发电成本持续下降，储能技术取得突破性进展。
    传统能源企业纷纷布局新能源业务，石油巨头加大在可再生能源领域的投资力度。然而，能源转型也带来了电网稳定性、
    储能技术成熟度等新挑战。
    
    在医疗健康领域，生物技术和基因编辑技术的突破为疾病治疗带来新希望。新冠疫情期间，mRNA疫苗技术的成功应用
    展示了生物医药创新的巨大潜力。数字医疗、远程诊疗等新模式快速发展，提升了医疗服务的可及性。
    
    消费市场呈现多元化趋势，电子商务持续渗透，社交电商、直播带货等新模式崛起。消费者对产品品质和个性化需求不断提高，
    品牌建设成为企业核心竞争力。同时，可持续消费理念日益普及，环保和ESG因素成为消费者决策的重要考量。
    
    总体而言，各行业在技术驱动下加速变革，创新与合规成为企业发展的双轮驱动。未来，能够快速适应变化、
    把握技术趋势的企业将在竞争中占据优势地位。
    """
    
    # 初始化分析器 (分析对"金融"行业的情感)
    analyzer = IndustrySentimentAnalyzer(target_industry="金融")
    
    # 执行分析 (verbose=True 输出详细过程)
    result = analyzer.analyze(long_text, verbose=True)
    
    # 打印最终结果
    print("\n" + "="*50)
    print(f"行业: {result['industry']}")
    print(f"总体情感: {result['overall_sentiment']} (分数: {result['sentiment_score']:.4f})")
    
    print("\n关键段落分析:")
    for i, para in enumerate(result['key_paragraphs']):
        print(f"\n关键段落 #{i+1} (相关性: {para['relevance']:.2f}, 情感: {para['sentiment']}):")
        print(para['text'][:300] + "..." if len(para['text']) > 300 else para['text'])