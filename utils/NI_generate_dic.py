import os
import json
import pandas as pd
from volcenginesdkarkruntime import Ark
import time

class IndustryKeywordGenerator:
    def __init__(self):
        # 初始化豆包客户端
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="bab6b861-266d-43c7-970d-04a0692e93f2"
        )
        
        # 行业关键词词典
        self.industry_keywords = {}
        
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
                # 尝试多种分隔符
                keywords = []
                for separator in [',', '，', '、']:
                    keywords = [kw.strip() for kw in response.split(separator) if kw.strip()]
                    if len(keywords) > 3:  # 如果分割出的关键词数量合理
                        break
                
                if keywords:
                    print(f"为行业【{industry}】生成 {len(keywords)} 个关键词")
                    return keywords
                else:
                    # 如果没有成功分割，尝试按空格分割
                    keywords = [kw.strip() for kw in response.split() if kw.strip()]
                    if keywords:
                        print(f"为行业【{industry}】生成 {len(keywords)} 个关键词")
                        return keywords
                    
            except Exception as e:
                print(f"第{attempt+1}次尝试失败: {e}")
                time.sleep(2)  # 失败后等待2秒再重试
                continue
                
        # 如果API调用失败，返回基础关键词
        base_keywords = self.get_base_keywords(industry)
        print(f"使用基础关键词 for {industry}: {len(base_keywords)} 个")
        return base_keywords
    
    def get_base_keywords(self, industry):
        """获取行业基础关键词（备用）"""
        base_keyword_map = {
            '传媒': ['传媒', '媒体', '广告', '影视', '出版', '新媒体', '短视频', '内容', 'IP', '传播', '宣传'],
            '金融': ['金融', '银行', '证券', '保险', '基金', '信托', '投资', '信贷', '理财', '股市', '债券'],
            '科技': ['科技', '技术', '创新', '研发', '软件', '硬件', '互联网', '数字化', '人工智能', '大数据'],
            '消费': ['消费', '零售', '电商', '品牌', '商品', '服务', '用户', '市场', '销售', '购买'],
            '医药': ['医药', '医疗', '药品', '医院', '健康', '生物', '制药', '器械', '治疗', '疾病'],
            '房地产': ['房地产', '地产', '房产', '楼市', '房价', '住宅', '商业地产', '开发商', '物业管理'],
            '能源': ['能源', '石油', '煤炭', '电力', '新能源', '光伏', '风电', '储能', '电网'],
            '汽车': ['汽车', '车企', '新能源汽车', '电动车', '自动驾驶', '零部件', '4S店', '销量'],
            '教育': ['教育', '培训', '学校', '在线教育', '课程', '教学', '学习', '考试'],
            '旅游': ['旅游', '旅行', '景区', '酒店', '航空', '旅行社', '游客', '度假']
        }
        return base_keyword_map.get(industry, [industry])
    
    def build_keywords_from_csv(self, csv_path, output_json_path="industry_keywords.json"):
        """从CSV文件构建行业关键词词典"""
        print(f"从CSV文件读取行业数据: {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        if '行业' not in df.columns:
            raise ValueError("CSV文件必须包含'行业'列")
        
        # 获取所有行业
        industries = df['行业'].dropna().unique()
        print(f"发现 {len(industries)} 个不同行业: {list(industries)}")
        
        # 为每个行业生成关键词
        for industry in industries:
            if industry not in self.industry_keywords:
                self.industry_keywords[industry] = self.generate_industry_keywords(industry)
                time.sleep(1)  # 避免API调用过于频繁
        
        # 保存到JSON文件
        self.save_keywords_to_json(output_json_path)
        
        return self.industry_keywords
    
    def build_keywords_from_list(self, industry_list, output_json_path="industry_keywords.json"):
        """从行业列表构建关键词词典"""
        print(f"为行业列表生成关键词: {industry_list}")
        
        for industry in industry_list:
            if industry not in self.industry_keywords:
                self.industry_keywords[industry] = self.generate_industry_keywords(industry)
                time.sleep(1)  # 避免API调用过于频繁
        
        # 保存到JSON文件
        self.save_keywords_to_json(output_json_path)
        
        return self.industry_keywords
    
    def save_keywords_to_json(self, output_path):
        """保存关键词词典到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.industry_keywords, f, ensure_ascii=False, indent=2)
        print(f"关键词词典已保存至: {output_path}")
    
    def load_keywords_from_json(self, json_path):
        """从JSON文件加载关键词词典"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.industry_keywords = json.load(f)
            print(f"已从 {json_path} 加载关键词词典，包含 {len(self.industry_keywords)} 个行业")
            return self.industry_keywords
        except FileNotFoundError:
            print(f"关键词文件 {json_path} 不存在")
            return {}
        except Exception as e:
            print(f"加载关键词文件失败: {e}")
            return {}

# 使用示例
if __name__ == "__main__":
    generator = IndustryKeywordGenerator()
    
    # 方法1: 从CSV文件生成关键词词典
    # csv_file = "your_data.csv"
    # keywords = generator.build_keywords_from_csv(csv_file, "industry_keywords.json")
    
    # 方法2: 从行业列表生成关键词词典
    industry_list = ['电力设备', '建筑材料', '商贸零售', '基础化工', '交通运输', '有色金属', '机械设备', '美容护理',
       '电子', '计算机', '国防军工', '纺织服饰', '医药生物', '非银金融', '汽车', '轻工制造', '公用事业',
       '石油石化', '房地产', '传媒', '家用电器', '食品饮料', '通信', '农林牧渔', '钢铁', '建筑装饰',
       '综合', '环保', '煤炭', '银行', '社会服务', '未知行业']
    keywords = generator.build_keywords_from_list(industry_list, "industry_keywords.json")
    
    print("关键词生成完成！")