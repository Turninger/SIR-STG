"""
数据集检查，是否有孤立节点，特征值是否缺失
"""

# dataset_inspector.py
import torch
from torch_geometric.data import HeteroData
import numpy as np
import os
import json
from collections import defaultdict
import sys

def inspect_hetero_dataset(graph_list, dataset_name="dataset"):
    """
    对异构图数据集进行全面检查
    参数:
        graph_list: HeteroData对象的列表
        dataset_name: 数据集名称(用于报告)
    """
    print(f"\n{'='*50}")
    print(f"开始检查数据集: {dataset_name}")
    print(f"包含图数量: {len(graph_list)}")
    print('='*50)
    
    # 初始化统计数据收集器
    stats = {
        'total_graphs': len(graph_list),
        'node_types': set(),
        'edge_types': set(),
        'feature_stats': defaultdict(list),
        'isolated_nodes': defaultdict(list),
        'missing_features': defaultdict(int),
        'empty_edges': defaultdict(int),
        'nan_inf_issues': 0,
        'metadata_issues': 0
    }
    
    # 检查每个图
    for idx, graph in enumerate(graph_list):
        print(f"\n检查图 #{idx+1}/{len(graph_list)}")
        graph_stats = inspect_single_hetero_graph(graph, idx)
        
        # 聚合统计数据
        stats['node_types'] |= set(graph_stats['node_types'])
        stats['edge_types'] |= set(graph_stats['edge_types'])
        
        for node_type, count in graph_stats['missing_features'].items():
            stats['missing_features'][node_type] += count
        
        for edge_type, count in graph_stats['empty_edges'].items():
            stats['empty_edges'][edge_type] += count
        
        stats['isolated_nodes'] = merge_isolated_stats(
            stats['isolated_nodes'], graph_stats['isolated_nodes'])
        
        stats['nan_inf_issues'] += graph_stats['nan_inf_issues']
        stats['metadata_issues'] += graph_stats['metadata_issues']
        
        # 收集特征维度信息
        for node_type, feat in graph.x_dict.items():
            if feat is not None:
                stats['feature_stats'][node_type].append(feat.shape[1])
    
    # 生成最终报告
    generate_final_report(stats, dataset_name, graph_list)
    
    return stats

def inspect_single_hetero_graph(graph: HeteroData, graph_idx: int):
    """检查单个异构图"""
    stats = {
        'graph_id': graph_idx,
        'node_types': [],
        'edge_types': [],
        'missing_features': {},
        'empty_edges': {},
        'isolated_nodes': defaultdict(list),
        'nan_inf_issues': 0,
        'metadata_issues': 0
    }
    
    # 1. 检查元数据一致性
    if not hasattr(graph, 'metadata') or graph.metadata() is None:
        print("  ⚠️ 警告: 缺少元数据")
        stats['metadata_issues'] += 1
    else:
        metadata=graph.metadata()
        # 验证元数据与实际节点/边类型匹配
        metadata_node_types = set(metadata[0])
        metadata_edge_types = set(metadata[1])
        
        actual_node_types = set(graph.x_dict.keys())
        actual_edge_types = set(graph.edge_index_dict.keys())
        
        if metadata_node_types != actual_node_types:
            print(f"  ⚠️ 元数据节点类型不匹配: "
                  f"元数据={metadata_node_types}, 实际={actual_node_types}")
            stats['metadata_issues'] += 1
        
        if metadata_edge_types != actual_edge_types:
            print(f"  ⚠️ 元数据边类型不匹配: "
                  f"元数据={metadata_edge_types}, 实际={actual_edge_types}")
            stats['metadata_issues'] += 1
    
    # 2. 检查节点特征
    print("\n[节点特征检查]")
    for node_type, features in graph.x_dict.items():
        stats['node_types'].append(node_type)
        
        if features is None:
            print(f"  ❌ 严重: 节点类型 '{node_type}' 特征为None")
            stats['missing_features'][node_type] = graph[node_type].num_nodes
            continue
        
        # 检查特征张量
        if not isinstance(features, torch.Tensor):
            print(f"  ❌ 严重: 节点类型 '{node_type}' 特征不是张量")
            stats['missing_features'][node_type] = graph[node_type].num_nodes
            continue
        
        # 检查空特征
        if features.nelement() == 0:
            print(f"  ❌ 严重: 节点类型 '{node_type}' 特征张量为空")
            stats['missing_features'][node_type] = features.size(0)
            continue
        
        # 检查NaN和Inf值
        nan_count = torch.isnan(features).sum().item()
        inf_count = torch.isinf(features).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️ 警告: 节点类型 '{node_type}' 包含 "
                  f"{nan_count}个NaN和{inf_count}个Inf值")
            stats['nan_inf_issues'] += 1
    
    # 3. 检查边连接
    print("\n[边连接检查]")
    edge_connections = defaultdict(set)
    
    for edge_type, edge_index in graph.edge_index_dict.items():
        stats['edge_types'].append(edge_type)
        src_type, _, dst_type = edge_type
        
        # 检查边索引是否存在
        if edge_index is None:
            print(f"  ❌ 严重: 边类型 '{edge_type}' 索引为None")
            stats['empty_edges'][edge_type] = 0
            continue
        
        # 检查边索引是否为空
        if edge_index.size(1) == 0:
            print(f"  ⚠️ 警告: 边类型 '{edge_type}' 没有连接")
            stats['empty_edges'][edge_type] = edge_index.size(1)
            continue
        
        # 记录连接信息
        src_nodes = edge_index[0].unique().tolist()
        dst_nodes = edge_index[1].unique().tolist()
        
        edge_connections[src_type] |= set(src_nodes)
        edge_connections[dst_type] |= set(dst_nodes)
    
    # 4. 检查孤立节点
    print("\n[孤立节点检查]")
    for node_type in graph.x_dict.keys():
        # 获取该类型的所有节点
        if node_type not in graph or not hasattr(graph[node_type], 'num_nodes'):
            continue
            
        all_nodes = set(range(graph[node_type].num_nodes))
        
        # 获取有连接的节点
        connected_nodes = edge_connections.get(node_type, set())
        
        # 找出孤立节点
        isolated_nodes = all_nodes - connected_nodes
        
        if isolated_nodes:
            print(f"  ⚠️ 警告: 节点类型 '{node_type}' 有 {len(isolated_nodes)} 个孤立节点")
            stats['isolated_nodes'][node_type].extend(isolated_nodes)
    
    return stats

def merge_isolated_stats(main_stats, new_stats):
    """合并孤立节点统计数据"""
    for node_type, nodes in new_stats.items():
        main_stats[node_type].extend(nodes)
    return main_stats

def generate_final_report(stats, dataset_name, graph_list):
    """生成最终汇总报告"""
    print('\n' + '='*50)
    print(f"数据集检查完成: {dataset_name}")
    print('='*50)
    
    # 基本统计
    print(f"\n📊 基本统计:")
    print(f"- 图总数: {stats['total_graphs']}")
    print(f"- 节点类型: {', '.join(stats['node_types'])}")
    print(f"- 边类型: {', '.join([str(et) for et in stats['edge_types']])}")
    
    # 特征问题
    print("\n🔍 特征问题:")
    if stats['missing_features']:
        for node_type, count in stats['missing_features'].items():
            print(f"- 节点类型 '{node_type}': {count} 个节点缺少特征")
    else:
        print("- 所有节点都有有效特征")
    
    # 特征维度统计
    print("\n📏 特征维度:")
    for node_type, dims in stats['feature_stats'].items():
        unique_dims = set(dims)
        if len(unique_dims) == 1:
            print(f"- '{node_type}': 所有图维度为 {dims[0]}")
        else:
            print(f"- ⚠️ '{node_type}': 维度不一致 {unique_dims}")
    
    # 边连接问题
    print("\n🔗 边连接问题:")
    if stats['empty_edges']:
        for edge_type, count in stats['empty_edges'].items():
            print(f"- 边类型 '{edge_type}' 在 {count} 个图中为空")
    else:
        print("- 所有边类型都有连接")
    
    # 孤立节点问题
    print("\n👤 孤立节点统计:")
    if stats['isolated_nodes']:
        for node_type, nodes in stats['isolated_nodes'].items():
            total_isolated = len(nodes)
            
            # 正确计算该节点类型在所有图中的总节点数
            total_nodes = 0
            for graph in graph_list:
                if hasattr(graph, node_type) and hasattr(graph[node_type], 'num_nodes'):
                    total_nodes += graph[node_type].num_nodes
            
            percentage = (total_isolated / total_nodes) * 100 if total_nodes > 0 else 0
            print(f"- 节点类型 '{node_type}': {total_isolated} 个孤立节点 ({percentage:.2f}%)")
    else:
        print("- 未检测到孤立节点")
    
    # 数据质量问题
    print("\n⚠️ 数据质量问题:")
    print(f"- 包含NaN/Inf的特征图数量: {stats['nan_inf_issues']}")
    print(f"- 元数据不一致的图数量: {stats['metadata_issues']}")
    
    # 保存详细报告
    save_detailed_report(stats, dataset_name, graph_list)
    
    # 总体评估
    print("\n✅ 总体评估:")
    if (stats['missing_features'] or stats['empty_edges'] or 
        stats['isolated_nodes'] or stats['nan_inf_issues'] or stats['metadata_issues']):
        print("- 数据集存在问题，需要进一步处理")
    else:
        print("- 数据集质量良好")

def save_detailed_report(stats, dataset_name, graph_list):
    """保存详细报告到JSON文件"""
    # 计算每个节点类型的总节点数
    total_node_counts = defaultdict(int)
    for node_type in stats['node_types']:
        for graph in graph_list:
            if hasattr(graph, node_type) and hasattr(graph[node_type], 'num_nodes'):
                total_node_counts[node_type] += graph[node_type].num_nodes
    
    report = {
        'dataset': dataset_name,
        'total_graphs': stats['total_graphs'],
        'node_types': list(stats['node_types']),
        'edge_types': [str(et) for et in stats['edge_types']],
        'total_node_counts': dict(total_node_counts),
        'feature_dimensions': {
            nt: {
                'min': min(dims) if dims else 0,
                'max': max(dims) if dims else 0,
                'consistent': len(set(dims)) == 1 if dims else True
            }
            for nt, dims in stats['feature_stats'].items()
        },
        'missing_features': stats['missing_features'],
        'empty_edges': stats['empty_edges'],
        'isolated_nodes': {
            nt: len(nodes) for nt, nodes in stats['isolated_nodes'].items()
        },
        'data_quality_issues': {
            'nan_inf_issues': stats['nan_inf_issues'],
            'metadata_issues': stats['metadata_issues']
        },
        'recommendations': generate_recommendations(stats)
    }
    
    # 创建输出目录
    os.makedirs("dataset_reports", exist_ok=True)
    filename = f"dataset_reports/{dataset_name}_inspection_report.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n详细报告已保存至: {filename}")

def generate_recommendations(stats):
    """根据检查结果生成处理建议"""
    recommendations = []
    
    # 缺失特征建议
    if stats['missing_features']:
        for node_type in stats['missing_features']:
            rec = {
                'issue': f"节点类型 '{node_type}' 缺少特征",
                'recommendation': (
                    f"1. 检查数据源确保特征存在\n"
                    f"2. 考虑使用零填充或均值填充\n"
                    f"3. 使用类型特定的特征初始化器"
                )
            }
            recommendations.append(rec)
    
    # 维度不一致建议
    for node_type, dims in stats['feature_stats'].items():
        if len(set(dims)) > 1:
            rec = {
                'issue': f"节点类型 '{node_type}' 特征维度不一致",
                'recommendation': (
                    f"1. 对所有图使用统一的特征处理管道\n"
                    f"2. 添加特征投影层统一维度\n"
                    f"3. 检查特征提取过程的一致性"
                )
            }
            recommendations.append(rec)
    
    # 孤立节点建议
    if stats['isolated_nodes']:
        rec = {
            'issue': "存在孤立节点",
            'recommendation': (
                "1. 添加自环边: graph.add_self_loop()\n"
                "2. 在HANConv中跳过孤立节点处理\n"
                "3. 使用虚拟边连接孤立节点"
            )
        }
        recommendations.append(rec)
    
    # 空边建议
    if stats['empty_edges']:
        for edge_type in stats['empty_edges']:
            rec = {
                'issue': f"边类型 '{edge_type}' 为空",
                'recommendation': (
                    f"1. 检查该关系类型的数据源\n"
                    f"2. 考虑移除该边类型\n"
                    f"3. 使用虚拟边填充"
                )
            }
            recommendations.append(rec)
    
    # 数据质量问题建议
    if stats['nan_inf_issues']:
        rec = {
            'issue': "特征包含NaN/Inf值",
            'recommendation': (
                "1. 使用 torch.nan_to_num() 处理\n"
                "2. 添加数据清洗步骤\n"
                "3. 检查特征计算过程"
            )
        }
        recommendations.append(rec)
    
    if stats['metadata_issues']:
        rec = {
            'issue': "元数据不一致",
            'recommendation': (
                "1. 确保元数据与实际数据匹配\n"
                "2. 动态生成元数据: metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))"
            )
        }
        recommendations.append(rec)
    
    return recommendations

def compare_datasets(train_stats, test_stats, train_graphs, test_graphs):
    """比较训练集和测试集的差异"""
    print("\n" + "="*50)
    print("训练集-测试集比较")
    print("="*50)
    
    # 比较节点类型
    train_nodes = set(train_stats['node_types'])
    test_nodes = set(test_stats['node_types'])
    
    if train_nodes != test_nodes:
        print("⚠️ 节点类型不一致:")
        print(f"- 训练集独有: {train_nodes - test_nodes}")
        print(f"- 测试集独有: {test_nodes - train_nodes}")
    else:
        print("✅ 节点类型一致")
    
    # 比较边类型
    train_edges = set(str(et) for et in train_stats['edge_types'])
    test_edges = set(str(et) for et in test_stats['edge_types'])
    
    if train_edges != test_edges:
        print("⚠️ 边类型不一致:")
        print(f"- 训练集独有: {train_edges - test_edges}")
        print(f"- 测试集独有: {test_edges - train_edges}")
    else:
        print("✅ 边类型一致")
    
    # 比较特征维度
    print("\n特征维度比较:")
    all_node_types = train_nodes | test_nodes
    for nt in all_node_types:
        train_dims = set(train_stats['feature_stats'].get(nt, []))
        test_dims = set(test_stats['feature_stats'].get(nt, []))
        
        if train_dims and test_dims:
            if train_dims != test_dims:
                print(f"⚠️ '{nt}' 特征维度不一致 - 训练: {train_dims}, 测试: {test_dims}")
            else:
                print(f"✅ '{nt}' 特征维度一致: {train_dims}")
        elif nt in train_stats['feature_stats'] and nt not in test_stats['feature_stats']:
            print(f"⚠️ '{nt}' 仅在训练集中存在特征")
        elif nt not in train_stats['feature_stats'] and nt in test_stats['feature_stats']:
            print(f"⚠️ '{nt}' 仅在测试集中存在特征")
    
    # 比较数据分布问题
    print("\n数据分布问题比较:")
    for issue_type in ['missing_features', 'empty_edges']:
        train_issues = train_stats[issue_type]
        test_issues = test_stats[issue_type]
        
        if not train_issues and not test_issues:
            print(f"✅ {issue_type}: 无问题")
            continue
        
        print(f"⚠️ {issue_type}问题分布:")
        all_keys = set(train_issues.keys()) | set(test_issues.keys())
        
        for key in all_keys:
            train_count = train_issues.get(key, 0)
            test_count = test_issues.get(key, 0)
            
            if train_count or test_count:
                print(f"- {key}: 训练集 {train_count}, 测试集 {test_count}")
    
    # 比较孤立节点比例
    print("\n孤立节点比例比较:")
    for node_type in (set(train_stats['isolated_nodes'].keys()) | 
                      set(test_stats['isolated_nodes'].keys())):
        train_isolated = len(train_stats['isolated_nodes'].get(node_type, []))
        test_isolated = len(test_stats['isolated_nodes'].get(node_type, []))
        
        # 计算训练集总节点数
        train_total = 0
        for graph in train_graphs:
            if hasattr(graph, node_type) and hasattr(graph[node_type], 'num_nodes'):
                train_total += graph[node_type].num_nodes
        
        # 计算测试集总节点数
        test_total = 0
        for graph in test_graphs:
            if hasattr(graph, node_type) and hasattr(graph[node_type], 'num_nodes'):
                test_total += graph[node_type].num_nodes
        
        train_percent = (train_isolated / train_total * 100) if train_total > 0 else 0
        test_percent = (test_isolated / test_total * 100) if test_total > 0 else 0
        
        print(f"- '{node_type}': 训练集 {train_percent:.2f}%, 测试集 {test_percent:.2f}%")

if __name__ == "__main__":
    try:
        # 尝试从您的数据加载模块导入
        from n_day_HeteroGraphSL import build_or_load_dataset
    except ImportError:
        print("❌ 错误: 无法导入 build_or_load_dataset 函数")
        print("请确保 n_day_HeteroGraphSL.py 在相同目录下")
        sys.exit(1)
    
    try:
        # 加载数据集
        print("加载数据集...")
        train_graphs, test_graphs = build_or_load_dataset()
        print(f"成功加载: 训练集 {len(train_graphs)} 个图, 测试集 {len(test_graphs)} 个图")
    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        sys.exit(1)
    
    # 检查训练集
    print("\n\n🔍 检查训练集...")
    train_stats = inspect_hetero_dataset(train_graphs, "训练集")
    
    # 检查测试集
    print("\n\n🔍 检查测试集...")
    test_stats = inspect_hetero_dataset(test_graphs, "测试集")
    
    # 比较训练集和测试集
    print("\n\n🔍 比较训练集和测试集...")
    compare_datasets(train_stats, test_stats, train_graphs, test_graphs)