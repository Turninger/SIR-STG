"""
Multi-seed evaluation pipeline with statistical significance testing.
Runs SIR-RAN and all graph-based baselines across 5 random seeds,
computes mean ± std, and performs paired t-tests.

Usage:
    python run_multiseed.py --dataset 100 --window 5 --epochs 100
    python run_multiseed.py --dataset 300 --window 3 --epochs 100
"""
import torch
import os
import json
import argparse
import numpy as np
import random
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from scipy import stats
from torch.nn import Dropout, Linear
import sys

# ---- Data Loading ----
from n_day_HeteroGraphSL import build_or_load_dataset

SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data():
    train_graphs, test_graphs = build_or_load_dataset()
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=0.2, random_state=42
    )
    return train_graphs, val_graphs, test_graphs


def compute_metrics(all_labels, all_preds, all_probs=None):
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'mcc': matthews_corrcoef(all_labels, all_preds),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            y = batch['stock'].y
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / total_samples
    return metrics


# =====================================================
# Model Definitions
# =====================================================

class SIR_RAN(torch.nn.Module):
    """SIR-RAN with residual feature reuse (the proposed model)."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        self.residual_proj = Linear(hidden_channels, hidden_channels)
        from torch_geometric.nn import HANConv
        self.conv = HANConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout_rate,
            metadata=metadata
        )
        self.dropout = Dropout(dropout_rate)
        self.residual_fc = Linear(hidden_channels * 2, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'stock': F.relu(self.dropout(self.lin_dict['stock'](x_dict['stock']))),
            'news': F.relu(self.dropout(self.lin_dict['news'](x_dict['news']))),
            'industry': F.relu(self.dropout(self.lin_dict['industry'](x_dict['industry'])))
        }
        original_stock = x_dict['stock']
        x_dict = self.conv(x_dict, edge_index_dict)
        stock_features = x_dict['stock']
        residual = self.residual_proj(original_stock)
        stock_features = stock_features + residual
        stock_features = F.relu(stock_features)
        mid_features = stock_features.clone()
        stock_features = self.dropout(stock_features)
        combined = torch.cat([stock_features, mid_features], dim=1)
        fused = F.relu(self.residual_fc(combined))
        return self.classifier(fused)


class HAN_Model(torch.nn.Module):
    """Standard HAN (baseline)."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.6):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        from torch_geometric.nn import HANConv
        self.conv = HANConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout_rate,
            metadata=metadata
        )
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'stock': F.relu(self.lin_dict['stock'](x_dict['stock'])),
            'news': F.relu(self.lin_dict['news'](x_dict['news'])),
            'industry': F.relu(self.lin_dict['industry'](x_dict['industry']))
        }
        x_dict = self.conv(x_dict, edge_index_dict)
        stock_features = F.relu(self.dropout(x_dict['stock']))
        return self.classifier(stock_features)


class GAT_Model(torch.nn.Module):
    """GAT adapted for heterogeneous graph (applies GATConv to stock-stock edges only)."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin = Linear(8, hidden_channels)
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout_rate)
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = F.relu(self.dropout(self.lin(x_dict['stock'])))
        if ('stock', 'to', 'stock') in edge_index_dict:
            ei = edge_index_dict[('stock', 'to', 'stock')]
        elif ('stock', 'related', 'stock') in edge_index_dict:
            ei = edge_index_dict[('stock', 'related', 'stock')]
        else:
            # find first stock edge
            ei = None
            for k in edge_index_dict:
                if k[0] == 'stock' and k[2] == 'stock':
                    ei = edge_index_dict[k]
                    break
        if ei is not None:
            x = F.relu(self.conv1(x, ei))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, ei))
        return self.classifier(x)


class GraphSAGE_Model(torch.nn.Module):
    """GraphSAGE adapted for heterogeneous graph."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin = Linear(8, hidden_channels)
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = F.relu(self.dropout(self.lin(x_dict['stock'])))
        ei = None
        for k in edge_index_dict:
            if k[0] == 'stock' and k[2] == 'stock':
                ei = edge_index_dict[k]
                break
        if ei is not None:
            x = F.relu(self.conv1(x, ei))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, ei))
        return self.classifier(x)


class RGCN_Model(torch.nn.Module):
    """RGCN adapted for heterogeneous graph."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        from torch_geometric.nn import RGCNConv
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations=4)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=4)
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'stock': F.relu(self.lin_dict['stock'](x_dict['stock'])),
            'news': F.relu(self.lin_dict['news'](x_dict['news'])),
            'industry': F.relu(self.lin_dict['industry'](x_dict['industry']))
        }
        # For RGCN, we use stock nodes and convert edges
        x = x_dict['stock']
        # find an edge set to use
        for k in edge_index_dict:
            ei = edge_index_dict[k]
            if ei.numel() > 0:
                # create dummy edge_type tensor
                et = torch.zeros(ei.size(1), dtype=torch.long, device=ei.device)
                x = F.relu(self.conv1(x, ei, et))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, ei, et))
                break
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        y = batch['stock'].y
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    return total_loss / total_samples


def run_model(model_class, model_name, train_loader, val_loader, test_loader,
              metadata, device, epochs=100, lr=0.01, weight_decay=5e-4,
              hidden_channels=16, **model_kwargs):
    """Train and evaluate a model. Returns dict of test metrics."""
    model = model_class(metadata=metadata, hidden_channels=hidden_channels, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='100', choices=['100', '300'])
    parser.add_argument('--window', type=int, default=5, choices=[3, 5, 7])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['SIR-RAN', 'HAN', 'GAT', 'GraphSAGE', 'RGCN'],
                        help='Models to evaluate')
    parser.add_argument('--output', type=str, default='./result/multiseed_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model registry
    model_registry = {
        'SIR-RAN': (SIR_RAN, {'dropout_rate': 0.3}),
        'HAN': (HAN_Model, {'dropout_rate': 0.6}),
        'GAT': (GAT_Model, {'dropout_rate': 0.3}),
        'GraphSAGE': (GraphSAGE_Model, {'dropout_rate': 0.3}),
        'RGCN': (RGCN_Model, {'dropout_rate': 0.3}),
    }

    # Load data once for all models
    print("Loading dataset...")
    train_graphs, val_graphs, test_graphs = load_and_split_data()
    metadata = train_graphs[0].metadata
    batch_size = 16

    all_results = {}
    metrics_keys = ['accuracy', 'f1', 'recall', 'mcc']

    for model_name in args.models:
        if model_name not in model_registry:
            print(f"Skipping unknown model: {model_name}")
            continue

        model_cls, model_kwargs = model_registry[model_name]
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} across {len(SEEDS)} seeds")
        print(f"{'='*60}")

        seed_results = {k: [] for k in metrics_keys}

        for seed in SEEDS:
            set_seed(seed)
            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

            metrics = run_model(
                model_cls, model_name, train_loader, val_loader, test_loader,
                metadata, device, epochs=args.epochs, **model_kwargs
            )

            for k in metrics_keys:
                seed_results[k].append(metrics[k])

            print(f"  Seed {seed}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, MCC={metrics['mcc']:.4f}")

        # Compute statistics
        summary = {}
        for k in metrics_keys:
            vals = seed_results[k]
            summary[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': [float(v) for v in vals]
            }

        all_results[model_name] = summary

        print(f"\n  {model_name} Summary:")
        for k in metrics_keys:
            print(f"    {k}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    # Statistical significance tests: SIR-RAN vs each baseline
    print(f"\n{'='*60}")
    print("Statistical Significance Tests (Paired t-test)")
    print(f"{'='*60}")

    if 'SIR-RAN' in all_results:
        sir_ran_results = all_results['SIR-RAN']
        sig_results = {}

        for model_name in all_results:
            if model_name == 'SIR-RAN':
                continue
            baseline_results = all_results[model_name]
            sig_results[model_name] = {}

            print(f"\n  SIR-RAN vs {model_name}:")
            for k in metrics_keys:
                t_stat, p_val = stats.ttest_rel(
                    sir_ran_results[k]['values'],
                    baseline_results[k]['values']
                )
                sig_results[model_name][k] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant_05': p_val < 0.05,
                    'significant_01': p_val < 0.01,
                    'significant_001': p_val < 0.001,
                }
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
                print(f"    {k}: t={t_stat:.3f}, p={p_val:.4f} {sig_marker}")

        all_results['_significance_tests'] = sig_results

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
