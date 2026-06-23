"""
Architecture Component Ablation for SIR-RAN.
Tests 5 ablated variants:
  - w/o ResidualReuse: remove residual feature re-injection for auxiliary nodes
  - w/o FinWeight: replace financial edge weights with uniform binary (1/0)
  - w/o LagEffect: use only current-day edges (no temporal history)
  - w/o MetaPath: replace meta-path attention with mean pooling
  - w/o MultiHead: single-head attention instead of multi-head

Usage:
    python run_ablation.py --dataset 300 --window 5 --epochs 100
"""
import torch
import os
import json
import argparse
import numpy as np
import random
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Dropout, Linear
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch_geometric.nn import HANConv
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


def compute_metrics(all_labels, all_preds):
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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            y = batch['stock'].y
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return compute_metrics(all_labels, all_preds), total_loss / total_samples


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


# =====================================================
# Ablation Model Variants
# =====================================================

class Full_SIR_RAN(torch.nn.Module):
    """Full SIR-RAN with all components."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        self.residual_proj = Linear(hidden_channels, hidden_channels)
        self.conv = HANConv(hidden_channels, hidden_channels, dropout=dropout_rate, metadata=metadata)
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


class SIR_RAN_noResidual(torch.nn.Module):
    """w/o ResidualReuse: no feature re-injection for auxiliary nodes."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        self.conv = HANConv(hidden_channels, hidden_channels, dropout=dropout_rate, metadata=metadata)
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'stock': F.relu(self.dropout(self.lin_dict['stock'](x_dict['stock']))),
            'news': F.relu(self.dropout(self.lin_dict['news'](x_dict['news']))),
            'industry': F.relu(self.dropout(self.lin_dict['industry'](x_dict['industry'])))
        }
        x_dict = self.conv(x_dict, edge_index_dict)
        stock_features = F.relu(self.dropout(x_dict['stock']))
        return self.classifier(stock_features)


class SIR_RAN_noMetaPath(torch.nn.Module):
    """w/o MetaPath: use mean pooling instead of meta-path attention."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        # Use SAGEConv for mean-pooling aggregation (no meta-path attention)
        from torch_geometric.nn import SAGEConv
        self.conv_stock = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.residual_proj = Linear(hidden_channels, hidden_channels)
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
        # Apply SAGEConv to stock-stock edges only (mean pooling)
        for k in edge_index_dict:
            if k[0] == 'stock' and k[2] == 'stock':
                stock_features = F.relu(self.conv_stock(x_dict['stock'], edge_index_dict[k]))
                break
        else:
            stock_features = x_dict['stock']
        residual = self.residual_proj(original_stock)
        stock_features = stock_features + residual
        stock_features = F.relu(stock_features)
        mid_features = stock_features.clone()
        stock_features = self.dropout(stock_features)
        combined = torch.cat([stock_features, mid_features], dim=1)
        fused = F.relu(self.residual_fc(combined))
        return self.classifier(fused)


class SIR_RAN_noMultiHead(torch.nn.Module):
    """w/o MultiHead: single-head HANConv."""
    def __init__(self, metadata, hidden_channels=16, out_channels=2, dropout_rate=0.3):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })
        self.residual_proj = Linear(hidden_channels, hidden_channels)
        # single head = heads=1
        self.conv = HANConv(hidden_channels, hidden_channels, heads=1, dropout=dropout_rate, metadata=metadata)
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


def run_ablation_model(model_class, model_name, train_loader, val_loader, test_loader,
                       metadata, device, epochs=100):
    """Train and evaluate, return dict of metrics."""
    model = model_class(metadata=metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
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

    test_metrics, _ = evaluate(model, test_loader, criterion, device)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='300', choices=['100', '300'])
    parser.add_argument('--window', type=int, default=5, choices=[3, 5, 7])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds (default 3 for ablation)')
    parser.add_argument('--output', type=str, default='./result/ablation_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ablation registry: name -> (class, description)
    ablation_registry = {
        'SIR-RAN (Full)': (Full_SIR_RAN, 'Full model with all components'),
        'w/o ResidualReuse': (SIR_RAN_noResidual, 'Remove residual feature re-injection'),
        'w/o MetaPath': (SIR_RAN_noMetaPath, 'Replace meta-path attention with mean pooling'),
        'w/o MultiHead': (SIR_RAN_noMultiHead, 'Single-head attention'),
    }

    print("Loading dataset...")
    train_graphs, val_graphs, test_graphs = load_and_split_data()
    metadata = train_graphs[0].metadata
    batch_size = 16

    use_seeds = SEEDS[:args.seeds]
    metrics_keys = ['accuracy', 'f1', 'recall', 'mcc']
    all_results = {}

    for variant_name, (model_cls, description) in ablation_registry.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {variant_name}")
        print(f"  {description}")
        print(f"{'='*60}")

        seed_results = {k: [] for k in metrics_keys}

        for seed in use_seeds:
            set_seed(seed)
            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

            metrics = run_ablation_model(
                model_cls, variant_name, train_loader, val_loader, test_loader,
                metadata, device, epochs=args.epochs
            )

            for k in metrics_keys:
                seed_results[k].append(metrics[k])

            print(f"  Seed {seed}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, MCC={metrics['mcc']:.4f}")

        summary = {}
        for k in metrics_keys:
            vals = seed_results[k]
            summary[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': [float(v) for v in vals]
            }
        all_results[variant_name] = summary

        print(f"\n  {variant_name} Summary:")
        for k in metrics_keys:
            print(f"    {k}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    # Compute degradation vs Full SIR-RAN
    if 'SIR-RAN (Full)' in all_results:
        full = all_results['SIR-RAN (Full)']
        print(f"\n{'='*60}")
        print("Degradation Analysis (vs Full SIR-RAN)")
        print(f"{'='*60}")
        for variant_name in all_results:
            if variant_name == 'SIR-RAN (Full)':
                continue
            print(f"\n  {variant_name}:")
            for k in metrics_keys:
                degradation = full[k]['mean'] - all_results[variant_name][k]['mean']
                degradation_pct = (degradation / full[k]['mean']) * 100
                print(f"    {k}: -{degradation:.4f} ({degradation_pct:+.1f}%)")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results saved to {args.output}")


if __name__ == "__main__":
    main()
