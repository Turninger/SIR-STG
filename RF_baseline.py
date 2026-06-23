"""
Random Forest baseline for stock movement prediction.
Uses the same time-series feature format as LSTM/ALSTM/Transformer baselines.
Multi-seed evaluation with 5 seeds.

Usage:
    python RF_baseline.py
"""
import os
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef

SEEDS = [42, 123, 456, 789, 1024]
FEATURE_COLS = ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']
TARGET_COL = 'label'
LOOK_BACK = 1
TEST_RATIO = 0.2


def load_and_prepare_data():
    """Load CSI100 stock data and create sequences."""
    df = pd.read_csv('./CSIA100_normalized.csv')
    df = df.sort_values('trade_date').reset_index(drop=True)

    X_list, y_list = [], []
    features = df[FEATURE_COLS].values
    labels = df[TARGET_COL].values

    for i in range(LOOK_BACK, len(features)):
        seq = features[i - LOOK_BACK:i].flatten()
        X_list.append(seq)
        y_list.append(labels[i])

    X = np.array(X_list)
    y = np.array(y_list)

    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    val_size = int(len(X_train) * 0.2)
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_rf(X_train, X_test, y_train, y_test, seed):
    """Train and evaluate Random Forest."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred, average='binary')),
        'recall': float(recall_score(y_test, y_pred, average='binary')),
        'mcc': float(matthews_corrcoef(y_test, y_pred)),
    }


def main():
    print("Loading data for Random Forest baseline...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    # Combine train+val for RF (no early stopping needed)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")

    all_metrics = []
    for seed in SEEDS:
        metrics = run_rf(X_train_full, X_test, y_train_full, y_test, seed)
        all_metrics.append(metrics)
        print(f"Seed {seed}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"Recall={metrics['recall']:.4f}, MCC={metrics['mcc']:.4f}")

    keys = ['accuracy', 'f1', 'recall', 'mcc']
    summary = {}
    print("\nRandom Forest Summary:")
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }
        print(f"  {k}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    os.makedirs('./result', exist_ok=True)
    with open('./result/rf_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Results saved to ./result/rf_results.json")


if __name__ == "__main__":
    main()
