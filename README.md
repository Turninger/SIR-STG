# SIR-STG: Stock-Industry-Relation based Stock Trend Prediction via Heterogeneous Graph

## Overview

SIR-STG is a research project that constructs a **Stock-Industry-News Heterogeneous Graph** and applies heterogeneous graph neural networks (HANConv, HGT, SAGEConv, GATConv, RGCNConv) to predict stock price movement. The project accompanies the paper *"Enhancing Stock Movement Prediction Accuracy via News-Market Heterogeneous Graph Construction"* (targeted at Expert Systems with Applications).

The core idea is to model three types of entities (Stocks, News, Industries) and their heterogeneous relationships in a unified graph:
- **Stock–Stock edges** (linkage): stocks co-mentioned in the same news article, weighted by Pearson correlation / DTW-based linkage value
- **News–Stock edges** (sentiment): news sentiment towards a specific stock, weighted by FinBERT sentiment confidence
- **Industry–Stock edges** (market ratio): industry classification membership, weighted by market capitalization ratio

## Datasets
Raw dataset and Processed datasets are:
https://drive.google.com/drive/folders/1m8uaBiFQUcjW2PsOWgKR1eleK3AmVBCC?usp=drive_link

## Architecture

```
data_raw/2022/           →  utils/edge_construct.py      →  data_processed/
(raw daily news CSVs)       utils/NS_edge_construct.py       (processed edges & nodes)
                            utils/IS_edge_construct.py
                            utils/SA.py (sentiment)
                            utils/marketval_ratio.py
                                    ↓
                            n_day_HeteroGraphSL.py
                            (N-day rolling window graph builder)
                                    ↓
                        *.pt HeteroData datasets
                        (3/5/7 day, 100/300 stocks)
                                    ↓
                  ┌─────────────────┼─────────────────┐
                  ↓                  ↓                  ↓
            SIR-RAN (proposed)   HAN Baseline      Other Baselines
            - Residual Reuse     - 3-layer HANConv  - GAT
            - HANConv            - Multi-head attn   - GraphSAGE
            - Feature Fusion                         - RGCN
                  ↓                                  - Random Forest
            run_multiseed.py                         - ECHO-GL (HeFF)
            run_ablation.py
            HARGN.py
```

## Key Components

| File | Description |
|------|-------------|
| `n_day_HeteroGraphSL.py` | Build or load N-day window heterogeneous graph datasets from processed data |
| `run_multiseed.py` | Multi-seed (5 seeds) evaluation of SIR-RAN vs baselines with paired t-tests |
| `run_ablation.py` | Ablation study: w/o ResidualReuse, w/o MetaPath, w/o MultiHead |
| `HARGN.py` | 3-layer HAN (Heterogeneous Attention Network) model training |
| `run_ECHO_GL.py` | ECHO-GL experiment pipeline with HGT-based HeFF model |
| `RF_baseline.py` | Random Forest baseline on time-series OHLCV features |
| `checkDataset.py` | Dataset quality inspection (isolated nodes, missing features, NaN/Inf) |
| `container.py` | Data filtering, preprocessing, and trading simulation utilities |
| `heterograph.py` | Single-day heterogeneous graph construction example |
| `EIG.ipynb` | Exploratory analysis notebook |
| `data_process_csi.ipynb` | CSI index stock data preprocessing |

## Data Processing

Utilities in `utils/`:

| Script | Purpose |
|--------|---------|
| `edge_construct.py` | Build stock-stock co-mention edges from raw news |
| `NS_edge_construct.py` | Build news-stock sentiment edges |
| `NS_edge_constructLLM.py` | LLM-enhanced news-stock edge construction |
| `NI_edge_construct.py` | Build news-industry edges with keyword matching |
| `IS_edge_construct.py` | Build industry-stock edges with market ratio weights |
| `SA.py` / `sentiment_analysis.py` | FinBERT-based Chinese sentiment analysis |
| `marketval_ratio.py` | Market capitalization ratio calculation |
| `Pearson_DTW.py` | Pearson correlation and DTW-based stock linkage |
| `filter_CSI_news.py` | Filter news related to CSI index constituent stocks |
| `filter_CSI_SSnodes.py` | Filter stock-stock edges for CSI constituents |
| `nodes.py` / `n_days_nodes.py` | Node feature construction with labels |
| `n_dayDataset.py` | N-day rolling window dataset builder |
| `NI.py` / `NI_generate_dic.py` | News-Industry keyword dictionary and matching |
| `data_raw_processed.py` | Raw data preprocessing entry point |
| `normalizedCSI.py` | CSI stock data normalization |
| `industry.py` / `industry_code.py` | Industry classification utilities |
| `Graph_construct.py` | Single-day heterogeneous graph assembly |
| `result_analyse.py` | Result analysis and visualization |

## Datasets

Pre-built graph datasets (PyTorch Geometric `HeteroData` lists) are stored as `.pt` files:

| File | Window | Stocks | Variant |
|------|--------|--------|---------|
| `3_day_100_hetero_graph_dataset.pt` | 3 days | 100 | Full |
| `3_day_300_hetero_graph_dataset.pt` | 3 days | 300 | Full |
| `3_day_300_woIS_hetero_graph_dataset.pt` | 3 days | 300 | w/o I-S edges |
| `3_day_300_woNS_hetero_graph_dataset.pt` | 3 days | 300 | w/o N-S edges |
| `5_day_100_hetero_graph_dataset.pt` | 5 days | 100 | Full |
| `5_day_300_hetero_graph_dataset.pt` | 5 days | 300 | Full |
| `5_day_300_woIS_hetero_graph_dataset.pt` | 5 days | 300 | w/o I-S edges |
| `5_day_300_woNS_hetero_graph_dataset.pt` | 5 days | 300 | w/o N-S edges |
| `7_day_100_hetero_graph_dataset.pt` | 7 days | 100 | Full |
| `7_day_300_hetero_graph_dataset.pt` | 7 days | 300 | Full |
| `7_day_300_woIS_hetero_graph_dataset.pt` | 7 days | 300 | w/o I-S edges |
| `7_day_300_woNS_hetero_graph_dataset.pt` | 7 days | 300 | w/o N-S edges |

Each graph contains:
- **stock** nodes (8-dim OHLCV features + binary movement label)
- **news** nodes (1-dim placeholder features)
- **industry** nodes (1-dim placeholder features)
- Edges: `(stock, link, stock)`, `(news, sentiment, stock)`, `(industry, marketratio, stock)`

## Models

### SIR-RAN (Proposed)
Heterogeneous graph neural network with:
- **Residual Feature Reuse**: original stock features are projected and added back after convolution
- **HANConv** with multi-head attention for meta-path-based neighborhood aggregation
- **Feature Fusion**: concatenation of mid-layer and post-dropout features followed by linear projection

### Baselines
- **HAN**: 3-layer HANConv with multi-head attention
- **GAT**: GATConv on stock-stock edges only
- **GraphSAGE**: SAGEConv on stock-stock edges only
- **RGCN**: RGCNConv adapted for heterogeneous graph
- **Random Forest**: traditional ML baseline on OHLCV time-series features
- **ECHO-GL (HeFF)**: HGT-based model with entity/topic embeddings and LSTM price encoding

## Usage

### 1. Build Datasets

```bash
# Generate N-day window heterogeneous graph datasets
python n_day_HeteroGraphSL.py
```

### 2. Multi-Seed Evaluation

```bash
# Run SIR-RAN + all baselines on 5 seeds
python run_multiseed.py --dataset 100 --window 5 --epochs 100

# With specific models
python run_multiseed.py --dataset 300 --window 3 --models SIR-RAN HAN GAT
```

### 3. Ablation Study

```bash
python run_ablation.py --dataset 100 --window 5 --epochs 100
```

### 4. HAN Training

```bash
python HARGN.py
```

### 5. Random Forest Baseline

```bash
python RF_baseline.py
```

### 6. Check Dataset Quality

```bash
python checkDataset.py
```

## Key Metrics

All models are evaluated using:
- **Accuracy**
- **F1 Score** (binary, macro)
- **Recall**
- **MCC** (Matthews Correlation Coefficient)

Multi-seed results include mean ± std and paired t-tests for statistical significance.

## Requirements

- Python 3.8+
- PyTorch >= 1.12
- PyTorch Geometric >= 2.3
- scikit-learn
- pandas, numpy
- scipy (for statistical tests)
- matplotlib (for visualization)
- FinBERT (for Chinese sentiment analysis in preprocessing)

## Directory Structure

```
SIR-STG/
├── data_raw/2022/          # Raw daily news CSVs (ignored by git)
├── data_processed/         # Processed edges and nodes (ignored by git)
├── utils/                  # Data processing and edge construction utilities
│   ├── finbert_tone_chinese/  # FinBERT model for Chinese sentiment (ignored)
│   └── ...
├── result/                 # Experiment results (ignored by git)
├── dataset_reports/        # Dataset inspection reports
├── edge/                   # Edge construction scripts
├── Enhancing_Stock_.../    # LaTeX manuscript for the paper
├── n_day_HeteroGraphSL.py  # Graph dataset builder
├── run_multiseed.py        # Multi-seed evaluation pipeline
├── run_ablation.py         # Ablation study
├── HARGN.py                # HAN model training
├── run_ECHO_GL.py          # ECHO-GL experiment
├── RF_baseline.py          # Random Forest baseline
├── checkDataset.py         # Dataset quality checker
├── container.py            # Data containers and filters
├── heterograph.py          # Single-day graph example
├── CSIA100.csv             # CSI A100 constituent list
├── CSIA100_normalized.csv  # Normalized CSI A100 OHLCV data
└── requirements.txt        # Dependencies
```

## Citation

If you use this code in your research, please cite the corresponding paper:

> Xie, W., et al. "Enhancing Stock Movement Prediction Accuracy via News-Market Heterogeneous Graph Construction." (Under review)
