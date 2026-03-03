# TDS-GCN: Time-aware Dynamic Sparse Graph Convolutional Network for Recommendation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于时间感知动态图神经网络的推荐系统，通过构建时间感知的动态图结构，有效捕捉用户兴趣的时间演化规律，提升推荐系统的准确性和时效性。

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [参数说明](#参数说明)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [实验结果](#实验结果)
- [引用](#引用)

## 🎯 项目简介

TDS-GCN (Time-aware Dynamic Sparse Graph Convolutional Network) 是一个基于**图神经网络（GCN）**和**协同过滤**的推荐系统模型。该模型创新性地将时间信息融入图结构，通过构建**动态图**来捕捉用户兴趣的动态变化，从而提升推荐效果。

### 核心创新点

- **时间感知的动态图建模**：将时间戳信息编码为图边权重，使用指数衰减函数处理历史交互的时间衰减效应
- **正负交互区分**：显式建模正向和负向交互，通过可学习的权重矩阵动态调整其对训练的贡献
- **多层GCN特征融合**：通过拼接不同GCN层的嵌入表示，保留浅层和深层的图结构信息
- **过相关问题缓解**：设计相关性损失函数，有效缓解图神经网络中的过平滑问题

## ✨ 核心特性

- ✅ 支持时间感知的动态图构建
- ✅ 多层图卷积网络（GCN）架构
- ✅ BPR损失函数与协同过滤优化
- ✅ 正负样本对比学习
- ✅ 过相关问题的处理机制
- ✅ 支持多种损失加权策略（调和平均、简单平均、均方）
- ✅ 完整的训练、验证、测试流程

## 🔧 环境要求

- Python >= 3.7
- PyTorch >= 1.8.0
- CUDA >= 10.2 (如果使用GPU)
- DGL >= 0.6.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Pandas >= 1.1.0

## 📦 安装步骤

### 1. 克隆仓库

```bash
git clone <repository-url>
cd TDS-GCN
```

### 2. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install dgl
pip install numpy scipy pandas
pip install scikit-learn
```

### 3. 配置路径

修改 `CONST/path_const.py` 中的 `root_path` 为您的数据路径：

```python
root_path = '/your/data/path'
```

## 📊 数据准备

### 数据格式要求

项目需要以下数据文件（存储在 `data/` 目录下）：

1. **原始数据文件**：
   - `yelp_academic_dataset_user.xlsx` - 用户信息
   - `yelp_academic_dataset_business.xlsx` - 商家信息
   - `yelp_academic_dataset_review.xlsx` - 评论数据（包含时间戳）

2. **处理后的数据文件**（通过数据预处理脚本生成）：
   - `train_iter_class.pkl` - 训练交互矩阵
   - `val_data.pkl` - 验证集数据
   - `test_data.pkl` - 测试集数据
   - `multi_graph_A.pkl` - 多图结构A
   - `multi_graph_B.pkl` - 多图结构B（时间信息）
   - `uu_graph.pkl` - 用户-用户图
   - `ii_graph.pkl` - 物品-物品图
   - `pos_inter_TDSGCN.pkl` - 正向交互矩阵
   - `neg_inter_TDSGCN.pkl` - 负向交互矩阵
   - `pos_inter_timestamp_TDSGCN.pkl` - 正向交互时间戳
   - `neg_inter_timestamp_TDSGCN.pkl` - 负向交互时间戳

### 数据预处理

运行数据预处理脚本生成所需的图结构数据：

```bash
python data_process/generate_adj.py
python data_process/split_dataset.py
```

## 🚀 使用方法

### 快速开始

使用提供的脚本运行TDS-GCN模型：

```bash
bash scripts/TDSGCN.sh
```

### 命令行运行

直接使用Python运行：

```bash
python run.py \
    --model_name 'TDSGCN' \
    --epochs 20 \
    --n_layers 2 \
    --hidden_dim 512 \
    --dims '[512, 512]' \
    --top_k 10 \
    --batch_size 128 \
    --time_step 30 \
    --loss_weight_method 'MS' \
    --learning_rate 0.01 \
    --scaling_factor 0.3
```

### 支持的其他模型

项目还支持以下模型：

- `my_model`: 基础模型
- `LightGCN`: LightGCN模型
- `Afd_LightGCN`: AFD-LightGCN模型
- `KGCN`: 知识图谱卷积网络

```bash
# 运行LightGCN
python run.py --model_name 'LightGCN' ...

# 运行KGCN
python run.py --model_name 'KGCN' ...
```

## ⚙️ 参数说明

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | str | 'TDSGCN' | 模型名称 |
| `--hidden_dim` | int | 16 | 隐藏层维度 |
| `--dims` | str | '[16]' | 各层维度列表 |
| `--n_layers` | int | 1 | GCN层数 |
| `--slope` | float | 0.1 | LeakyReLU斜率 |
| `--weight` | flag | False | 是否使用可学习权重 |

### 时间相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--time_step` | int | 3 | 时间步长（天数） |
| `--scaling_factor` | float | 0.3 | 时间衰减缩放因子 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 20 | 训练轮数 |
| `--batch_size` | int | 128 | 批次大小 |
| `--learning_rate` | float | 0.001 | 学习率 |
| `--decay` | float | 0.5 | 学习率衰减率 |
| `--patience` | int | 5 | 早停耐心值 |
| `--min_lr` | float | 1e-5 | 最小学习率 |

### 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--coef_bpr` | float | 1.0 | BPR损失权重 |
| `--coef_reg` | float | 0.1 | 正则化损失权重 |
| `--coef_uu` | float | 0.1 | 用户-用户DGI损失权重 |
| `--coef_ii` | float | 0.1 | 物品-物品DGI损失权重 |
| `--handle_over_corr` | flag | True | 是否处理过相关问题 |
| `--loss_weight_method` | str | 'MS' | 损失加权方法（'HM'/'SM'/'MS'） |

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--datasetPath` | str | '/root/autodl-temp/data' | 数据集路径 |
| `--n_NegSamples` | int | 50 | 负样本数量 |
| `--top_k` | int | 10 | Top-K推荐 |
| `--rating_class` | int | 3 | 评分类别数 |

## 📁 项目结构

```
TDS-GCN/
├── CONST/                  # 常量配置
│   └── path_const.py      # 路径配置
├── data_loader/           # 数据加载器
│   └── BPRData.py         # BPR数据加载
├── data_process/          # 数据处理
│   ├── generate_adj.py    # 生成邻接矩阵
│   └── split_dataset.py   # 数据集划分
├── exp/                   # 实验脚本
│   ├── exp_basic.py       # 基础实验类
│   ├── exp_main.py        # 主实验类
│   ├── exp_TDSGCN.py      # TDSGCN实验
│   ├── exp_LightGCN.py    # LightGCN实验
│   └── ...
├── model/                 # 模型定义
│   ├── TDSGCN.py         # TDS-GCN主模型
│   ├── layers.py          # GCN层定义
│   ├── Embed.py           # 时间嵌入
│   ├── DGI.py             # 深度图信息最大化
│   └── ...
├── utils/                 # 工具函数
│   ├── loss.py            # 损失函数
│   ├── metrics.py         # 评估指标
│   ├── get_data_tools.py  # 数据获取工具
│   └── ...
├── scripts/               # 运行脚本
│   ├── TDSGCN.sh         # TDSGCN运行脚本
│   ├── yelp.sh           # Yelp数据集脚本
│   └── ...
├── run.py                 # 主入口文件
└── README.md              # 项目说明
```

## 🏗️ 模型架构

### TDS-GCN架构图

```
输入: 用户-物品交互图 + 时间戳信息
  ↓
时间嵌入层 (TimeEmbedding)
  ↓
动态图构建 (weighted_pos_neg_matrix)
  ├─ 正向交互图 (带时间衰减)
  └─ 负向交互图 (带时间衰减)
  ↓
多层GCN
  ├─ GCN Layer 1
  ├─ GCN Layer 2
  └─ ...
  ↓
特征融合 (拼接各层嵌入)
  ↓
输出: 用户嵌入 + 物品嵌入
```

### 核心组件

1. **时间嵌入模块** (`TimeEmbedding`): 使用正弦/余弦位置编码将时间信息映射到高维特征空间
2. **动态图构建** (`weighted_pos_neg_matrix`): 基于时间衰减和可学习权重构建动态图
3. **GCN层** (`GCNLayer`): 实现图卷积操作，支持消息传递和特征聚合
4. **相关性损失** (`cal_corr_loss`): 防止多层GCN导致的特征过度平滑

## 📈 实验结果

模型在Yelp等公开数据集上进行了验证，评估指标包括：

- **Hit Ratio@K (HR@K)**: 命中率
- **NDCG@K**: 归一化折损累积增益

### 性能表现

（请根据实际实验结果填写）

| 数据集 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|--------|-------|---------|-------|---------|
| Yelp   | -     | -       | -     | -       |

## 📝 日志

训练日志保存在 `scripts/logs/train.log`，包含：
- 训练损失
- 验证集性能（HR, NDCG）
- 测试集性能
- 模型保存信息

## 🔍 评估指标

项目使用以下评估指标：

- **Hit Ratio (HR)**: 推荐列表中包含目标物品的比例
- **NDCG (Normalized Discounted Cumulative Gain)**: 考虑排序位置的推荐质量指标

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件

## 🙏 致谢

感谢以下开源项目：
- [DGL](https://www.dgl.ai/) - 深度图学习库
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**注意**: 使用前请确保已正确配置数据路径和依赖环境。
