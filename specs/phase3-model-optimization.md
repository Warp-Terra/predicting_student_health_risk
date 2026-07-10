# Phase 3: 模型优化

## 回顾

| 阶段 | 方案 | CV Accuracy | Public Score |
|---|---|---|---|
| Phase 1 | LightGBM 基线 (13 特征) | 0.94694 | 0.94868 |
| Phase 2 | 特征工程 (多次尝试) | ≤0.94862 | ≤0.94815 |

**结论**：特征工程不奏效，回归 13 原始特征，转向模型侧优化。

## 目标

在 Phase 1 基线 0.94868 基础上提升。

## 方案

### Step 1: Optuna 超参数调优

对 LightGBM 做 5-fold CV + Optuna TPE 搜索：

| 超参 | 搜索范围 |
|---|---|
| `num_leaves` | 31 ~ 255 |
| `learning_rate` | 0.01 ~ 0.1 (log scale) |
| `min_data_in_leaf` | 20 ~ 500 |
| `feature_fraction` | 0.5 ~ 1.0 |
| `bagging_fraction` | 0.5 ~ 1.0 |
| `lambda_l1` | 1e-8 ~ 10.0 (log scale) |
| `lambda_l2` | 1e-8 ~ 10.0 (log scale) |

- 30 trials, early stopping with pruning
- 优化目标：CV multi_logloss（或 accuracy）

### Step 2: 多模型训练

| 模型 | 说明 |
|---|---|
| LightGBM (tuned) | 主模型，用 Optuna 最优参数 |
| CatBoost | 原生类别 + 缺失值支持，默认参数 |
| XGBoost | 需预处理（label encode 类别 + 简单填充缺失值） |

每个模型：
- 5-fold CV 训练
- 保存 fold 模型 + 全量模型
- 输出 OOF 预测

### Step 3: Ensemble

- **简单平均**：3 个模型的 OOF 预测加权平均，测试集取 5-fold 均值
- **加权搜索**：网格搜索最优权重组合

### Step 4: 提交

- 最优 ensemble 方案生成提交文件

## 预期收益

- Optuna 调参：CV +0.001 ~ +0.003
- 多模型集成：CV +0.001 ~ +0.003
- 合计：Public Score 目标 ≥ 0.9495
