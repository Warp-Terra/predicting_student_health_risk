# 学生健康风险预测

> [Kaggle Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) — 基于 LightGBM 的三分类基线模型

*[English version](README.md)*

## 问题描述

根据学生的生活方式和生理数据，将健康状态分为三类：

| 标签 | 描述 |
|------|------|
| `fit` | 健康（5.8%） |
| `unhealthy` | 不健康（8.4%） |
| `at-risk` | 存在风险（85.9%） |

评估指标：**accuracy**。

## 项目结构

```
predicting_student_health_risk/
├── main.py                   # 入口：完整流水线（数据 → 训练 → 预测）
├── resubmit.py               # 快速重新提交（跳过训练，仅加载已有模型预测）
├── src/
│   ├── config.py             # 全局配置：路径、超参、特征列表、smoke test 开关
│   ├── data.py               # 数据加载与预处理（标签编码、类别特征编码）
│   ├── train.py              # 5折 StratifiedKFold 交叉验证 + 全量模型训练
│   └── predict.py            # 集成预测与提交文件生成
├── specs/                    # 方案文档
│   ├── phase1-lightgbm-baseline.md
│   └── phase2-feature-engineering.md
├── data/                     # 原始数据（git 忽略）
├── models/                   # 训练好的模型（git 忽略）
└── submissions/              # 提交文件（git 忽略）
```

## 特征

| 类型 | 特征 |
|------|------|
| 数值型（7 个） | `sleep_duration`（睡眠时长）、`heart_rate`（心率）、`bmi`（BMI 指数）、`calorie_expenditure`（卡路里消耗）、`step_count`（步数）、`exercise_duration`（运动时长）、`water_intake`（饮水量） |
| 类别型（6 个） | `diet_type`（饮食类型）、`stress_level`（压力水平）、`sleep_quality`（睡眠质量）、`physical_activity_level`（体力活动水平）、`smoking_alcohol`（吸烟/饮酒）、`gender`（性别） |

- 缺失值由 LightGBM 原生处理，无需填充
- 类别特征转为整数编码，利用 LightGBM 原生类别特征支持

## 快速开始

### 环境要求

- Python 3.10+
- 依赖：`pandas`、`numpy`、`lightgbm`、`scikit-learn`

### 安装

```bash
pip install pandas numpy lightgbm scikit-learn
```

### 运行

**完整流水线**（训练 + 预测）：
```bash
python main.py
```

**快速验证**（10% 数据、减少轮数 — 用于快速迭代）：
```bash
python main.py --smoke
```

**重新提交**（跳过训练，复用已有模型）：
```bash
python resubmit.py
```

### 输出

- 训练好的模型保存在 `models/lgb_fold{1..5}.pkl` 和 `models/lgb_full.pkl`
- 提交文件保存在 `submissions/submission_phase1_v1.csv`

## 模型

| 细节 | 值 |
|------|-----|
| 算法 | LightGBM (GBDT) |
| 交叉验证策略 | 5折 StratifiedKFold |
| 类别权重 | Balanced |
| 早停 | 50 轮（监控 val logloss） |
| 最大轮数 | 1,000 |
| 学习率 | 0.05 |

### 为什么选择 LightGBM

- **原生缺失值处理** — 分裂时自动将缺失值分配到增益更大的分支，无需手动填充
- **原生类别特征支持** — 无需 one-hot 编码，内部使用最优切分算法
- **训练速度快** — 约 70 万行数据在 CPU 上几分钟内完成
- **类别权重** — 直接处理严重的类别不均衡问题

## 交叉验证结果（Phase 1 基线）

| 指标 | 值 |
|------|-----|
| CV Accuracy（均值 ± 标准差） | ~0.9469 |
| OOF Accuracy | ~0.9469 |
| OOF LogLoss | ~0.1643 |
| Public LB Score | ~0.9487 |

## 路线图

- [x] Phase 1：LightGBM 基线
  - [x] 数据加载与预处理
  - [x] 5折 StratifiedKFold 交叉验证
  - [x] 全量模型训练
  - [x] Kaggle 提交
- [ ] Phase 2：特征工程
  - [ ] 缺失值指示器
  - [ ] 交叉特征 / 组合特征
  - [ ] 目标编码
  - [ ] 异常值裁剪
- [ ] Phase 3：超参数调优（Optuna）
- [ ] Phase 4：模型集成（XGBoost、CatBoost、Stacking）

## 许可证

MIT
