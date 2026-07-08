# Phase 1: LightGBM 快速基线

## 目标

用 LightGBM 搭建一个端到端的快速基线，完成一次提交，作为后续优化的基准分数。

## 为什么选择 LightGBM

1. **原生缺失值处理**：LightGBM 在分裂时会自动将缺失值分配到增益更大的分支，无需手动填充，避免了因填充不当引入的偏差
2. **原生类别特征支持**：只需将类别列标记为 `categorical_feature`，LightGBM 内部使用最优切分算法处理，无需 one-hot 或 label encoding
3. **训练速度快**：69 万行数据在 CPU 上几分钟内完成，迭代周期短
4. **对不均衡数据友好**：通过 `class_weight` 参数平衡各类别权重

## 数据概况

| 属性 | 值 |
|---|---|
| 训练集行数 | 690,089 |
| 测试集行数 | 295,754 |
| 任务类型 | 三分类 |
| 目标列 | `health_condition` |
| 目标分布 | `at-risk` 592,561 (85.9%), `unhealthy` 57,724 (8.4%), `fit` 39,803 (5.8%) |

### 特征列表

**数值特征（7 个）**：
- `sleep_duration` - 睡眠时长
- `heart_rate` - 心率
- `bmi` - BMI 指数
- `calorie_expenditure` - 卡路里消耗
- `step_count` - 步数
- `exercise_duration` - 运动时长
- `water_intake` - 饮水量

**类别特征（6 个）**：
- `diet_type` - 饮食类型（veg / non-veg / balanced）
- `stress_level` - 压力水平（high / low / medium）
- `sleep_quality` - 睡眠质量（average / poor / good）
- `physical_activity_level` - 体力活动水平（sedentary / moderate / active）
- `smoking_alcohol` - 吸烟/饮酒（yes / occasional / no）
- `gender` - 性别（female / other / male）

**ID 列**：
- `id` - 样本标识，不参与训练

### 缺失值分布

| 特征 | 缺失数量 | 缺失比例 |
|---|---|---|
| `sleep_duration` | 75,999 | 11.0% |
| `heart_rate` | 7,833 | 1.1% |
| `bmi` | 13,898 | 2.0% |
| `calorie_expenditure` | 52,853 | 7.7% |
| `step_count` | 13,916 | 2.0% |
| `exercise_duration` | 6,901 | 1.0% |
| `water_intake` | 43,477 | 6.3% |
| `diet_type` | 6,901 | 1.0% |
| `stress_level` | 82,811 | 12.0% |
| `sleep_quality` | 58,331 | 8.5% |
| `physical_activity_level` | 36,621 | 5.3% |
| `smoking_alcohol` | 28,582 | 4.1% |
| `gender` | 21,373 | 3.1% |

## 评估指标

Kaggle 使用 **accuracy** 作为评估指标。预测结果为三个类别（`fit`、`at-risk`、`unhealthy`）。

## 实施计划

### Step 1: 项目骨架搭建

```
├── data/                    # 数据目录（已存在）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── specs/                   # 方案文档
│   └── phase1-lightgbm-baseline.md
├── src/
│   ├── config.py            # 全局配置（路径、参数、标签映射）
│   ├── data.py              # 数据加载与预处理
│   ├── train.py             # 训练脚本
│   └── predict.py           # 推断与提交生成
├── submissions/             # 提交文件输出
└── models/                  # 模型保存
```

### Step 2: 数据加载与预处理 (`src/data.py`)

- 加载 `train.csv` 和 `test.csv`
- 将目标列 `health_condition` 映射为整数标签：`{'fit': 0, 'unhealthy': 1, 'at-risk': 2}`
- 标记类别特征列和数值特征列
- 将类别特征转为 LightGBM 所需的整数编码（LightGBM 要求 `categorical_feature` 列必须是整数类型）
- **不做缺失值填充**，交给 LightGBM 原生处理
- 标记 `id` 列不作为特征

### Step 3: 模型训练 (`src/train.py`)

- **交叉验证策略**：5-fold StratifiedKFold（保持每折目标分布一致）
- **模型参数**：

```python
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'class_weight': 'balanced',     # 处理类别不均衡
    'n_estimators': 5000,
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}
```

- **Early Stopping**：patience=100, 监控 validation multi_logloss
- 记录每折的验证 accuracy 和多分类 logloss
- 使用所有训练数据训练一个全量模型用于最终预测（或使用 OOF 预测作为后续 stacking 的基础）

### Step 4: 推断与提交 (`src/predict.py`)

- 使用训练好的模型对测试集进行预测
- 输出 `submissions/submission_phase1_v1.csv`，格式与 `sample_submission.csv` 一致（`id`, `health_condition`）

### Step 5: 结果验证

- 输出 5-fold CV 的 mean accuracy ± std
- 检查预测的类别分布是否与训练集分布一致

## 预期产出

1. ✅ 一套可复用的训练/预测代码框架
2. ✅ 一次 Kaggle 提交（作为后续优化的基线分数）
3. ✅ 5-fold CV accuracy 评估结果
4. ✅ 特征重要性排名

## 后续优化方向（Phase 2+）

- 特征工程：缺失指示器、交叉特征、异常值处理
- 调参：Optuna / Hyperopt 自动调参
- 模型对比：CatBoost、XGBoost
- 集成：加权平均 / Stacking / Blending
