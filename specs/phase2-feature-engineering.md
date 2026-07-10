# Phase 2: 特征工程

> 历史方案：本阶段按普通 accuracy 选择特征，与竞赛官方 Balanced Accuracy 不一致，实验结论仅供追溯。

## 目标

在不改动模型架构的前提下，通过构造衍生特征提升数据质量，使 5-fold CV Accuracy 和 Kaggle Public Score 获得提升。

| 指标 | Phase 1 基线 | Phase 2 目标 |
|---|---|---|
| CV Accuracy | 0.94694 | ≥ 0.9475 |
| Public Score | 0.94868 | ≥ 0.9490 |

## 核心原则

- **原始特征不清除**：树模型天然免疫"特征权重膨胀"，衍生特征与原始特征互为补充
- **增量验证**：每完成一类特征，运行 smoke test 观察 CV 变化，无效则回退
- **不引入数据泄漏**：所有编码类特征必须在 Cross-Validation 内完成，禁止用全局统计量

---

## Feature Set 1: 缺失值指示器

### 原理

LightGBM 原生处理缺失值（分裂时分配到增益更大的一侧），但这只告诉了模型"这个值在哪一侧更好"，没有告诉模型"**这个值根本不存在**"。缺失本身是一个有信息的信号——比如 `stress_level` 缺失的用户，可能本身健康风险不同。

### 实现

为原始 13 个特征各新增一列 `{feat}_is_missing`（int8, 0/1）：

| 原始特征（13 个全含缺失） |
|---|
| `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake` |
| `diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender` |

### 注意事项

- 类别特征的缺失已在 Phase 1 被 `fillna("missing")` 转为普通类别，所以需要**在 fillna 之前**生成缺失指示器
- 测试集同理，两个集合的缺失指示器独立计算

### 预期收益

- CV Accuracy +0.0005 ~ +0.002

---

## Feature Set 2: 交叉特征 & 复合特征

### 2a. BMI 分桶

```
bmi_group（类别）:
  < 18.5  → "underweight"
  18.5-24 → "normal"
  24-28   → "overweight"
  > 28    → "obese"
```

参考中国 BMI 标准。作为新的类别特征加入。

### 2b. 运动强度

```
exercise_intensity = calorie_expenditure / (exercise_duration + 1e-5)
```

单位时间热量消耗，反映运动强度。数值特征。

### 2c. 睡眠综合分

```
sleep_score = sleep_duration × sleep_quality_encoded
```

其中 `sleep_quality_encoded` = {poor: 0, average: 1, good: 2}。数值特征。

### 2d. 健康行为综合分

```
health_behavior_score = step_count × physical_activity_encoded + water_intake
```

其中 `physical_activity_encoded` = {sedentary: 0, moderate: 1, active: 2}。数值特征。

### 2e. 心率健康比

```
heart_bmi_ratio = heart_rate / (bmi + 1e-5)
```

### 2f. 类别特征交互

从 6 个原始类别特征中选择两个组合生成交互特征：

```
diet_stress = diet_type + "_" + stress_level
sleep_activity = sleep_quality + "_" + physical_activity_level
```

共 +2 个新类别特征。

### 注意事项

- `calorie_expenditure / exercise_duration` 在 `exercise_duration` 缺失时结果为 NaN，LightGBM 可原生处理
- 交互特征的缺失值已通过 `fillna("missing")` 处理，不会产生新的缺失类别

### 预期收益

- CV Accuracy +0.001 ~ +0.003

---

## Feature Set 3: 目标编码（Target Encoding）

### 原理

类别特征在 LightGBM 中基于梯度做一阶分裂，但**不会考虑类别与目标的一阶均值关系**。Target Encoding 将类别替换为该类别对应目标的编码值，让树在第一时间就能利用这一阶信息。

### 实现

使用 5-fold 训练数据的 Target Encoding，每折用**其他 4 折**的均值编码当前折，防止数据泄漏。对于测试集，使用 5 折编码的均值。

```python
for col in CATEGORICAL_FEATURES:
    for fold in range(5):
        trn_idx, val_idx = folds[fold]
        # 用 trn_idx 的均值编码 val_idx
        mapping = y_train[trn_idx].groupby(X_train[col][trn_idx]).mean()
        X_train_encoded[col][val_idx] = X_train[col][val_idx].map(mapping)
    # 测试集：5 折均值
    X_test_encoded[col] = X_test[col].map(overall_mapping)
```

### 注意事项

- 编码后原始类别特征**保留**（两者共存）
- 编码特征使用 LightGBM 缺失值处理（NaN 即该类别在训练集中未出现）
- 编码列命名为 `{col}_te`

### 预期收益

- CV Accuracy +0.001 ~ +0.002

---

## Feature Set 4: 异常值处理

### 实现

检查各数值特征的分布，对 >99.5% 分位点或 <0.5% 分位点的值进行 clip：

```python
for col in NUMERICAL_FEATURES:
    lower = train[col].quantile(0.005)
    upper = train[col].quantile(0.995)
    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)
```

仅对原始 7 个数值特征做 clip，不 clip 衍生特征。

### 预期收益

- CV Accuracy +0.0001 ~ +0.0005

---

## 实施计划

### Step 1: 重构 data.py

- 新增 `FEATURE_SETS` 枚举或字典，控制哪些特征集启用
- `prepare_data()` 接受 `feature_sets` 参数
- 每类特征封装为独立函数：
  - `add_missing_indicators()`
  - `add_cross_features()`
  - `add_target_encoding()`
  - `clip_outliers()`

### Step 2: 增量验证

```
python3 main.py --smoke  # Phase 1 基线
→ 开启 Feature Set 1 → smoke test → 记录 CV
→ 开启 Feature Set 2 → smoke test → 记录 CV
→ 开启 Feature Set 3 → smoke test → 记录 CV
→ 开启 Feature Set 4 → smoke test → 记录 CV
```

仅当 smoke test CV 上升（>0.0003）时保留该 Feature Set。

### Step 3: 全量训练 & 提交

选定最优 Feature Set 组合，全量训练，生成提交文件。

---

## 文件变更清单

| 文件 | 变更 |
|---|---|
| `src/config.py` | 新增 `FEATURE_SETS`、交叉特征参数（BMI 阈值等） |
| `src/data.py` | 重构，新增特征工程函数 |
| `src/train.py` | 自动适配新特征列（尤其是类别特征列表变化） |
| `src/predict.py` | 不变 |
| `main.py` | 不变 |

---

## 风险 & 回退策略

- **过拟合风险**：特征集 1+2 是确定的，风险低；特征集 3（目标编码）如果实现不当可能泄漏，必须严格使用 K-Fold 防泄漏
- **回退**：每个 Feature Set 独立开关，关闭即可回退
- **smoke test 参数**：10% 数据 + 2 折，确保快速迭代（<2 分钟/次）
