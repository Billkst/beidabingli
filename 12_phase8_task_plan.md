# Phase 8 方法与工程优化任务文档

目标：在样本量 N=277 的前提下，最大化 AUC 稳定性与可复现性，并探索算法与工程层面的增益空间。

## 总体原则
- 所有实验统一数据切分策略，记录随机种子。
- 每个任务输出：AUC 均值、AUC 标准差、Brier Score。
- **[新增]** 每个任务需输出决策曲线分析 (DCA) 图与 Youden Index 敏感度/特异度指标。
- 所有结果输出到独立目录，便于回滚与对比。

---

## 任务清单（6项全部执行）

### Task 0（新增）：增加临床效用评估指标
**目的**：引入 DCA (Decision Curve Analysis) 评估模型在不同阈值下的临床净获益。
**方法**：
- 创建 `dca_utils.py` 模块，封装 Net Benefit 计算与绘图函数。
- 在后续所有任务脚本中调用。

### Task 1：CatBoost 替代建模
**目的**：验证 CatBoost 对小样本与缺失值的鲁棒优势。
**方法**：
- 采用 CatBoostClassifier。
- 保持与当前特征工程一致（缺失指示 + 比值/交互）。
- 5折分层交叉验证。
**输出**：`results_phase8_catboost/phase8_catboost_results.csv`

### Task 2：LightGBM 替代建模
**目的**：验证 LightGBM 的 leaf-wise 生长策略是否提升区分度。
**方法**：
- 采用 LGBMClassifier。
- 与 Task1 相同的特征工程与验证方式。
**输出**：`results_phase8_lightgbm/phase8_lightgbm_results.csv`

### Task 3：XGBoost 单调约束（Monotonic Constraint）
**目的**：将医学先验纳入模型以减少过拟合。
**方法**：
- 对核心变量设置单调约束：
  - `baseline GFR`：正约束
  - `baseline UTP`：负约束
  - `MAP`：负约束
  - `age`：负约束
  - `Alb`：正约束
- 其余变量不设约束。
**输出**：`results_phase8_xgb_mono/phase8_xgb_mono_results.csv`

### Task 4：MICE/IterativeImputer 插补增强
**目的**：对比 MICE 与 KNN 插补的差异，检验是否提升稳定性。
**方法**：
- 使用 IterativeImputer (MICE) + 缺失指示变量。
- 保持模型为 XGBoost/SVM/LR 作为对照。
**输出**：`results_phase8_mice_impute/phase8_mice_impute_results.csv`

### Task 5：Repeated Stratified CV 稳健性评估
**目的**：减少小样本 CV 评估方差。
**方法**：
- RepeatedStratifiedKFold（5折 × 5次重复）。
- 评估模型：XGBoost 与 Voting Ensemble。
**输出**：`results_phase8_repeated_cv/phase8_repeated_cv_results.csv`

### Task 6：类别不平衡处理
**目的**：检查样本比例对模型性能的影响。
**方法**：
- 统计 label1 / label2 的正负比例。
- 尝试以下策略并对比：
  1) `class_weight='balanced'`（LR/SVM）
  2) XGB `scale_pos_weight`
  3) 轻量级随机下采样（仅用于训练折）
**输出**：`results_phase8_class_imbalance/phase8_class_imbalance_results.csv`

---

## 交付物
1. 各任务脚本（独立文件）。
2. 每个任务对应的结果 CSV。
3. `10_final_report.md` 增加 Phase 8 章节与结论。

---

## 执行顺序建议
1. Task 1 + Task 2（替代模型）
2. Task 3（单调约束）
3. Task 4（插补方法）
4. Task 5（稳健性评估）
5. Task 6（不平衡处理）

如需执行，请确认后续优先级或直接回复“按计划开始”。
