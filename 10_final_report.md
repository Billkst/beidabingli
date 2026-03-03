# IgAN 激素疗效预测项目技术报告

## 1. 项目目标与范围
本项目目标是基于 IgA 肾病患者基线临床变量与病理评分（MEST-C），预测激素治疗的短期（6个月，`label1`）与长期（12个月，`label2`）疗效概率。当前仅聚焦两条任务线：
1. 仅临床变量预测疗效
2. 临床变量 + MEST-C 评分预测疗效

> 说明：病理图像相关任务不在当前范围。

---

## 2. 数据纳入与处理策略
**数据来源**：277 例符合 TESTING 纳排标准病人（见 [变量说明.txt](变量说明.txt)）

**基线信息约束**
- 除 `Biopsydate` 外均为基线信息，`Biopsydate` 在建模中已排除。

**共线性处理**（严格执行）
- `baseline GFR` 与 `baseline Scr` 不同时入模。
- `MAP` 与 `SBP/DBP` 不同时入模。

**核心预测变量**（按说明至少纳入）
- 年龄、GFR（或 Scr）、UTP、MAP（或 SBP+DBP）。

**缺失值处理策略**
- **双策略并行**：
  1) **核心稳健版**：仅使用低缺失率（<5%）的核心变量，避免引入噪音。
  2) **插补增强版**：对高缺失变量（如 IgA, C3）进行插补，保留更多信息。
- **具体方法**：
  - **KNN 插补**：基于特征相似性填补缺失值，保留数据局部结构。
  - **缺失指示变量**（Missing Indicator）：对于非随机缺失（MNAR），增加二值变量（如 `IgA_missing`）以捕捉“缺失本身”携带的信息。

**MEST-C 对照**
- 同时训练“仅临床”与“临床+MEST-C”方案并对比性能。

---

## 3. 分析流程与方法
### 3.1 数据检查
脚本：[01_data_inspection.py](01_data_inspection.py)
- 列字段核对与类型检查
- 缺失值统计

### 3.2 建模与优化阶段
- 基础模型训练与解释：[02_model_training.py](02_model_training.py)
- 插补增强与调参：[04_phase2_optimization.py](04_phase2_optimization.py)
- 深度随机搜索：[05_phase3_max_optimization.py](05_phase3_max_optimization.py)
- 校准优化：[06_phase4_calibration.py](06_phase4_calibration.py)
- 共线性对照试验：[07_phase5_confounder_sensitivity.py](07_phase5_confounder_sensitivity.py)

### 3.3 最终模型确立与预测工具
- 模型训练与保存：[08_train_final_models.py](08_train_final_models.py)
- 高阶集成优化：[12_phase6_advanced_optimization.py](12_phase6_advanced_optimization.py)
- 深度表示学习探索：[13_phase7_deep_feature_embedding.py](13_phase7_deep_feature_embedding.py)
- 预测脚本：[09_predict.py](09_predict.py)
- 模型文件：
  - [models/short_term_xgb.joblib](models/short_term_xgb.joblib)
  - [models/long_term_xgb.joblib](models/long_term_xgb.joblib)
  - 元信息：
    - [models/short_term_xgb_meta.json](models/short_term_xgb_meta.json)
    - [models/long_term_xgb_meta.json](models/long_term_xgb_meta.json)

---

## 4. 模型方案与技术细节
### 4.1 特征工程（Feature Engineering）
- **缺失处理**：引入 `_missing` 后缀变量，显式建模缺失模式。
- **领域知识特征**：
  - **比值特征**：`IgA / C3`（反映由于免疫复合物沉积导致的补体消耗相对水平）。
  - **交互特征**：`baseline UTP × S`（探索蛋白尿与病理 S 分段的非线性协同效应）。
- **非线性变换**：
  - **分箱（Binning）**：将连续变量（Age, UTP）离散化为 One-hot 编码，辅助线性模型捕捉非线性关系。

### 4.2 模型选择逻辑
采用了从“可解释基线”到“高性能黑盒”的阶梯式建模策略：
1.  **ElasticNet Logistic Regression**：
    -   作为线性基线（Baseline）。
    -   利用 L1/L2 正则化进行特征筛选，适合高相关性特征场景。
2.  **SVM (RBF Kernel)**：
    -   引入径向基核函数，捕捉高维空间中的非线性模式。
3.  **XGBoost (Gradient Boosting Decision Tree)**：
    -   主要性能模型。
    -   优势：自动处理缺失值（Sparsity-aware）、能够捕捉高阶特征交互、鲁棒性强。

### 4.3 调参与评估策略
- **评估框架**：**5折分层交叉验证（Stratified 5-fold CV）**
  -   *解读*：确保每折（Fold）中正负样本比例与总体一致，解决样本不平衡带来的评估偏差。
- **评价指标**：
  -   **AUC (Area Under Curve)**：衡量模型的排序能力（区分度）。
  -   **Brier Score**：衡量预测概率与真实标签的均方误差（校准度），越低越好。
- **超参数优化**：
  -   **RandomizedSearchCV**：在参数空间内随机采样，比网格搜索（GridSearch）更高效地找到局部最优解。
  -   重点调节：`learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha/lambda`。

---

## 5. 关键结果摘要
### 5.1 Phase 1（基线）
- 短期：AUC≈0.612（临床）
- 长期：AUC≈0.634（临床+MEST）

### 5.2 Phase 2（插补+调参）
- 短期最佳：AUC≈0.706（XGBoost + MEST）
- 长期最佳：AUC≈0.681（SVM）

### 5.3 Phase 3（深度随机搜索）
- 短期最佳：AUC≈0.699（XGBoost Core）
- 长期最佳：AUC≈0.685（XGBoost Core+MEST）

### 5.4 Phase 4（校准优化）
- **Platt Scaling / Isotonic Regression**：尝试对模型输出概率进行校准。
- 结果：校准总体无显著收益，仅“长期 Core + SVM”略有改善。说明 XGBoost 原始输出概率分布已相对合理。

### 5.5 Phase 5（共线性对照）
- 短期最稳健：GFR + MAP（XGB AUC≈0.691）
- 长期 GFR 与 Scr 表现接近，MAP 优于 SBP/DBP。

### 5.6 Phase 6（集成与深度优化）
- **Stacking / Voting Ensemble**：融合 LR (Linear), SVM (RBF), XGBoost (Tree) 和 Random Forest。
- **结果**：
  - **短期**：Voting 提升微弱（0.687 -> 0.690），说明单一强模型（XGB）已接近数据上限。
  - **长期**：Voting 带来约 1% 的 AUC 提升（0.674 -> 0.684），证明异构模型互补性在长期任务中更明显。

### 5.7 Phase 7（深度表示学习）
- **Deep Feature Embedding**：训练层数为 [d_in, 128, 64] 的 **Denoising AutoEncoder (DAE)** 进行无监督特征提取。
  - 目的：引入非线性流形（Manifold）特征，辅助树模型决策。
- **结果**（XGBoost + DAE Features）：
  - **短期**：AUC 下降至 0.631。*分析*：小样本下，深度网络的 Latent Feature 引入了噪音多于信息，导致过拟合。
  - **长期**：AUC 0.681。与传统模型持平，但未超越 Voting Ensemble。
- **结论**：在 N=277 的尺度下，深度学习生成的 Embedding 并未比专家手工特征（Feature Engineering）更有效。

### 5.8 Phase 8（稳健性与临床增益评估）
本阶段引入 **重复交叉验证 (Repeated CV)**、**DCA 决策曲线** 及更多算法对照，验证模型在小样本下的真实效能。
- **真实性能摸底 (Repeated 5-Fold CV)**：
  - **短期**：Voting Ensemble (AUC 0.671) > Single XGB (0.662)。
  - **长期**：Voting Ensemble (AUC 0.660) > Single XGB (0.630)。
  - *解读*：单次 CV 的结果（此前约 0.70）存在一定乐观偏差。集成模型在多次重复验证中表现出更强的抗干扰能力，尤其是长期任务。
- **算法对照**：CatBoost (AUC~0.62) 与 LightGBM (AUC~0.65) 未能超越 XGBoost 体系。
- **工程改进**：
  - **MICE 多重插补** 对 SVM 模型有显著增益（长期 AUC 达 0.670），成为仅次于集成模型的单模型强棒。
  - **类别不平衡处理**：使用 RandomUndersampling 或 Class Weight 虽略微提升 AUC，但会显著破坏 Brier Score（校准度恶化），不建议在最终部署中过度使用。
- **临床效用 (DCA)**：决策曲线显示，在阈值 0.3-0.6 的区间内，Voting Ensemble 模型能提供正向的净获益（Net Benefit），优于“全都不治”或“全都治”策略。

详细结果表格：
- [results_phase2/phase2_optimization_results.csv](results_phase2/phase2_optimization_results.csv)
- [results_phase3/phase3_max_optimization_results.csv](results_phase3/phase3_max_optimization_results.csv)
- [results_phase4/phase4_calibration_results.csv](results_phase4/phase4_calibration_results.csv)
- [results_phase5/phase5_confounder_sensitivity.csv](results_phase5/phase5_confounder_sensitivity.csv)
- [results_phase6_ensemble/phase6_ensemble_results.csv](results_phase6_ensemble/phase6_ensemble_results.csv)
- [results_phase8_repeated_cv/phase8_repeated_cv_results.csv](results_phase8_repeated_cv/phase8_repeated_cv_results.csv)

---

## 6. 最终模型定稿
综合 Phase 1-8 的全方位评估：

- **短期疗效预测（label1）**：**Voting Ensemble (XGB+LR+SVM+RF)**
  - *变更理由*：虽然单模型 XGBoost表现尚可，但在 Phase 8 重复验证中，Voting 展现了更佳的稳定性（AUC 0.671 vs 0.662）与更高的特异度。
- **长期疗效预测（label2）**：**Voting Ensemble (XGB+LR+SVM+RF)**
  - *理由*：集成方案优势显著（AUC 0.660 vs 0.630），是应对长期预测不确定性的最佳选择。

模型已保存：
- [models/short_term_xgb.joblib](models/short_term_xgb.joblib) (保留作为轻量级备选)
- [results_phase6_ensemble/ShortTerm_Ensemble_Voting.joblib](results_phase6_ensemble/ShortTerm_Ensemble_Voting.joblib) (推荐)
- [results_phase6_ensemble/LongTerm_Ensemble_Voting.joblib](results_phase6_ensemble/LongTerm_Ensemble_Voting.joblib) (推荐)

---

## 7. 输出与使用方式
### 7.1 输出
输入患者变量后输出：
- `pred_label1_prob`：短期疗效概率
- `pred_label2_prob`：长期疗效概率

### 7.2 使用脚本
- 预测脚本：[09_predict.py](09_predict.py)
- 输入模板：[input_template.csv](input_template.csv)
- 使用示例：[11_prediction_example.md](11_prediction_example.md)

---

## 8. 局限性与后续建议
1. 样本量较小（N=277），模型波动仍存在。
2. 部分变量缺失率高，插补带来不确定性。
3. 后续可引入病理图像特征作为增量提升方向。

---

## 9. 附录：主要中间产物
- 初始建模脚本：[02_model_training.py](02_model_training.py)
- 优化脚本：[04_phase2_optimization.py](04_phase2_optimization.py)
- 深度优化：[05_phase3_max_optimization.py](05_phase3_max_optimization.py)
- 校准优化：[06_phase4_calibration.py](06_phase4_calibration.py)
- 共线性对照：[07_phase5_confounder_sensitivity.py](07_phase5_confounder_sensitivity.py)
- 最终模型训练：[08_train_final_models.py](08_train_final_models.py)
