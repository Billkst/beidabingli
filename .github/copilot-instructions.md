# Copilot instructions

## 项目概览与数据流
- 这是一个以脚本为中心的建模流程：从数据检查 → 多阶段调参 → 校准/敏感性分析 → 最终模型训练 → 预测输出。
- 数据主文件为 Excel：/home/UserData/ljx/beidabingli/队列符合277.xlsx（所有脚本默认硬编码此绝对路径）。
- 目标变量为 `label1`（短期 6 个月）与 `label2`（长期 12 个月）。
- 非特征列通常被丢弃：`Unnamed: 26`, `number`, `Biopsydate`, `病理扫片`。

## 关键脚本与阶段产物
- 数据检查：[01_data_inspection.py](01_data_inspection.py)
- 基线模型与解释：[02_model_training.py](02_model_training.py)
- 参数搜索与优化：
  - Phase2 GridSearch：[04_phase2_optimization.py](04_phase2_optimization.py)
  - Phase3 RandomizedSearch：[05_phase3_max_optimization.py](05_phase3_max_optimization.py)
  - Phase4 校准：[06_phase4_calibration.py](06_phase4_calibration.py)
  - Phase5 共线性敏感性：[07_phase5_confounder_sensitivity.py](07_phase5_confounder_sensitivity.py)
- 最终模型训练（读取 Phase3 结果）：[08_train_final_models.py](08_train_final_models.py)
- 预测脚本（读取 models/ 中的模型与元信息）：[09_predict.py](09_predict.py)
- 集成与深度探索：
  - Phase6 集成模型：[12_phase6_advanced_optimization.py](12_phase6_advanced_optimization.py)
  - Phase7 DAE 特征：[13_phase7_deep_feature_embedding.py](13_phase7_deep_feature_embedding.py)

## 项目内固定的特征工程约定
- 多个阶段复用 `add_features()` 逻辑（见 [04_phase2_optimization.py](04_phase2_optimization.py), [05_phase3_max_optimization.py](05_phase3_max_optimization.py), [06_phase4_calibration.py](06_phase4_calibration.py), [07_phase5_confounder_sensitivity.py](07_phase5_confounder_sensitivity.py), [08_train_final_models.py](08_train_final_models.py), [12_phase6_advanced_optimization.py](12_phase6_advanced_optimization.py))：
  - 高缺失变量缺失指示：`<col>_missing`
  - 比值：`IgA_C3_ratio`
  - 交互：`UTP_x_S`
  - 部分阶段还包含分箱 one-hot（`age_bin`, `utp_bin`）
- 缺失值插补以 KNN 为主（`KNNImputer(n_neighbors=5)`），SVM/LR 通常配合 `StandardScaler`。

## 模型与输出
- 默认模型族：ElasticNet LR / SVM(RBF) / XGBoost，评估采用 5 折分层 CV（AUC + Brier）。
- 最终可部署模型保存在：
  - 短期模型：[models/short_term_xgb.joblib](models/short_term_xgb.joblib)
  - 长期模型：[models/long_term_xgb.joblib](models/long_term_xgb.joblib)
  - 元信息：[models/short_term_xgb_meta.json](models/short_term_xgb_meta.json), [models/long_term_xgb_meta.json](models/long_term_xgb_meta.json)
- 预测输出字段：`pred_label1_prob`, `pred_label2_prob`（见 [09_predict.py](09_predict.py) 与 [11_prediction_example.md](11_prediction_example.md)）。

## 运行与使用示例（已在项目内给出）
- 预测示例命令在 [11_prediction_example.md](11_prediction_example.md) 中，使用 conda 环境下的 Python 直跑脚本。

## 修改注意事项
- 绝对路径被硬编码在多个脚本中；如需迁移数据位置，需统一更新各脚本的 `DATA_PATH`/`OUTPUT_DIR` 常量。
- 若修改特征工程，请保持所有阶段脚本的一致性，避免训练-预测特征不对齐（尤其是 `FeatureEngineer`/`add_features()` 的列逻辑）。
