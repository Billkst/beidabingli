# 预测输入模板与使用示例

## 1. 输入模板
模板文件：
- [input_template.csv](input_template.csv)

字段说明：
- 必填（核心）：`age`, `gender`, `baseline GFR`, `baseline UTP`, `MAP`, `Alb`, `RASB`, `尿酸`
- 可选（缺失可留空）：`前驱感染`, `肉眼血尿`, `IgA`, `C3`, `血尿（RBC）`, `Hb`, `M`, `E`, `S`, `T`, `C`

> 说明：缺失字段可留空，模型会自动插补并使用缺失指示变量。

---

## 2. 使用示例
运行预测脚本：

- 输入文件：`input_template.csv`
- 输出文件：`pred_output.csv`

执行方式：
```
/home/UserData/ljx/conda_envs/beidabingli/bin/python /home/UserData/ljx/beidabingli/09_predict.py \
  --input /home/UserData/ljx/beidabingli/input_template.csv \
  --output /home/UserData/ljx/beidabingli/pred_output.csv
```

输出字段：
- `pred_label1_prob`: 短期疗效概率
- `pred_label2_prob`: 长期疗效概率
