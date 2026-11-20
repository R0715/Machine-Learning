# A 第二阶段 标签与数据集交接

本包已接收 B 的 17 列原始特征，完成日期对齐、未来收益标签和训练/验证/测试切分。已与第一阶段项目整合。原始行情和 B 的特征 CSV 保留不变。

## 现在看什么 发给谁

处理好的文件全部在 **dataset/**。可以把整个压缩包发给 B、C、D，让大家使用同一套数据。

| 文件夹 | 包含什么 | 用途 |
|---|---|---|
| dataset/train/ | X_train.csv、y_train.csv、metadata_train.csv | 学习参数；特征筛选与缩放参数在这里拟合 |
| dataset/val/ | X_val.csv、y_val.csv、metadata_val.csv | 比较方案、调整模型和策略参数 |
| dataset/test/ | X_test.csv、y_test.csv、metadata_test.csv | 最终预测与评估；测试答案不得参与训练、选参或特征选择 |
| outputs/tables/ | A 的 clean_prices.csv、B 的 raw_features.csv | 输入，便于复现 |
| src/ | B 特征代码、A 标签切分代码及配置 | 程序实现 |
| docs/ | B 原始交接说明 | 特征含义与公式 |
| tests/ | 日期、标签、边界和数据完整性测试 | 修改后复核 |

X 是模型看到的 17 个线索；y 是后来发生的收益率答案。metadata 是核查用的价格和目标日期，包含真实未来值，**不能当作模型特征**。加载时用 date 对齐，不能把 date 或 y 混入特征列。

## 本次统一约定

预测目标：下一交易日收盘收益率 `y_t = close[t+1] / close[t] - 1`。0.03 表示上涨 3%。下一天按数据中的下一交易行计算，不是加一个自然日。

验证起点设为 2024-10-11，历史截止为 2025-10-10，投资结束为 2026-10-23。预测周期和验证起点是 A 为本次交接采用的实现约定，不是老师指定的额外要求。请全组按此版本统一实验；如后续调整，应重新生成并更新所有成员的数据。

| 数据集 | 样本数 | 特征数 | 特征日期 | 答案对应日期 |
|---|---:|---:|---|---|
| 训练 | 3465 | 17 | 2010-07-02 至 2024-10-09 | 2010-07-05 至 2024-10-10 |
| 验证 | 242 | 17 | 2024-10-11 至 2025-10-09 | 2024-10-14 至 2025-10-10 |
| 测试 | 251 | 17 | 2025-10-10 至 2026-10-22 | 2025-10-13 至 2026-10-23 |

测试特征从 10-10 开始，是因为当日收盘形成信号，预测的是 10-13 的收盘表现；对应测试目标仍从 10-13 开始。10-10 这一行没有参与训练或验证输入。标签是收盘到收盘收益，不等于回测中的实际账户收益；账户仍按次日开盘成交计算。

## 为什么排除了 22 行

- 前 20 行滚动窗口尚未预热，部分特征为空，构造样本时排除，不填零。
- 2024-10-10 的下一交易日答案来自 2024-10-11，跨入验证期，因此排除这一训练候选行。
- 2026-10-23 是最后一天，范围内没有下一交易日答案，也没有后续执行日，因此不作为预测样本。

合计 3465 + 242 + 251 + 22 = 3980，所有输入行都有明确去向。具体排除日期和原因见 dataset/excluded_rows.csv。日期摘要见 dataset/split_summary.csv；参数、源文件哈希和检查统计见 dataset/split_audit.json。

## 尚未做的处理

这批 X 保留 B 的全部 17 列原始特征，未做特征筛选、标准化或模型训练。这些属于后续工作：B 只能在训练子集拟合筛选规则和缩放器，再应用于验证与测试。不能用全部 3980 行一起计算缩放参数。

本包包含测试答案是为了统一评估。实际训练函数只接收训练和验证数据；最终预测阶段只传 X_test，拿到预测后再由评估程序与 y_test 比较。

## 运行与复核

在本包目录打开终端，使用已有 Python 3.11 环境：

```powershell
python build_dataset.py
python -m unittest discover -s tests -v
```

你的电脑也可直接使用：

```powershell
& 'D:\miniconda\envs\ml\python.exe' -X utf8 build_dataset.py
& 'D:\miniconda\envs\ml\python.exe' -X utf8 -m unittest discover -s tests -v
```

在新环境中先安装 requirements.txt。运行第一条会覆盖本包 dataset/ 中同名生成结果，但不改动 outputs/tables/ 的输入文件。若改变预测周期或验证起点，务必指定新的输出目录，例如 `python build_dataset.py --horizon 5 --output dataset_h5`；本文统计仅对应默认 H=1。

读取示例：

```python
import pandas as pd
X_train = pd.read_csv('dataset/train/X_train.csv', parse_dates=['date']).set_index('date')
y_train = pd.read_csv('dataset/train/y_train.csv', parse_dates=['date']).set_index('date')['y_true_ret']
assert X_train.index.equals(y_train.index)
```

src/features.py 是 B 的原文件，src/dataset.py 是 A 新增模块，build_dataset.py 是第二阶段独立运行入口。第一阶段 main.py 保持原有清洗、EDA 与含税费回测流程。此说明位于 docs；执行命令和读取数据时应在仓库根目录操作。
