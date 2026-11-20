# 第二阶段统一建模数据修订版

本包复用 A 的第二阶段价格预测切分及 17 列原始特征，新增交易收益标签、模型输入预处理和连续历史窗口。原始行情和原第二阶段包是复现输入。

## 数据和标签

| 路径 | 内容与用法 |
| --- | --- |
| `train/X_train.csv`、`val/X_val.csv`、`test/X_test.csv` | 原有 17 列无量纲特征，三组各为 3,465、242、251 行。日期为收盘后生成预测的信号日 |
| `train/y_train.csv`、`val/y_val.csv` | 原有 `y_true_ret = close[t+1] / close[t] - 1`，用于次日收盘点位预测 |
| `train/y_trade.csv`、`val/y_trade.csv` | `label_trade_ret = open[t+2] / open[t+1] - 1`，用于次日开盘成交的收益预测，同时记录两个开盘日期。分别为 3,464、241 行 |
| `test/target_dates.csv` | 251 个测试信号对应的目标日期。最后一个信号日缺少 `t+2` 日期，其开盘到开盘收益在本投资期内无法评分 |
| `processed/<split>/X_<split>_ridge.csv` | 13 列去冗余特征，经训练集 1%/99% 截尾并标准化 |
| `processed/<split>/X_<split>_lightgbm.csv` | 16 列去冗余特征，保留原始数值与单位 |
| `history/features_history.csv` | 按交易日连续排列的特征，含原 17 列和 4 列长周期候选。构造序列窗口时使用这张表 |
| `preprocessing.json` | 训练期拟合范围、使用列和转换参数，可对历史行或新数据重放相同预处理 |
| `train/recency_weights.csv` | 按交易日指数衰减的对照权重，半衰期 756 个交易日，按价格及交易任务分别归一到均值 1 |
| `audit/` | 训练集相关系数和 VIF、训练与验证的漂移统计、两类目标上的验证对照结果 |

价格任务继续按 `date` 对齐 A 的 X 与 y。交易任务单独按 `y_trade.csv` 的信号日选择 X，可用 `X.loc[y_trade.index]`。交易目标结束日 `t+2` 必须早于验证开始日或不晚于验证截止日。训练交易标签的最后信号日为 2024-10-08，验证为 2025-10-08。相应的价格标签各多一天。共同的 Ridge 转换参数用两种训练标签都可用的 3,464 行拟合。选择按日期对齐有标签的行后再训练。

测试信号仍从 2025-10-10 开始，第一笔对应 2025-10-13 开盘。最后一个测试信号 2026-10-22 有次日收盘目标，开盘到开盘的收益终点超出包内行情，`return_end_date` 留空。本包只带测试特征和目标日期，评分真值由 A 单独管理。交易标签是未扣费的价格收益，交易费用由新版回测在实际调仓时扣除。

## 模型输入与验证

Ridge 13 列去掉 `ret_1`、`amount_rel_20`、`true_range_pct_mean_14` 和 `close_vs_sma_20`。LightGBM 16 列去掉与 `volume_rel_20` 高度重复的 `amount_rel_20`。两组列均覆盖五类特征。任何模型可从原始表研究其他组合，正式比较前记录所用输入。

长周期候选为 `ret_60`、`ret_120`、`close_vs_sma_60` 和 `short_long_vol_ratio = ret_vol_5 / ret_vol_20`。这些列只依赖信号日及以前的数据。最长窗口需要 120 个历史交易日。验证报告中的长周期对照使用相同训练和验证日期，近期加权及长周期作为独立配置记录。

`audit/validation_comparison.csv` 使用原验证段做探索对照。Ridge 的截尾对照使用同样的列及正则系数 `alpha=5000`。LightGBM 沿用 B 的模型配置，验证集参与早停。验证分数只用于筛选方案，正式测试由组内冻结模型后统一评估。

## 时序模型取窗

窗口必须以当日为末行。使用完整历史表可补足训练、验证、测试分界前已知但没有训练标签的行。若采用组内 Ridge 路径的 13 列量纲处理，可以把存档的训练集预处理参数应用到整张历史表，再按每个信号日取窗。

```python
import json
import pandas as pd
from src.modeling_data import apply_ridge_recipe, build_windows

history = pd.read_csv("history/features_history.csv", index_col="date", parse_dates=["date"])
test = pd.read_csv("test/X_test.csv", index_col="date", parse_dates=["date"])
with open("preprocessing.json", encoding="utf-8") as file:
    recipe = json.load(file)["ridge"]
history = apply_ridge_recipe(history, recipe)
test_windows = build_windows(history, test.index, lookback=64)
```

对 `train`、`val`，在各自的标签日期中选出能提供完整回看窗口的样本，再用相同方法取窗。开头不足窗口的行需从该模型的训练样本中剔除。验证和测试可向前借历史，窗口永远止于信号日。使用同一组评分日期比较不同窗口或模型。

## 从 A 的交付重建

依赖为 `numpy`、`pandas`、`scikit-learn` 和 `lightgbm`。在本包解压后的根目录运行，参数指向 A 的两个原始 ZIP。

```powershell
python -m src.modeling_data --stage1 "第一阶段.zip" --stage2 "第二阶段_统一建模数据.zip" --output "第二阶段_统一建模数据_修订版.zip"
```

程序会核对 A 的文件哈希，复算原始特征与价格标签，然后生成可独立分发的新包。
