# 创业板指数增强课程项目

已整合 A 的行情清洗和数据划分、B 的特征处理、Ridge / LightGBM / TQNet / PatchTST 四基准模型、交易策略与风控。2026-11-20 加入“第二轮最终交付”和“第二阶段统一建模数据修订版”。本次整理保留已交付的模型、预测和成绩，补齐运行依赖，没有重新调参或训练。

## 本次交付入口

| 用途 | 路径 |
| --- | --- |
| A / 组长接收四份最终预测 | [outputs/preds/](outputs/preds/)，按 date 合并 |
| C / D 使用当前统一建模数据 | [dataset/revised/README.md](dataset/revised/README.md) |
| 查看清晰目录与代码职责 | [目录与协作指南](docs/目录与协作指南.md) |
| 最终成绩、配置与复现操作 | [B最终结果与复现](docs/B最终结果与复现.md) |
| 最终预测与资金指标 | [outputs/metrics/](outputs/metrics/) |
| 三轮报告、候选与交接说明 | [reports/baseline_tuning/](reports/baseline_tuning/) |
| 原包说明与来源哈希 | [docs/delivery_20261120/](docs/delivery_20261120/) |

初始资金 100,000 元，投资期 2025-10-10 收盘至 2026-10-23 收盘。使用冻结的收益—风险—成本分档策略，次日开盘成交，含约定税费、零滑点、期末市值估值。

| 模型 | 累计收益率 | 净收益 | 期末资产 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| Ridge | +6.74% | +6,742.28 元 | 106,742.28 元 | 14.06% |
| LightGBM | +29.58% | +29,582.51 元 | 129,582.51 元 | 11.67% |
| TQNet | +16.19% | +16,192.51 元 | 116,192.51 元 | 20.22% |
| PatchTST | +1.22% | +1,223.84 元 | 101,223.84 元 | 11.27% |

最终运行为 `runs/20261119_233827_annualret_final_cd62b2`。这些是已查看过测试区间的历史开发回测。最终推荐参考过测试诊断，并非严格的仅验证集选参；早停与搜索重复使用同一验证期。第三轮只收录报告、试验表和代码，原交付包没有第三轮运行目录。完整披露和具体成员配置见最终结果说明。

```powershell
# 不训练，检查归档与最终交付
python -X utf8 verify_delivery.py

# 使用本机已有 CPU 与 RTX 5060 环境，从最终权重复现
.\reproduce_final.ps1
```

`requirements-models.txt` 记录模型额外依赖，PyTorch/CUDA 复用硬件兼容的现成环境。新复现输出写入 `work/`，不修改归档。官方骨干来源、固定提交及许可证在 `vendor/`。

以下保留 A 原阶段的流程说明；当前模型使用修订数据及文末接口更新。

## 先读说明

- C/D 开始建模：[A第二阶段数据集说明](docs/A第二阶段数据集说明.md)。
- B 特征公式与交付：[B阶段特征交接说明](docs/B阶段特征交接说明.md)。
- 协作接口：[接口约定](接口约定.md)。
- 回测税费：[税费更新说明](税费更新说明.md)。

## 文件位置

| 路径 | 内容 |
|---|---|
| data/ | 老师的原始行情 TXT |
| outputs/tables/clean_prices.csv | A 的干净行情 |
| outputs/tables/raw_features.csv | B 的原始特征表 |
| src/features.py | B 的特征生成代码 |
| src/dataset.py | A 的标签和时间切分代码 |
| dataset/train/ | 训练特征和答案，3465行 |
| dataset/val/ | 验证特征和答案，242行 |
| dataset/test/ | 测试特征和评估答案，251行 |
| main.py | 第一阶段清洗、EDA、含税费回测入口 |
| build_dataset.py | 第二阶段标签与切分入口 |
| tests/ | 数据集和税费验证 |

## 统一数据规则

标签为下一交易日收盘收益率 close[t+1]/close[t]-1，0.03表示上涨3%。验证起点2024-10-11，历史截止2025-10-10。测试信号日期2025-10-10至2026-10-22，答案对应2025-10-13至2026-10-23。排除20行预热、1行跨训练/验证边界和最后1行。

A 原始 X 保留 17 列特征。B 已提供 `dataset/revised/` 的 13 列 Ridge 处理表、16 列 LightGBM 输入、交易标签和连续历史；转换只在训练期拟合，全组使用统一日期。原 `dataset/train|val|test` 保留用于追溯。

本仓库为复现保留测试答案和原始行情。测试答案只用于最终评估，不能用于训练、选参和策略调参。metadata含真实未来价格，也不能当模型输入。模型固定后再预测X_test并评估。

## 运行

建议Python 3.11，在仓库根目录依次执行：

```bash
python -m pip install -r requirements.txt
python main.py
python -m src.features
python build_dataset.py
python -m unittest discover -s tests -v
```

已有结果可直接读取，上述命令用于复现 A 原流程。最后一条执行原有 22 项检查。A 依赖仍由 `requirements.txt` 记录，模型依赖另列。

## 交易规则

收盘信号次日开盘成交；双向佣金万3最低5元、双向过户费十万分之一、卖出印花税万5。按扣费后的资产确定仓位，零滑点，允许可分割指数单位，末日按市值估值不强制卖出。费用由回测账户扣除，不从标签中再次扣除。

work/与__pycache__/缓存由.gitignore排除。仓库保留代码、输入和生成数据，无需重复上传交接压缩包。
