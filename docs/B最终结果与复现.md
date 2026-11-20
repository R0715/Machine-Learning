# B 最终结果与复现

本次整合用户提供的第二轮最终交付和修订数据。原包 SHA256、文件来源与补入依赖见 [provenance.json](delivery_20261120/provenance.json)，打包者原说明见 [原交付说明](delivery_20261120/原交付说明.md)。

## 最终配置与成绩

以下是交易任务，价格预测器作为此前固定对照保留。两个任务的输出头独立训练，不能用交易配置推断价格模型结构。具体结构以检查点和 ensemble.json 为准，run_config.json 还含继承的旧基准设置。

| 模型 | 最终交易预测配置 | 累计收益 | 年化收益 | 净收益 |
| --- | --- | ---: | ---: | ---: |
| Ridge | alpha=10000，单模型 | 6.7423% | 6.7700% | 6,742.28 元 |
| LightGBM | leaves=15，min_child_samples=40，种子 42/43/44 平均 | 29.5825% | 29.7164% | 29,582.51 元 |
| TQNet | 回看 64，周期 1，Adam，lr=5e-4，种子 42/43/44 平均 | 16.1925% | 16.2620% | 16,192.51 元 |
| PatchTST | 回看 64、2 层，lr=5e-4 / 1e-4 / 3e-4 三成员平均 | 1.2238% | 1.2287% | 1,223.84 元 |

初始资金 100,000 元，投资期 2025-10-10 收盘至 2026-10-23 收盘。风险系数 3、方差窗口 60、成本系数 1、回撤线 10%、冷静期 5 个信号日。分档候选为 0%、25%、50%、75%、100% 及当前持仓。允许可分割指数单位，零滑点，次日开盘成交，佣金双向万 3 且最低 5 元，过户费双向万分之 0.1，卖出印花税万 5，期末不强制清仓。

策略是 Boyd 收益—风险—成本目标的项目适配，分档和回撤规则属于项目实现。本表使用冻结策略，不能与早期零阈值满仓/空仓结果直接归因比较。

精确指标在 [outputs/metrics/](../outputs/metrics/)。测试信号 251 条；价格评分 251 条，交易评分 250 条。最后一条交易目标超出行情，但仍可用于下一开盘调仓，投资期末按收盘市值评价。

## 复现最终检查点

无需重训。在仓库根目录执行，入口会核对验证和测试预测、恢复全部集成成员，再生成新预测、测试指标与冻结策略回测。资金结果与原成绩误差须小于 0.01 元。

```powershell
# 本机已有 Anaconda CPU 与 RTX 5060 环境
.\reproduce_final.ps1

# 其他电脑显式传入解释器
.\reproduce_final.ps1 -CpuPython C:/envs/cpu/python.exe -GpuPython C:/envs/torch/python.exe

# 一个已有全部依赖的环境，也可 CPU 复现全部模型
python -X utf8 reproduce_final.py --stage all --device cpu --output work/reproduction_new
```

输出使用新目录，检查记录为 `cpu_reproduction_checks.json`、`gpu_reproduction_checks.json` 和 `REPRODUCTION_PASSED.json`。集成从本地 member 文件读取。新表的 model_id 包含新运行名，故哈希可不同；预测数值和资金结果另行核对。

CPU 依赖为 scikit-learn 1.6.1、LightGBM 4.6.0。现成 GPU 环境使用 PyTorch 2.10.0+cu128。模型其他依赖见 requirements-models.txt，不默认重装 PyTorch/CUDA。

原包中的交易 Ridge 序列化于 scikit-learn 1.9.1，本机 1.6.1 加载时会提示版本差异。本次已逐条核对该模型的验证、测试预测一致，资金回测也一致；保留原权重，没有为消除提示改写模型。后续换环境仍须运行检查点复现核对。

## 从头重跑搜索

使用新输出目录，不要覆盖归档。搜索在同一个环境还需 scikit-learn、LightGBM 和 PyTorch。本机现有 GPU 环境缺少 LightGBM，没有把搜索作为自动复现步骤，也没有自动安装依赖。

```powershell
python -X utf8 tune_annual_return.py --source runs/20261119_165036_ed592b --output work/r1_new --device cpu --cpu-threads 2
python -X utf8 verify_annualret.py --source runs/20261119_165036_ed592b --output work/r1_new --device cpu
python -X utf8 tune_annual_return_r2.py --r1-run work/r1_new --output work/r2_new --device cpu --cpu-threads 2
python -X utf8 verify_annualret_r2.py --source work/r1_new --output work/r2_new --device cpu
python -X utf8 assemble_final.py --r1-run work/r1_new --r2-run work/r2_new --output work/final_new --device cpu
```

这是原搜索和组装流程。重新训练的 RNG、环境和浮点差异可能改变候选及收益；原组装脚本固化了历史推荐成员。核对已交付成绩优先使用检查点复现。`run_5060.ps1` 是较早 336 日基准轮的训练入口，不是最终集成复现入口。

## 如实披露

1. 测试区间此前已查看。多轮开发和最终推荐参考过测试诊断，属于历史开发回测，不是新的未见样本成绩。
2. 早停与搜索共用同一验证期，种子排名波动较大，不能据此认定模型已稳定泛化。
3. 最终集成并非全部是验证优胜者，Ridge 回退、种子集成与测试反馈详见 final_selection.json 和原报告。
4. 原第三轮报告中“验证到天花板”等说法是实验者解释，有限试验不能证明后续调整必然无效。原报告保留作过程记录。
5. 本次仅整理、补齐依赖和复核，不重新选择模型，也不将附件内的后续建议视为额外执行指令。
