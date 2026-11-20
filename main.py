"""第一阶段入口：只运行数据清洗、历史 EDA 和两个传统基准。"""
from pathlib import Path
from dataclasses import asdict
import hashlib
from src.config import Config
from src.data_loader import load_data
from src.backtest import benchmark_weights, run_backtest
from src.a_reporting import save_json, export_eda, export_result, export_comparison

ROOT = Path(__file__).resolve().parent


def main():
    config = Config()
    source = ROOT / "data" / "SZ#399006(2).txt"
    output = ROOT / "outputs"
    for folder in (output / "tables", output / "figures"):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. 读取原始 TXT，校验并筛选到课程规定的结束日。
    prices, quality = load_data(source, config.end)
    prices.to_csv(output / "tables" / "clean_prices.csv", encoding="utf-8-sig")
    save_json(quality, output / "tables" / "data_quality.json")

    # 2. 探索分析只使用截至训练截止日的历史行情。
    export_eda(prices, config.cutoff, output / "eda")

    # 3. 生成买入持有、5/20日双均线信号，再用同一套账本回测。
    results = {}
    for name, weights in benchmark_weights(prices, config).items():
        result = run_backtest(prices, config, target_weights=weights)
        results[name] = result
        export_result(name, result, output)
        print(f"{name}: 期末资产 {result.metrics['final_equity']:.2f}，"
              f"累计收益 {result.metrics['total_return']:.2%}，"
              f"最大回撤 {result.metrics['max_drawdown']:.2%}，"
              f"累计税费 {result.metrics['total_fees']:.2f} 元")
    export_comparison(results, output)
    save_json({"stage": "第一阶段：数据、EDA、回测、传统基准；尚无正式模型",
               "config": asdict(config),
               "fee_model": "双向佣金万3最低5元；双向过户费十万分之一；卖出印花税万5",
               "execution": "收盘信号次日开盘成交，扣费后目标仓位，零滑点，可分割指数单位；期末按市值估值不强制卖出",
               "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
              output / "run_manifest.json")
    print(f"干净行情：{len(prices)} 行；结果位置：{output}")


if __name__ == "__main__":
    main()
