"""A 的基础 EDA 与回测检查图；组长可另写 evaluate.py 统一正式报告图。"""
from pathlib import Path
import json
import os
# 缓存放在项目工作目录，避免要求修改用户的全局 Matplotlib 配置目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "work" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_json(data: dict, path: Path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export_eda(prices: pd.DataFrame, cutoff: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    historical = prices.loc[:cutoff].copy()
    historical["daily_return"] = historical.close.pct_change()
    r = historical.daily_return.dropna()
    summary = (f"# 历史数据探索摘要\n\n分析只使用截至 {cutoff} 的行情，共 {len(historical)} 行。\n\n"
               f"- 历史起点：{historical.index[0].date()}。\n"
               f"- 收盘价范围：{historical.close.min():.2f} 至 {historical.close.max():.2f} 点。\n"
               f"- 日收盘收益率均值：{r.mean():.4%}；样本标准差：{r.std():.4%}。\n"
               f"- 最大单日上涨：{r.max():.4%}（{r.idxmax().date()}）。\n"
               f"- 最大单日下跌：{r.min():.4%}（{r.idxmin().date()}）。\n\n"
               "极端涨跌记录应核查但不能仅因幅度大就删除；价格水平、成交量的量级变化可以作为 B 构造相对收益、量比等特征的动机。\n\n"
               "这里是历史区间描述，不构成对未来可预测性的证明。验证期也包含在这份历史概览中，正式特征选择和缩放参数仍仅在训练子集拟合。成交量及成交额保留原文件单位，不额外猜测单位。\n")
    (output / "EDA说明.md").write_text(summary, encoding="utf-8")
    historical.describe().to_csv(output / "historical_summary.csv", encoding="utf-8-sig")
    historical.corr().to_csv(output / "historical_correlation.csv", encoding="utf-8-sig")
    historical["year"] = historical.index.year
    historical.groupby("year").agg(observations=("close", "size"),
                                    mean_daily_return=("daily_return", "mean"),
                                    daily_volatility=("daily_return", "std"),
                                    mean_volume=("volume", "mean")).to_csv(output / "historical_by_year.csv", encoding="utf-8-sig")
    historical.loc[historical.daily_return.abs().nlargest(10).index].sort_index().to_csv(
        output / "historical_extreme_returns.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(historical.index, historical.close, lw=0.8, color="#175b85")
    axes[0].set(title=f"Historical close (through {cutoff})", ylabel="Index points")
    axes[1].plot(historical.index, historical.volume, lw=0.6, color="#43877c")
    axes[1].set(ylabel="Volume (source units)")
    axes[2].hist(historical.daily_return.dropna(), bins=80, color="#175b85")
    axes[2].set(xlabel="Daily close-to-close return", ylabel="Count")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "historical_eda.png", dpi=180)
    plt.close(fig)


def export_result(name, result, output):
    result.equity.to_csv(output / "tables" / f"{name}_equity.csv", encoding="utf-8-sig")
    result.trades.to_csv(output / "tables" / f"{name}_trades.csv", index=False, encoding="utf-8-sig")


def export_comparison(results: dict, output: Path):
    metrics = pd.DataFrame({name: result.metrics for name, result in results.items()}).T
    metrics["excess_return_vs_buy_hold"] = metrics.total_return - metrics.loc["buy_hold", "total_return"]
    metrics.index.name = "strategy"
    metrics.to_csv(output / "tables" / "strategy_metrics.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for name, result in results.items():
        axes[0].plot(result.equity.index, result.equity.nav, label=name, lw=1.2)
        axes[1].plot(result.equity.index, result.equity.drawdown * 100, label=name, lw=1.0)
    axes[0].set(title="Strategy comparison (next-open execution)", ylabel="NAV")
    axes[1].set(ylabel="Drawdown (%)", xlabel="Date")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / "figures" / "strategy_comparison.png", dpi=180)
    plt.close(fig)

