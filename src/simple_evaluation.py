"""用固定的正预测满仓规则评价收益，复用既有税费账本。"""

import numpy as np
import pandas as pd

from backtest_baselines import binary_backtest


def evaluate_predictions(prices, prediction, start, end, initial_cash=100000.0):
    """当日收盘预测为正则满仓，否则空仓，次日开盘成交。

    预测须覆盖回测信号日的非空连续前缀。若验证末尾没有新预测，
    沿用最后的目标状态并保持持仓单位。预测缺失信息另行标记，
    不生成新的收益预测。税费和期末市值计价沿用 binary_backtest。
    """
    if not isinstance(prediction, pd.Series):
        raise ValueError("预测须为以信号日为索引的 Series。")
    for dates, name in ((prices.index, "行情"), (prediction.index, "预测")):
        if (
            not isinstance(dates, pd.DatetimeIndex) or dates.hasnans
            or not dates.is_unique or not dates.is_monotonic_increasing
        ):
            raise ValueError(f"{name}须使用唯一、升序且没有缺失的交易日期索引。")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("初始资金须为有限正数。")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = prices.loc[start:end]
    if len(days) < 2 or days.index[0] != start or days.index[-1] != end:
        raise ValueError("行情须完整覆盖回测起止日，且至少有两个交易日。")
    signal_days = days.index[:-1]
    if (
        prediction.empty or len(prediction) > len(signal_days)
        or not prediction.index.equals(signal_days[:len(prediction)])
        or not np.isfinite(prediction.to_numpy(dtype=float)).all()
    ):
        raise ValueError("预测须覆盖信号日的非空连续前缀，且每个预测值有限。")

    targets = prediction.gt(0).astype(float).reindex(signal_days).ffill()
    metrics, daily, trades = binary_backtest(days, targets, initial_cash=initial_cash)
    daily["daily_return"] = daily["equity"].pct_change().fillna(0.0)
    daily["current_weight"] = daily["units"] * daily["close"] / daily["equity"]
    daily["target_weight"] = targets.reindex(daily.index)
    daily["prediction_available"] = daily.index.isin(prediction.index)
    realized_returns = daily["daily_return"].iloc[1:]
    volatility = float(realized_returns.std(ddof=1)) if len(realized_returns) > 1 else 0.0
    sharpe = (
        float(realized_returns.mean() / volatility * np.sqrt(252))
        if volatility > 0 else 0.0
    )
    traded_notional = float(trades["notional"].sum())
    metrics.update({
        "annual_return": float((1 + metrics["total_return"]) ** (252 / len(realized_returns)) - 1),
        "sharpe_rf0": sharpe,
        "sharpe": sharpe,
        "return_periods": len(realized_returns),
        "turnover_notional": traded_notional,
        "turnover_initial_equity": traded_notional / initial_cash,
        "average_exposure": float(daily["current_weight"].iloc[1:].mean()),
        "signals_provided": len(prediction),
        "signals_forward_filled": len(signal_days) - len(prediction),
    })
    return metrics, daily, trades
