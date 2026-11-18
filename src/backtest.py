"""日频、仅做多、零手续费、可持有小数指数单位的模拟回测。

每天最多在开盘调仓一次；当天收盘才决定下个交易日的目标仓位。
不模拟盘中止损、限价单、涨跌停或真实 ETF 跟踪误差。
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from .config import Config


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def run_backtest(prices: pd.DataFrame, config: Config,
                 target_weights: pd.Series | None = None,
                 signal_fn: Callable | None = None) -> BacktestResult:
    """两种输入二选一。

    target_weights: 以信号日为索引的目标仓位，内部只推迟一次至次日开盘。
    signal_fn(history, account): 当天收盘后调用，仅传入截至当天的行情。
    account 包括 date/cash/units/equity/current_weight/peak_equity。
    回调不能读取闭包里预存的未来行情。
    """
    if (target_weights is None) == (signal_fn is None):
        raise ValueError("target_weights 和 signal_fn 必须且只能提供一种。")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("行情必须按日期升序且无重复。")
    if not np.isfinite(prices[["open", "close"]].to_numpy()).all() or (prices[["open", "close"]] <= 0).any().any():
        raise ValueError("开收盘价必须有限且大于零。")
    if not np.isfinite(config.initial_cash) or config.initial_cash <= 0:
        raise ValueError("初始资金必须大于零。")
    days = prices.loc[config.cutoff:config.end].index
    if len(days) < 2 or days[0] != pd.Timestamp(config.cutoff) or days[-1] != pd.Timestamp(config.end):
        raise ValueError("行情必须覆盖初始日和结束日。")
    if target_weights is not None:
        if target_weights.index.has_duplicates or not days[:-1].isin(target_weights.index).all():
            raise ValueError("仓位必须覆盖每一个信号日，且日期不能重复。")
        weights = target_weights.reindex(days[:-1]).astype(float)
        if not np.isfinite(weights).all() or not weights.between(0, 1).all():
            raise ValueError("目标仓位必须为 [0,1] 内有限数。")
    cash, units, peak = float(config.initial_cash), 0.0, float(config.initial_cash)
    pending, pending_date, last_buy = None, None, None
    records, trades = [], []
    for date in days:
        opening, closing = prices.loc[date, ["open", "close"]]
        # 上一信号日在收盘后确定权重；今日开盘按当前资产换算持仓。
        if pending is not None:
            assets_open = cash + units * opening
            desired_units = assets_open * pending / opening
            delta = desired_units - units
            if delta < -1e-9 and last_buy is not None and date <= last_buy:
                raise ValueError("违反 T+1：当天新增持仓不能当天卖出。")
            if abs(delta) > 1e-9:
                cash -= delta * opening
                units += delta
                if delta > 0:
                    last_buy = date
                if cash < -1e-7 or units < -1e-7:
                    raise AssertionError("现金或持仓出现负数。")
                cash = max(cash, 0.0)
                trades.append({"signal_date": pending_date, "trade_date": date,
                               "side": "BUY" if delta > 0 else "SELL", "price": opening,
                               "units": abs(delta), "notional": abs(delta) * opening,
                               "target_weight": pending, "cash_after": cash, "units_after": units})
        equity = cash + units * closing
        peak = max(peak, equity)
        account = {"date": date, "cash": cash, "units": units, "equity": equity,
                   "current_weight": units * closing / equity, "peak_equity": peak}
        target = np.nan
        if date != days[-1]:
            target = float(weights.loc[date] if target_weights is not None
                           else signal_fn(prices.loc[:date].copy(), account.copy()))
            if not np.isfinite(target) or not 0 <= target <= 1:
                raise ValueError(f"{date}: 回调输出仓位必须在 [0,1] 内。")
            pending, pending_date = target, date
        records.append({**account, "close": closing, "nav": equity / config.initial_cash,
                        "drawdown": equity / peak - 1, "target_weight": target})
    daily = pd.DataFrame(records).set_index("date")
    daily["daily_return"] = daily.equity.pct_change().fillna(0.0)
    returns = daily.daily_return.iloc[1:]
    total = daily.equity.iloc[-1] / config.initial_cash - 1
    annual = (1 + total) ** (252 / len(returns)) - 1
    max_dd = float(-daily.drawdown.min())  # 成绩表以正数表示最大亏损幅度。
    std = returns.std(ddof=1)
    metrics = {"initial_cash": config.initial_cash, "final_equity": float(daily.equity.iloc[-1]),
               "total_return": float(total), "annual_return": float(annual),
               "max_drawdown": max_dd,
               "sharpe_rf0": float(returns.mean() / std * np.sqrt(252)) if std > 0 else None,
               "calmar": float(annual / max_dd) if max_dd > 0 else None,
               "trade_count": len(trades), "return_periods": len(returns)}
    cols = ["signal_date", "trade_date", "side", "price", "units", "notional", "target_weight", "cash_after", "units_after"]
    return BacktestResult(daily, pd.DataFrame(trades, columns=cols), metrics)


def benchmark_weights(prices: pd.DataFrame, config: Config) -> dict[str, pd.Series]:
    if not 0 < config.short_window < config.long_window:
        raise ValueError("均线周期必须满足 0 < short < long。")
    short = prices.close.rolling(config.short_window).mean()
    long = prices.close.rolling(config.long_window).mean()
    return {"buy_hold": pd.Series(1.0, index=prices.index),
            "dual_ma": (short > long).astype(float)}
