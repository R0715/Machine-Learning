"""日频、仅做多、含小组约定股票税费、可持有小数指数单位的模拟回测。

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


def transaction_fees(notional: float, side: str, config: Config) -> dict:
    """仅实际成交收费。保留浮点精度，输出时显示小数；不逐项取整到分。"""
    if side not in ("BUY", "SELL") or not np.isfinite(notional) or notional < 0:
        raise ValueError("成交方向或成交金额不合法。")
    commission = max(config.min_commission, notional * config.commission_rate) if notional > 0 else 0.0
    transfer = notional * config.transfer_fee_rate
    stamp = notional * config.stamp_tax_rate if side == "SELL" else 0.0
    return {"commission": commission, "transfer_fee": transfer, "stamp_tax": stamp,
            "total_fees": commission + transfer + stamp}


def rebalance_notional(cash: float, holding_value: float, weight: float, config: Config) -> float:
    """按扣费后的开盘资产实现目标权重，返回带符号成交金额。

    买入：n + weight*fee(n) = weight*assets - holding_value。
    卖出：n - weight*fee(n) = holding_value - weight*assets。
    分别解最低佣金和比例佣金两段，避免满仓时买光现金后再欠手续费。
    """
    gap = weight * (cash + holding_value) - holding_value
    if abs(gap) <= 1e-8:
        return 0.0
    if gap > 0:
        n = (gap - weight * config.min_commission) / (1 + weight * config.transfer_fee_rate)
        if n <= 1e-8:
            return 0.0  # 现金/调整量不足以支付最低佣金，不成交也不扣费。
        if n * config.commission_rate > config.min_commission:
            n = gap / (1 + weight * (config.commission_rate + config.transfer_fee_rate))
        return n
    rate = config.transfer_fee_rate + config.stamp_tax_rate
    n = (-gap + weight * config.min_commission) / (1 - weight * rate)
    if n * config.commission_rate > config.min_commission:
        n = -gap / (1 - weight * (rate + config.commission_rate))
    n = min(n, holding_value)
    # 极小账户若全部资产都不足以付最低费用，保留持仓并跳过成交。
    if n <= 1e-8 or transaction_fees(n, "SELL", config)["total_fees"] >= cash + n - 1e-8:
        return 0.0
    return -n


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
    rates = [config.commission_rate, config.transfer_fee_rate, config.stamp_tax_rate]
    if not all(np.isfinite(v) and v >= 0 for v in rates + [config.min_commission]) or sum(rates) >= 1:
        raise ValueError("税费参数必须有限且非负，比例费率之和须小于 1。")
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
        day_fees = 0.0
        opening, closing = prices.loc[date, ["open", "close"]]
        # 上一收盘信号，今日开盘成交；目标权重以扣除本笔费用后的资产计。
        if pending is not None:
            signed_notional = rebalance_notional(cash, units * opening, pending, config)
            delta = signed_notional / opening
            if delta < -1e-9 and last_buy is not None and date <= last_buy:
                raise ValueError("违反 T+1：当天新增持仓不能当天卖出。")
            if abs(signed_notional) > 1e-8:
                side = "BUY" if delta > 0 else "SELL"
                fees = transaction_fees(abs(signed_notional), side, config)
                day_fees = fees["total_fees"]
                cash -= signed_notional + day_fees
                units += delta
                if delta > 0:
                    last_buy = date
                if cash < -1e-7 or units < -1e-7:
                    raise AssertionError("现金或持仓出现负数。")
                cash = max(cash, 0.0)
                units = max(units, 0.0)
                trades.append({"signal_date": pending_date, "trade_date": date,
                               "side": side, "price": opening,
                               "units": abs(delta), "notional": abs(signed_notional), **fees,
                               "target_weight": pending, "cash_after": cash, "units_after": units,
                               "actual_weight_after": units * opening / (cash + units * opening)})
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
                        "drawdown": equity / peak - 1, "target_weight": target,
                        "fees_paid": day_fees})
    daily = pd.DataFrame(records).set_index("date")
    daily["cumulative_fees"] = daily.fees_paid.cumsum()
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
               "trade_count": len(trades), "return_periods": len(returns),
               "total_fees": float(daily.fees_paid.sum()),
               "total_commission": float(sum(t["commission"] for t in trades)),
               "total_transfer_fee": float(sum(t["transfer_fee"] for t in trades)),
               "total_stamp_tax": float(sum(t["stamp_tax"] for t in trades))}
    cols = ["signal_date", "trade_date", "side", "price", "units", "notional",
            "commission", "transfer_fee", "stamp_tax", "total_fees", "target_weight",
            "cash_after", "units_after", "actual_weight_after"]
    return BacktestResult(daily, pd.DataFrame(trades, columns=cols), metrics)


def benchmark_weights(prices: pd.DataFrame, config: Config) -> dict[str, pd.Series]:
    if not 0 < config.short_window < config.long_window:
        raise ValueError("均线周期必须满足 0 < short < long。")
    short = prices.close.rolling(config.short_window).mean()
    long = prices.close.rolling(config.long_window).mean()
    return {"buy_hold": pd.Series(1.0, index=prices.index),
            "dual_ma": (short > long).astype(float)}
