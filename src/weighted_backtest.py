"""日频分档仓位回测，信号在下一交易日开盘执行并计入组内税费。"""

import numpy as np
import pandas as pd

from backtest_baselines import Fees


TRADE_COLUMNS = [
    "signal_date", "trade_date", "side", "price", "units", "notional",
    "commission", "transfer_fee", "stamp_tax", "total_fee", "target_weight",
    "realized_weight", "cash_after", "units_after",
]


def _order(cash, units, opening, target, fees):
    """按成交后资产权重解出金额，同时满足最低佣金和现金约束。"""
    holding = units * opening
    equity = cash + holding
    tolerance = max(equity, 1.0) * 1e-12
    difference = target * equity - holding
    if abs(difference) <= tolerance:
        return None

    if difference > 0:
        side = "BUY"
        notional = difference / (1 + target * (fees.commission + fees.transfer))
        if notional * fees.commission < fees.minimum_commission:
            notional = (difference - target * fees.minimum_commission) / (1 + target * fees.transfer)
        stamp_rate = 0.0
    else:
        side = "SELL"
        sell_rate = fees.commission + fees.transfer + fees.sell_stamp
        notional = -difference / (1 - target * sell_rate)
        if notional * fees.commission < fees.minimum_commission:
            notional = (-difference + target * fees.minimum_commission) / (
                1 - target * (fees.transfer + fees.sell_stamp)
            )
        notional = min(notional, holding)
        stamp_rate = fees.sell_stamp
    if notional <= tolerance:
        return None

    commission = max(notional * fees.commission, fees.minimum_commission)
    transfer, stamp = notional * fees.transfer, notional * stamp_rate
    total_fee = commission + transfer + stamp
    if total_fee >= equity:
        return None
    direction = 1 if side == "BUY" else -1
    new_cash = cash - direction * notional - total_fee
    if new_cash < -tolerance:
        return None
    quantity = notional / opening
    new_units = units + direction * quantity
    if new_units < -1e-12:
        raise AssertionError("成交后持仓为负。")
    new_cash, new_units = max(0.0, new_cash), max(0.0, new_units)
    after_equity = new_cash + new_units * opening
    return {
        "side": side, "price": opening, "units": quantity, "notional": notional,
        "commission": commission, "transfer_fee": transfer, "stamp_tax": stamp,
        "total_fee": total_fee, "target_weight": target,
        "realized_weight": new_units * opening / after_equity,
        "cash_after": new_cash, "units_after": new_units,
    }


def run_backtest(
    prices, predictions, start, end, policy, initial_cash=100000.0, fees=Fees(),
):
    """回调在每日收盘运行，只收到截至当天的行情、预测和账户。

    predictions 应覆盖信号日期的连续前缀。验证段尾部没有预测时保持持仓，
    以便把最后一个可用预测区间持有到标签结束日。目标等于当前收盘权重时，
    视为持有现有单位，下一开盘不因隔夜权重漂移产生额外订单。
    """
    if (
        not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
        or not prices.index.is_unique or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("行情需使用唯一、升序且无缺失的交易日期索引。")
    if (
        not isinstance(predictions.index, pd.DatetimeIndex) or predictions.index.hasnans
        or not predictions.index.is_unique or not predictions.index.is_monotonic_increasing
    ):
        raise ValueError("预测需使用唯一、升序且无缺失的信号日期索引。")
    if not {"open", "close"}.issubset(prices.columns):
        raise ValueError("行情缺少 open 或 close 列。")
    values = prices[["open", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("开收盘价须为有限正数。")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("初始资金须为有限正数。")
    fee_values = np.asarray([
        fees.commission, fees.minimum_commission, fees.transfer, fees.sell_stamp,
    ], dtype=float)
    if (
        not np.isfinite(fee_values).all() or (fee_values < 0).any()
        or fees.commission + fees.transfer + fees.sell_stamp >= 1
    ):
        raise ValueError("费用须非负，合计比例费用须小于 1。")

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = prices.loc[start:end]
    if len(days) < 2 or days.index[0] != start or days.index[-1] != end:
        raise ValueError("行情需完整覆盖回测起止日，至少有两个交易日。")
    signal_days = days.index[:-1]
    if (
        predictions.empty or len(predictions) > len(signal_days)
        or not predictions.index.equals(signal_days[:len(predictions)])
        or not np.isfinite(predictions.to_numpy(dtype=float)).all()
    ):
        raise ValueError("预测应为回测信号日的非空连续前缀，且数值有限。")

    cash, units, peak = float(initial_cash), 0.0, float(initial_cash)
    pending, pending_date, pending_hold, last_buy = None, None, True, None
    records, trades = [], []
    for date, row in days.iterrows():
        execution = "initial" if pending is None else "hold"
        if pending is not None and not pending_hold:
            order = _order(cash, units, row.open, pending, fees)
            execution = "no_order" if order is None else "filled"
            if order is not None:
                if order["side"] == "SELL" and last_buy is not None and date <= last_buy:
                    raise AssertionError("卖出日期须晚于最近买入日期。")
                if order["side"] == "BUY":
                    last_buy = date
                cash, units = order["cash_after"], order["units_after"]
                trades.append({"signal_date": pending_date, "trade_date": date, **order})

        equity = cash + units * row.close
        if equity <= 0:
            raise AssertionError("账户资产须为正。")
        peak = max(peak, equity)
        account = {
            "date": date, "cash": cash, "units": units, "equity": equity,
            "current_weight": units * row.close / equity, "peak_equity": peak,
        }
        available = date in predictions.index and date != end
        target = np.nan
        hold = True
        if date != end:
            target = (
                float(policy(prices.loc[:date].copy(), float(predictions.loc[date]), account.copy()))
                if available else account["current_weight"]
            )
            if not np.isfinite(target) or not 0 <= target <= 1:
                raise ValueError(f"{date.date()} 的目标仓位须为 [0,1] 内有限值。")
            hold = abs(target - account["current_weight"]) <= 1e-12
            pending, pending_date, pending_hold = target, date, hold
        records.append({
            **account, "close": row.close, "nav": equity / initial_cash,
            "drawdown": equity / peak - 1, "target_weight": target,
            "prediction_available": available, "hold_units": hold, "execution": execution,
        })

    daily = pd.DataFrame(records).set_index("date")
    daily["daily_return"] = daily.equity.pct_change().fillna(0.0)
    fills = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    returns = daily.daily_return.iloc[1:]
    total = float(daily.equity.iloc[-1] / initial_cash - 1)
    annual = float((1 + total) ** (252 / len(returns)) - 1)
    max_drawdown = float(-daily.drawdown.min())
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    metrics = {
        "initial_cash": initial_cash, "final_equity": float(daily.equity.iloc[-1]),
        "profit": float(daily.equity.iloc[-1] - initial_cash), "total_return": total,
        "annual_return": annual, "max_drawdown": max_drawdown,
        "sharpe_rf0": sharpe, "sharpe": sharpe,
        "calmar": annual / max_drawdown if max_drawdown > 0 else None,
        "total_fees": float(fills.total_fee.sum()), "trade_count": len(fills),
        "ending_units": float(units), "return_periods": len(returns),
    }
    return metrics, daily, fills
