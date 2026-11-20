"""用相同的固定仓位规则回测四种模型的交易预测。"""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fees:
    commission: float = 0.0003
    minimum_commission: float = 5.0
    transfer: float = 0.00001
    sell_stamp: float = 0.0005


def binary_backtest(prices, weights, fees=Fees(), initial_cash=100000.0):
    """收盘信号次日开盘执行，满仓或空仓，买入金额预留费用。"""
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("行情日期须唯一且升序。")
    if not np.isfinite(prices[["open", "close"]]).all().all() or (prices[["open", "close"]] <= 0).any().any():
        raise ValueError("开收盘价须为有限正数。")
    pd.testing.assert_index_equal(weights.index, prices.index[:-1], check_names=False)
    if not weights.isin([0.0, 1.0]).all():
        raise ValueError("本回测只接收 0 或 1 的目标仓位。")
    cash, units, peak = initial_cash, 0.0, initial_cash
    last_buy = None
    trades, records = [], []
    for position, (date, row) in enumerate(prices.iterrows()):
        if position:
            signal_date = prices.index[position - 1]
            target = weights.loc[signal_date]
            side = None
            if target == 1 and units == 0 and cash > fees.minimum_commission:
                # 先按比例佣金求满仓金额，再检查最低佣金分支。
                notional = cash / (1 + fees.commission + fees.transfer)
                if notional * fees.commission < fees.minimum_commission:
                    notional = (cash - fees.minimum_commission) / (1 + fees.transfer)
                side = "BUY"
                quantity = notional / row.open
                commission = max(notional * fees.commission, fees.minimum_commission)
                transfer, stamp = notional * fees.transfer, 0.0
                cash -= notional + commission + transfer
                units = quantity
                last_buy = date
            elif target == 0 and units > 0:
                if last_buy is not None and date <= last_buy:
                    raise AssertionError("卖出日须晚于买入日。")
                side = "SELL"
                quantity, notional = units, units * row.open
                commission = max(notional * fees.commission, fees.minimum_commission)
                transfer, stamp = notional * fees.transfer, notional * fees.sell_stamp
                cash += notional - commission - transfer - stamp
                units = 0.0
            if side is not None:
                if cash < -1e-7:
                    raise AssertionError("计费后现金出现负数。")
                cash = max(0.0, cash)
                trades.append({
                    "signal_date": signal_date, "trade_date": date, "side": side,
                    "price": row.open, "units": quantity, "notional": notional,
                    "commission": commission, "transfer_fee": transfer, "stamp_tax": stamp,
                    "total_fee": commission + transfer + stamp, "cash_after": cash, "units_after": units,
                })
        equity = cash + units * row.close
        peak = max(peak, equity)
        records.append({
            "date": date, "cash": cash, "units": units, "close": row.close, "equity": equity,
            "nav": equity / initial_cash, "drawdown": equity / peak - 1,
        })
    daily = pd.DataFrame(records).set_index("date")
    columns = [
        "signal_date", "trade_date", "side", "price", "units", "notional",
        "commission", "transfer_fee", "stamp_tax", "total_fee", "cash_after", "units_after",
    ]
    fills = pd.DataFrame(trades, columns=columns)
    metrics = {
        "initial_cash": initial_cash, "final_equity": float(daily.equity.iloc[-1]),
        "profit": float(daily.equity.iloc[-1] - initial_cash),
        "total_return": float(daily.equity.iloc[-1] / initial_cash - 1),
        "max_drawdown": float(-daily.drawdown.min()), "total_fees": float(fills.total_fee.sum()),
        "trade_count": len(fills), "ending_units": float(units),
    }
    return metrics, daily, fills


def run(run_path, prices_path, output):
    full = pd.read_csv(
        prices_path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
        float_precision="round_trip",
    )
    handoff = run_path / "handoff"
    source = json.loads((handoff / "manifest.json").read_text(encoding="utf-8"))
    if sha256(prices_path.read_bytes()).hexdigest() != source["prices_sha256"]:
        raise ValueError("行情与预测交付时的价格来源不一致。")
    days = full.loc["2025-10-10":"2026-10-23"]
    if days.index[0] != pd.Timestamp("2025-10-10") or days.index[-1] != pd.Timestamp("2026-10-23"):
        raise ValueError("行情须覆盖课程投资起止日期。")
    signals = {
        "buy_hold": pd.Series(1.0, index=days.index[:-1]),
        "dual_ma_5_20": (
            full.close.rolling(5).mean() > full.close.rolling(20).mean()
        ).astype(float).reindex(days.index[:-1]),
    }
    for name in ("ridge", "lightgbm", "tqnet", "patchtst"):
        path = handoff / f"{name}_predictions.csv"
        if sha256(path.read_bytes()).hexdigest() != source["files_sha256"][path.name]:
            raise ValueError(f"{name} 预测交付文件校验失败。")
        prediction = pd.read_csv(
            path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
            float_precision="round_trip",
        )["pred_trade_ret"]
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} 含无效预测。")
        signals[name] = prediction.gt(0).astype(float)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, weights in signals.items():
        metrics, equity, trades = binary_backtest(days, weights)
        rows.append({"model": name, **metrics})
        equity.to_csv(output / f"{name}_equity.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
        trades.to_csv(output / f"{name}_trades.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    result = pd.DataFrame(rows)
    result["excess_return_vs_buy_hold"] = result.total_return - result.loc[result.model.eq("buy_hold"), "total_return"].iloc[0]
    result.to_csv(output / "strategy_metrics.csv", index=False, encoding="utf-8-sig")
    config = {
        "rule": "pred_trade_ret > 0 => full long; otherwise cash",
        "rule_selection": "查看预测测试指标后补充的固定规则回测，阈值为零，未搜索阈值或风控参数。",
        "initial_cash": 100000.0, "start": "2025-10-10", "end": "2026-10-23",
        "execution": "signal at close, execute at following trading-day open",
        "units": "fractional index units", "slippage": 0.0,
        "ending": "mark to final close", "fees": asdict(Fees()),
        "prices_sha256": source["prices_sha256"], "predictions_manifest": source,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path(__file__).parent / "data/clean_prices.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.run, args.prices, args.output or args.run / "simple_strategy_backtest")


if __name__ == "__main__":
    main()
