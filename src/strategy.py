"""按收益、方差风险和交易成本选择分档仓位。"""

from dataclasses import dataclass

import numpy as np

from backtest_baselines import Fees


@dataclass(frozen=True)
class StrategyConfig:
    risk_aversion: float = 3.0
    risk_window: int = 20
    cost_aversion: float = 1.0
    drawdown_stop: float = 0.10
    cooldown_sessions: int = 5


class CostAwareStrategy:
    """Boyd 等人单期优化目标的单资产、分档仓位适配。"""

    def __init__(self, config=StrategyConfig(), fees=Fees()):
        if min(config.risk_aversion, config.cost_aversion) <= 0:
            raise ValueError("风险和成本系数须为正。")
        if config.risk_window < 2 or not 0 < config.drawdown_stop < 1 or config.cooldown_sessions < 1:
            raise ValueError("风险窗口、止损线和冷静期参数无效。")
        self.config = config
        self.fees = fees
        self.reference_peak = None
        self.cooldown_remaining = 0
        self.decisions = []

    def __call__(self, history, prediction, account):
        equity = float(account["equity"])
        current = float(account["current_weight"])
        if not np.isfinite(prediction) or not np.isfinite(equity) or equity <= 0:
            raise ValueError("预测须有限，账户权益须为有限正数。")
        self.reference_peak = max(self.reference_peak or equity, equity)
        drawdown = 1 - equity / self.reference_peak
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.reference_peak = equity
            weight, reason = 0.0, "cooldown"
            variance = np.nan
        elif drawdown >= self.config.drawdown_stop - 1e-12:
            self.cooldown_remaining = self.config.cooldown_sessions - 1
            if self.cooldown_remaining == 0:
                self.reference_peak = equity
            weight, reason = 0.0, "drawdown_stop"
            variance = np.nan
        else:
            known = history["open"].pct_change(fill_method=None).dropna()
            if len(known) < self.config.risk_window:
                raise ValueError("历史不足以估计开盘到开盘收益方差。")
            variance = float(known.iloc[-self.config.risk_window:].var(ddof=0))
            candidates = np.array(sorted({0.0, 0.25, 0.5, 0.75, 1.0, current}))
            turnover = candidates - current
            notional = np.abs(turnover) * equity
            commissions = np.where(
                notional > 1e-8,
                np.maximum(notional * self.fees.commission, self.fees.minimum_commission),
                0.0,
            )
            cost = (commissions + notional * self.fees.transfer
                    + np.maximum(-turnover, 0) * equity * self.fees.sell_stamp) / equity
            utility = (
                prediction * candidates
                - self.config.risk_aversion * variance * candidates ** 2
                - self.config.cost_aversion * cost
            )
            best = int(np.argmax(utility))
            current_utility = prediction * current - self.config.risk_aversion * variance * current ** 2
            if utility[best] <= current_utility + 1e-12:
                weight, reason = current, "hold"
            else:
                weight, reason = float(candidates[best]), "risk_cost_optimum"
        self.decisions.append({
            "date": account["date"], "prediction": prediction, "variance": variance,
            "current_weight": current, "target_weight": weight, "reason": reason,
            "stop_reference_drawdown": drawdown, "cooldown_remaining": self.cooldown_remaining,
        })
        return weight
