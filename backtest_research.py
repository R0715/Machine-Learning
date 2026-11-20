"""使用冻结参数或验证期选参，运行文献目标驱动的分档策略。"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import pandas as pd

from src.strategy import CostAwareStrategy, StrategyConfig
from src.weighted_backtest import run_backtest


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")


def read_prediction(path):
    return pd.read_csv(
        path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
        float_precision="round_trip",
    )["y_pred_ret"]


def component(run, model):
    return run / ("cpu" if model in ("ridge", "lightgbm") else "gpu")


def select_on_validation(prices, run, output):
    """这一阶段只读取验证预测和截至验证截止日的历史行情。"""
    forecasts = {
        name: read_prediction(component(run, name) / "trade" / name / "validation_predictions.csv")
        for name in MODELS
    }
    historical = prices.loc[:"2025-10-10"]
    rows = []
    for window in (20, 60):
        for gamma in (1.0, 3.0, 10.0, 30.0):
            config = StrategyConfig(risk_aversion=gamma, risk_window=window)
            for name, prediction in forecasts.items():
                policy = CostAwareStrategy(config)
                metrics, _, _ = run_backtest(
                    historical, prediction, prediction.index[0], pd.Timestamp("2025-10-10"), policy,
                )
                rows.append({"risk_aversion": gamma, "risk_window": window, "model": name, **metrics})
    trials = pd.DataFrame(rows)
    trials.to_csv(output / "validation_trials.csv", index=False, encoding="utf-8-sig")
    scores = trials.groupby(["risk_aversion", "risk_window"], as_index=False).agg(
        mean_sharpe=("sharpe_rf0", "mean"), mean_return=("total_return", "mean"),
        mean_drawdown=("max_drawdown", "mean"),
    )
    scores = scores.sort_values(
        ["mean_sharpe", "mean_return", "mean_drawdown", "risk_aversion", "risk_window"],
        ascending=[False, False, True, True, True],
    )
    scores.to_csv(output / "validation_selection.csv", index=False, encoding="utf-8-sig")
    winner = scores.iloc[0]
    chosen = StrategyConfig(
        risk_aversion=float(winner.risk_aversion), risk_window=int(winner.risk_window),
    )
    record = {
        "config": asdict(chosen), "criterion": "mean validation Sharpe across four models",
        "validation_start": "2024-10-11", "validation_end": "2025-10-10",
        "missing_final_validation_signal": "hold existing units when no new prediction is available",
        "validation_predictions_sha256": {
            name: sha256((component(run, name) / "trade" / name / "validation_predictions.csv").read_bytes()).hexdigest()
            for name in MODELS
        },
        "sources": [
            "https://web.stanford.edu/~boyd/papers/cvx_portfolio.html",
            "https://www.cvxportfolio.com/en/stable/examples/paper_examples/single_period_opt.html",
        ],
        "project_adaptations": "0/25/50/75/100% target weights plus current hold, minimum commission, 10% drawdown stop and 5-signal cooldown",
        "test_status": "the same test period was inspected in preceding experiments; this is a retrospective extension",
    }
    (output / "selected_config.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return chosen


def load_frozen_config(path):
    """复用存档的完整参数，并记录其来源文件和校验值。"""
    payload = path.read_bytes()
    source_record = json.loads(payload)
    if not isinstance(source_record, dict):
        raise ValueError("冻结配置文件必须是 JSON 对象。")
    parameters = source_record.get("config")
    expected = set(StrategyConfig.__dataclass_fields__)
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("冻结配置必须包含完整的 StrategyConfig 参数。")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
        for value in parameters.values()
    ):
        raise ValueError("冻结配置的各项参数必须为有限数值。")
    if any(not isinstance(parameters[key], int) for key in ("risk_window", "cooldown_sessions")):
        raise ValueError("风险窗口和冷静期必须为整数。")
    config = StrategyConfig(**parameters)
    CostAwareStrategy(config)
    record = {
        "config": asdict(config),
        "selection_mode": "reuse_frozen_config",
        "frozen_config_source": str(path.resolve()),
        "frozen_config_sha256": sha256(payload).hexdigest(),
        "source_selection_record": source_record,
        "test_status": "the same test period was inspected in preceding experiments; this is a retrospective extension",
    }
    return config, record


def run(run_path, prices_path, output, config_path=None):
    prices = pd.read_csv(
        prices_path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
        float_precision="round_trip",
    )
    manifest = json.loads((run_path / "handoff/manifest.json").read_text(encoding="utf-8"))
    if sha256(prices_path.read_bytes()).hexdigest() != manifest["prices_sha256"]:
        raise ValueError("行情来源校验失败。")
    frozen = load_frozen_config(config_path) if config_path is not None else None
    output.mkdir(parents=True, exist_ok=False)
    if frozen is None:
        config = select_on_validation(prices, run_path, output)
        print(f"验证期选中公共配置 {asdict(config)}", flush=True)
    else:
        config, record = frozen
        (output / "selected_config.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"沿用已冻结公共配置 {asdict(config)}", flush=True)
    results = []
    for name in MODELS:
        path = run_path / "handoff" / f"{name}_predictions.csv"
        if sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"][path.name]:
            raise ValueError(f"{name} 测试预测来源校验失败。")
        prediction = pd.read_csv(
            path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
            float_precision="round_trip",
        )["pred_trade_ret"]
        policy = CostAwareStrategy(config)
        metrics, equity, trades = run_backtest(
            prices, prediction, pd.Timestamp("2025-10-10"), pd.Timestamp("2026-10-23"), policy,
        )
        equity.to_csv(output / f"{name}_equity.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
        trades.to_csv(output / f"{name}_trades.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        pd.DataFrame(policy.decisions).to_csv(
            output / f"{name}_decisions.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d",
        )
        results.append({"model": name, **metrics})
    summary = pd.DataFrame(results)
    summary.to_csv(output / "strategy_metrics.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path(__file__).parent / "data/clean_prices.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, help="复用存档策略配置，跳过验证期重新选参")
    args = parser.parse_args()
    run(args.run, args.prices, args.output or args.run / "research_strategy_backtest", args.config)


if __name__ == "__main__":
    main()
