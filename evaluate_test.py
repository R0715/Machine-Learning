"""对照真实行情评价已冻结模型的两项测试预测。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")


def evaluate(run, prices_path):
    prices = pd.read_csv(
        prices_path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
        float_precision="round_trip",
    )
    handoff = run / "handoff"
    manifest = json.loads((handoff / "manifest.json").read_text(encoding="utf-8"))
    if sha256(prices_path.read_bytes()).hexdigest() != manifest["prices_sha256"]:
        raise ValueError("行情文件与生成交付预测时的价格来源不一致。")
    for name, digest in manifest["files_sha256"].items():
        if sha256((handoff / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"交付文件哈希不一致：{name}")

    records = []
    expected = {}
    for model in MODELS:
        frame = pd.read_csv(
            handoff / f"{model}_predictions.csv", encoding="utf-8-sig", index_col="date",
            parse_dates=["date", "price_target_date", "return_start_date", "return_end_date"],
            float_precision="round_trip",
        )
        for task in ("close", "trade"):
            subset = frame if task == "close" else frame.loc[frame["return_end_date"].notna()]
            if task == "close":
                true_price = prices.loc[subset["price_target_date"], "close"].to_numpy()
                truth = true_price / subset["close"].to_numpy() - 1
                pred = subset["pred_close_ret"].to_numpy()
            else:
                opening = prices.loc[subset["return_start_date"], "open"].to_numpy()
                ending = prices.loc[subset["return_end_date"], "open"].to_numpy()
                truth = ending / opening - 1
                pred = subset["pred_trade_ret"].to_numpy()
            if task in expected:
                pd.testing.assert_index_equal(subset.index, expected[task])
            expected[task] = subset.index
            if not np.isfinite(truth).all() or not np.isfinite(pred).all():
                raise ValueError(f"{model}/{task} 包含无效答案或预测。")
            zero_rmse = float(np.sqrt(np.mean(truth ** 2)))
            rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
            correlation = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 0 and truth.std() > 0 else None
            positive, negative = truth > 0, truth < 0
            balanced_direction = (
                float(0.5 * (np.mean(pred[positive] > 0) + np.mean(pred[negative] < 0)))
                if positive.any() and negative.any() else None
            )
            record = {
                "task": task, "model": model, "rows": len(subset),
                "signal_start": subset.index[0].strftime("%Y-%m-%d"),
                "signal_end": subset.index[-1].strftime("%Y-%m-%d"),
                "rmse": rmse,
                "mae": float(np.mean(np.abs(truth - pred))),
                "direction_accuracy": float(np.mean(np.sign(pred) == np.sign(truth))),
                "balanced_direction_accuracy": balanced_direction,
                "pearson_ic": correlation,
                "zero_return_rmse": zero_rmse,
                "rmse_improvement_vs_zero": 1 - rmse / zero_rmse,
                "always_positive_direction_accuracy": float(np.mean(truth > 0)),
                "prediction_mean": float(pred.mean()),
                "prediction_std": float(pred.std()),
            }
            if task == "close":
                price_error = subset["pred_close"].to_numpy() - true_price
                record["price_rmse"] = float(np.sqrt(np.mean(price_error ** 2)))
                record["price_mae"] = float(np.mean(np.abs(price_error)))
                record["unchanged_price_rmse"] = float(np.sqrt(
                    np.mean((subset["close"].to_numpy() - true_price) ** 2),
                ))
            records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(run / "test_metrics.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path(__file__).parent / "data/clean_prices.csv")
    args = parser.parse_args()
    evaluate(args.run, args.prices)


if __name__ == "__main__":
    main()
