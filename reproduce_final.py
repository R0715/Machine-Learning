"""从最终检查点复现预测和冻结策略回测；输出到新目录，不重训。"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.models_baselines import predict_saved


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "runs/20261119_233827_annualret_final_cd62b2"


def infer_group(args, group):
    names = ("ridge", "lightgbm") if group == "cpu" else ("tqnet", "patchtst")
    if (args.output / group).exists():
        raise ValueError(f"输出分组已存在，请使用新目录：{args.output / group}")
    device = None
    if group == "gpu":
        from src.training import verify_torch
        device = verify_torch(args.device, args.cpu_threads)
    dataset = load_dataset(args.dataset)
    checks = []
    # 各组只复制需要的最终权重与配置，不复制历史搜索。
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.source / group, args.output / group)
    for task in ("close", "trade"):
        for name in names:
            folder = args.source / group / task / name
            for split, filename in [("val", "validation_predictions.csv"), ("test", "predictions.csv")]:
                expected = pd.read_csv(folder / filename, index_col="date", parse_dates=["date"],
                                       encoding="utf-8-sig", float_precision="round_trip")
                values = predict_saved(name, task, args.source, args.dataset, split, expected.index, device)
                np.testing.assert_allclose(values, expected.y_pred_ret, rtol=1e-5, atol=1e-8)
                checks.append({"task": task, "model": name, "split": split, "rows": len(values),
                               "max_abs_prediction_error": float(np.max(abs(values-expected.y_pred_ret)))})
                if split == "test":
                    pd.DataFrame({"y_pred_ret": values}, index=expected.index).to_csv(
                        args.output / group / task / name / filename, encoding="utf-8-sig", date_format="%Y-%m-%d")
            print(f"Verified {group}/{task}/{name} (all ensemble members included)", flush=True)
    (args.output / f"{group}_reproduction_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")


def combine(args):
    from backtest_research import run as backtest
    from evaluate_test import evaluate
    from export_predictions import combine_run

    combine_run(args.output, args.dataset, args.prices)
    evaluate(args.output, args.prices)
    backtest(args.output, args.prices, args.output / "research_strategy_backtest", args.strategy)
    actual = pd.read_csv(args.output / "research_strategy_backtest/strategy_metrics.csv").set_index("model")
    expected = pd.read_csv(args.source / "research_strategy_backtest/strategy_metrics.csv").set_index("model")
    for col in ("final_equity", "profit", "total_fees"):
        np.testing.assert_allclose(actual.loc[expected.index, col], expected[col], rtol=1e-8, atol=0.01)
    (args.output / "REPRODUCTION_PASSED.json").write_text(json.dumps({
        "source_run": args.source.name, "status": "passed", "retrained": False,
        "predictions_checked": "validation and test; single models and full ensembles",
        "account_tolerance_yuan": 0.01, "test_status": "historical development backtest, previously inspected",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All four final models and account results reproduced successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "gpu", "combine", "all"), default="all")
    parser.add_argument("--source", type=Path, default=FINAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/第二阶段_统一建模数据_修订版.zip")
    parser.add_argument("--prices", type=Path, default=ROOT / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=ROOT / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("复现输出不能是源运行目录。")
    if args.stage in ("cpu", "all"):
        infer_group(args, "cpu")
    if args.stage in ("gpu", "all"):
        infer_group(args, "gpu")
    if args.stage in ("combine", "all"):
        combine(args)


if __name__ == "__main__":
    main()
