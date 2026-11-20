"""组装年化收益调参的优胜模型，生成测试预测并用冻结策略回测验证。

只在选参冻结后运行一次：测试集不参与选择。复用 export_predictions 的
检查点读回（先核对验证预测一致再生成测试预测）、combine_run 的十列交付、
evaluate_test 的测试指标与 backtest_research 的冻结策略回测。
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from backtest_research import run as run_backtest_research
from evaluate_test import evaluate as evaluate_test
from export_predictions import combine_run, restore_predict, saved_config
from src.data import load_dataset


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")
MODEL_FILES = {"ridge": "model.joblib", "lightgbm": "model.txt"}


def component_of(name):
    return "cpu" if name in ("ridge", "lightgbm") else "gpu"


def assemble(source, output, dataset_path, prices_path, strategy_path, device_name):
    winners = json.loads((output / "selected_models.json").read_text(encoding="utf-8"))
    dataset = load_dataset(dataset_path)
    for group in ("cpu", "gpu"):
        shutil.copytree(source / group, output / group)
    for name, winner in winners.items():
        candidate = winner["candidate"]
        selected = output / "search" / name / candidate
        destination = output / component_of(name) / "trade" / name
        filename = MODEL_FILES.get(name, "model.pt")
        shutil.copy2(selected / filename, destination / filename)
        shutil.copy2(selected / "validation_predictions.csv",
                     destination / "validation_predictions.csv")
        history = selected / "training_history.csv"
        if history.exists():
            shutil.copy2(history, destination / "training_history.csv")
        (destination / "predictions.csv").unlink(missing_ok=True)
    for group in ("cpu", "gpu"):
        path = output / group / "run_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["tuning_source_run"] = str(source.resolve())
        config["tuned_trade_models"] = list(winners)
        config["tuning_objective"] = "validation annualized return under frozen strategy"
        config["tuning_device"] = device_name
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat([
        pd.read_csv(output / group / "validation_metrics.csv", encoding="utf-8-sig")
        for group in ("cpu", "gpu")
    ]).to_csv(output / "source_validation_metrics.csv", index=False, encoding="utf-8-sig")
    for group in ("cpu", "gpu"):
        path = output / group / "validation_metrics.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for name, winner in winners.items():
            if component_of(name) != group:
                continue
            prediction = pd.read_csv(
                output / group / "trade" / name / "validation_predictions.csv",
                index_col="date", parse_dates=["date"],
                encoding="utf-8-sig", float_precision="round_trip",
            )
            error = prediction["y_pred_ret"] - prediction["y_true_ret"]
            mask = frame.model.eq(name) & frame.task.eq("trade")
            frame.loc[mask, "val_rmse"] = np.sqrt(np.mean(error ** 2))
            frame.loc[mask, "val_mae"] = np.mean(np.abs(error))
            frame.loc[mask, "val_direction_accuracy"] = np.mean(
                np.sign(prediction["y_pred_ret"]) == np.sign(prediction["y_true_ret"]),
            )
            best = winner.get("best_epoch", winner.get("best_iteration"))
            if best is not None and not (isinstance(best, float) and np.isnan(best)):
                frame.loc[mask, "best_step"] = best
            frame.loc[mask, "fit_seconds"] = winner["seconds"]
            frame.loc[mask, "peak_gpu_mib"] = winner.get("peak_gpu_mib")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    pd.concat([
        pd.read_csv(output / group / "validation_metrics.csv", encoding="utf-8-sig")
        for group in ("cpu", "gpu")
    ]).to_csv(output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    print("已组装优胜模型到运行目录。", flush=True)
    return winners


def generate_test_predictions(output, dataset_path, device_name):
    import torch

    dataset = load_dataset(dataset_path)
    test_dates = dataset.raw["test"].index
    device = torch.device(device_name)
    for name in MODELS:
        group = component_of(name)
        component = output / group
        config = saved_config(component, dataset_path)
        folder = component / "trade" / name
        validation = pd.read_csv(
            folder / "validation_predictions.csv", encoding="utf-8-sig",
            index_col="date", parse_dates=["date"], float_precision="round_trip",
        )
        restored = restore_predict(name, folder, config, dataset, "val", validation.index, device)
        np.testing.assert_allclose(restored, validation["y_pred_ret"], rtol=1e-5, atol=1e-8)
        values = restore_predict(name, folder, config, dataset, "test", test_dates, device)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}/trade 的测试预测含非有限值。")
        pd.DataFrame({"y_pred_ret": values}, index=test_dates.rename("date")).to_csv(
            folder / "predictions.csv", encoding="utf-8-sig", date_format="%Y-%m-%d",
        )
        print(f"已生成 {name}/trade 测试预测，{len(values)} 行", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prices", type=Path, default=root / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=root / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    dataset = args.dataset or next((root / "data").glob("*.zip"))
    assemble(args.source, args.output, dataset, args.prices, args.strategy, args.device)
    generate_test_predictions(args.output, dataset, args.device)
    combine_run(args.output, dataset, args.prices)
    evaluate_test(args.output, args.prices)
    run_backtest_research(args.output, args.prices, args.output / "research_strategy_backtest", args.strategy)
    (root / "latest_annualret_run.txt").write_text(str(args.output.resolve()), encoding="utf-8")
    print(f"验证完成，输出 {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
