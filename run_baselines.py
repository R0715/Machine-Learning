"""训练创业板指的四个基准回归模型。"""

import argparse
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data import build_sequence, load_dataset
from src.training import fit_sequence, fit_tabular, verify_torch


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")
SEQUENCE_MODELS = ("tqnet", "patchtst")


def make_sequence_inputs(dataset, args, tasks):
    """用已确定的 LightGBM 列选择，另拟合时序模型自己的输入变换。"""
    columns = dataset.lightgbm["train"].columns.tolist()
    fit_end = min(dataset.labels[task]["train"].index[-1] for task in tasks)
    end = dataset.raw["test"].index[-1] if args.predict_test else dataset.raw["val"].index[-1]
    history = dataset.history.loc[:end, columns]
    fitting = history.loc[:fit_end].dropna()
    low, high = fitting.quantile(0.01), fitting.quantile(0.99)
    clipped = fitting.clip(lower=low, upper=high, axis=1)
    mean = clipped.mean()
    scale = clipped.std(ddof=0).replace(0, 1.0)
    inputs = (history.clip(lower=low, upper=high, axis=1) - mean) / scale
    normalization = {
        "features": columns,
        "rows": len(fitting),
        "fit_start": fitting.index[0].strftime("%Y-%m-%d"),
        "fit_end": fit_end.strftime("%Y-%m-%d"),
        "clip_lower": low.to_dict(),
        "clip_upper": high.to_dict(),
        "mean": mean.to_dict(),
        "scale": scale.to_dict(),
    }
    return inputs, normalization


def prediction_csv(path, dates, prediction, truth=None):
    if len(prediction) != len(dates) or not np.isfinite(prediction).all():
        raise ValueError(f"{path.name} 的预测长度或数值不合法。")
    table = pd.DataFrame({"y_pred_ret": prediction}, index=dates.rename("date"))
    if truth is not None:
        pd.testing.assert_index_equal(dates, truth.index, check_names=False)
        table.insert(0, "y_true_ret", truth.to_numpy())
    table.to_csv(path, encoding="utf-8-sig", date_format="%Y-%m-%d")


def run(args):
    dataset = load_dataset(args.dataset)
    tasks = ("close", "trade") if args.task == "both" else (args.task,)
    # 分开启动价格、交易任务时，仍以本包共同训练截止日拟合序列变换。
    inputs, normalization = make_sequence_inputs(dataset, args, ("close", "trade"))
    device = verify_torch(args.device, args.cpu_threads) if any(
        name in SEQUENCE_MODELS for name in args.models
    ) else None
    output = args.output or Path("runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    lookbacks = {"tqnet": args.lookback, "patchtst": args.patchtst_lookback}
    alignment_lookback = max(lookbacks.values())
    if "tqnet" in args.models:
        from src.cycle_selection import select_cycle

        acf_dates = build_sequence(
            inputs, dataset.labels["trade"]["train"].index, alignment_lookback,
        )[2]
        acf_input = dataset.raw["train"].loc[acf_dates, normalization["features"]]
        args.tqnet_cycle = select_cycle(acf_input, output / "acf")
        print(f"训练集自相关分析确定 TQNet 周期 W={args.tqnet_cycle}", flush=True)
    metadata = {
        "dataset_version": dataset.version,
        "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest(),
        "models": args.models,
        "tasks": tasks,
        "lookback": args.lookback,
        "sequence_lookbacks": lookbacks,
        "training_alignment_lookback": alignment_lookback,
        "tqnet_cycle": getattr(args, "tqnet_cycle", None),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "tqnet_max_epochs": args.tqnet_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "cpu_threads": args.cpu_threads,
        "test_predictions_requested": args.predict_test,
        "sequence_device": str(device) if device is not None else None,
        "sequence_normalization": normalization,
        "versions": {name: version(name) for name in ("numpy", "pandas")},
    }
    if device is not None:
        import torch

        metadata["versions"]["torch"] = torch.__version__
        metadata["cuda_runtime"] = torch.version.cuda
        if device.type == "cuda":
            metadata["gpu"] = torch.cuda.get_device_name(device)
    if "ridge" in args.models:
        metadata["versions"]["scikit-learn"] = version("scikit-learn")
    if "lightgbm" in args.models:
        metadata["versions"]["lightgbm"] = version("lightgbm")
    (output / "run_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    test_dates = None
    if args.predict_test:
        test_dates = dataset.raw["test"].index
        dataset.target_dates.to_csv(output / "test_target_dates.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")

    metrics = []
    for task in tasks:
        labels = dataset.labels[task]
        aligned = {
            split: build_sequence(inputs, labels[split].index, alignment_lookback)
            for split in ("train", "val")
        }
        # 四种模型使用同一训练和验证日期，去掉时序窗口开头的预热差异。
        train_dates, val_dates = aligned["train"][2], aligned["val"][2]
        if not len(train_dates) or not len(val_dates):
            raise ValueError(f"{task} 缺少完整历史窗口，请检查回看长度。")
        pd.testing.assert_index_equal(val_dates, labels["val"].index, check_names=False)
        y_train, y_val = labels["train"].loc[train_dates], labels["val"].loc[val_dates]
        print(f"[{task}] train={len(train_dates)}, val={len(val_dates)}, "
              f"PatchTST 窗口={args.patchtst_lookback}, TQNet 窗口={args.lookback}", flush=True)

        for name in args.models:
            folder = output / task / name
            folder.mkdir(parents=True)
            if name in SEQUENCE_MODELS:
                sequence = {
                    "train": build_sequence(inputs, train_dates, lookbacks[name]),
                    "val": build_sequence(inputs, val_dates, lookbacks[name]),
                }
                if test_dates is not None:
                    sequence["test"] = build_sequence(inputs, test_dates, lookbacks[name])
                result = fit_sequence(name, sequence, y_train, y_val, folder, device, args)
            else:
                tables = dataset.ridge if name == "ridge" else dataset.lightgbm
                x_test = tables["test"].loc[test_dates] if test_dates is not None else None
                result = fit_tabular(
                    name, tables["train"].loc[train_dates], y_train,
                    tables["val"].loc[val_dates], y_val, x_test, folder, args.seed,
                )
            prediction_csv(folder / "validation_predictions.csv", val_dates, result.validation, y_val)
            if result.test is not None:
                prediction_csv(folder / "predictions.csv", test_dates, result.test)
            errors = y_val.to_numpy() - result.validation
            row = {
                "task": task, "model": name, "device": str(device) if name in SEQUENCE_MODELS else "cpu",
                "train_rows": len(y_train), "val_rows": len(y_val),
                "val_rmse": np.sqrt(np.mean(errors ** 2)), "val_mae": np.mean(np.abs(errors)),
                "val_direction_accuracy": np.mean(np.sign(y_val.to_numpy()) == np.sign(result.validation)),
                "zero_return_val_rmse": np.sqrt(np.mean(y_val.to_numpy() ** 2)),
                "best_step": result.best_step, "fit_seconds": result.seconds, "peak_gpu_mib": result.peak_gpu_mib,
            }
            metrics.append(row)
            print(f"  {name}: RMSE={row['val_rmse']:.6f}, DA={row['val_direction_accuracy']:.2%}, {result.seconds:.2f}s", flush=True)
    pd.DataFrame(metrics).to_csv(output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"输出目录 {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="第二阶段统一建模数据修订版 ZIP")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--task", choices=("close", "trade", "both"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="时序模型设备")
    parser.add_argument("--lookback", type=int, choices=(32, 64), default=64)
    parser.add_argument("--patchtst-lookback", type=int, choices=(336,), default=336)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tqnet-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="运行输出目录，须是新目录")
    parser.add_argument("--predict-test", action="store_true", help="同时输出测试预测")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("--models 中的模型不能重复。")
    if min(args.batch_size, args.epochs, args.tqnet_epochs, args.patience, args.cpu_threads) < 1:
        parser.error("batch-size、epochs、patience 和 cpu-threads 必须为正。")
    run(args)


if __name__ == "__main__":
    main()
