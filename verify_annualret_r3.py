"""组装第二轮优胜配置（单模型/集成/多种子集成），生成测试预测并用冻结策略回测验证。

只在选参冻结后运行一次：测试集不参与选择。集成优胜者用成员检查点
分别读回测试预测再平均；128 日窗口检查点用中窗口模型读回。
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from backtest_research import run as run_backtest_research
from evaluate_test import evaluate as evaluate_test
from export_predictions import combine_run, restore_predict, saved_config
from src.data import build_sequence, load_dataset


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")


def component_of(name):
    return "cpu" if name in ("ridge", "lightgbm") else "gpu"


def restore_mid_predict(folder, config, dataset, split, dates, device):
    """读回 128 日窗口检查点（export_predictions 不识别该类型）。"""
    import torch
    from src.mid_models import build_model as build_mid
    from src.tuning_data import apply_inputs

    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    normalized = apply_inputs(dataset.history.loc[:dates[-1]], checkpoint["input_recipe"])
    windows, positions, available = build_sequence(normalized, dates, checkpoint["lookback"])
    pd.testing.assert_index_equal(available, dates, check_names=False)
    spec = checkpoint["model_spec"]
    model = build_mid(spec["name"], spec["num_features"], spec["lookback"], spec["options"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    batch = config["batch_size"]
    with torch.no_grad():
        for start in range(0, len(windows), batch):
            inputs = torch.as_tensor(windows[start:start + batch], dtype=torch.float32, device=device)
            phase = torch.as_tensor(positions[start:start + batch], dtype=torch.long, device=device)
            predictions.append(model(inputs, phase).cpu().numpy().astype(np.float64))
    return np.concatenate(predictions) * checkpoint["target_std"] + checkpoint["target_mean"]


def restore_any(name, folder, config, dataset, split, dates, device):
    if name in ("ridge", "lightgbm"):
        return restore_predict(name, folder, config, dataset, split, dates, device)
    import torch

    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("mid_window_model"):
        return restore_mid_predict(folder, config, dataset, split, dates, device)
    return restore_predict(name, folder, config, dataset, split, dates, device)


def assemble(source, output, dataset_path, winners, device_name):
    for group in ("cpu", "gpu"):
        shutil.copytree(source / group, output / group)
    for name, winner in winners.items():
        destination = output / component_of(name) / "trade" / name
        shutil.copy2(Path(winner["folder"]) / "validation_predictions.csv",
                     destination / "validation_predictions.csv")
        if winner["kind"] == "single":
            checkpoint = winner["checkpoints"][0]
            src_folder = Path(checkpoint["folder"])
            shutil.copy2(src_folder / checkpoint["file"], destination / checkpoint["file"])
            history = src_folder / "training_history.csv"
            if history.exists():
                shutil.copy2(history, destination / "training_history.csv")
        else:
            manifest = {"kind": winner["kind"], "candidate": winner["candidate"], "members": []}
            for index, checkpoint in enumerate(winner["checkpoints"]):
                src = Path(checkpoint["folder"]) / checkpoint["file"]
                dest_name = f"member_{index}{Path(checkpoint['file']).suffix}"
                shutil.copy2(src, destination / dest_name)
                manifest["members"].append({"file": dest_name,
                                            "source_folder": checkpoint["folder"]})
            (destination / "ensemble.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "predictions.csv").unlink(missing_ok=True)
    for group in ("cpu", "gpu"):
        path = output / group / "run_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["tuning_source_run"] = str(source.resolve())
        config["tuning_round"] = 3
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
            if winner.get("seconds") is not None:
                frame.loc[mask, "fit_seconds"] = winner["seconds"]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    pd.concat([
        pd.read_csv(output / group / "validation_metrics.csv", encoding="utf-8-sig")
        for group in ("cpu", "gpu")
    ]).to_csv(output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    print("已组装优胜模型到运行目录。", flush=True)


def generate_test_predictions(output, dataset_path, winners, device_name):
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
        winner = winners[name]
        if winner["kind"] == "single":
            restored = restore_any(name, folder, config, dataset, "val", validation.index, device)
            np.testing.assert_allclose(restored, validation["y_pred_ret"], rtol=1e-5, atol=1e-8)
            values = restore_any(name, folder, config, dataset, "test", test_dates, device)
        else:
            member_folders = [Path(item["folder"]) for item in winner["checkpoints"]]
            val_preds = [restore_any(name, member, config, dataset, "val", validation.index, device)
                         for member in member_folders]
            restored = np.mean(val_preds, axis=0)
            np.testing.assert_allclose(restored, validation["y_pred_ret"], rtol=1e-5, atol=1e-8)
            test_preds = [restore_any(name, member, config, dataset, "test", test_dates, device)
                          for member in member_folders]
            values = np.mean(test_preds, axis=0)
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
    winners = json.loads((args.output / "selected_models_r3.json").read_text(encoding="utf-8"))
    assemble(args.source, args.output, dataset, winners, args.device)
    generate_test_predictions(args.output, dataset, winners, args.device)
    combine_run(args.output, dataset, args.prices)
    evaluate_test(args.output, args.prices)
    run_backtest_research(args.output, args.prices, args.output / "research_strategy_backtest", args.strategy)
    (root / "latest_annualret3_run.txt").write_text(str(args.output.resolve()), encoding="utf-8")
    print(f"验证完成，输出 {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
