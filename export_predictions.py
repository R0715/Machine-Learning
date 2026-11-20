"""从现有检查点导出测试预测，或将两个任务合并为组内十列格式。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract_output import load_prices, write_contract
from src.data import build_sequence, load_dataset
from src.training import verify_torch


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")


def read_predictions(path):
    return pd.read_csv(
        path, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
        float_precision="round_trip",
    )


def saved_config(component, dataset_path):
    config = json.loads((component / "run_config.json").read_text(encoding="utf-8"))
    if config["dataset_sha256"] != sha256(dataset_path.read_bytes()).hexdigest():
        raise ValueError(f"{component} 与当前数据包的 SHA256 不一致。")
    return config


def restore_predict(name, folder, config, dataset, split, dates, device, model_file=None):
    """使用保存的权重和预处理参数，仅做前向预测。"""
    if name == "ridge":
        from joblib import load

        model = load(model_file or folder / "model.joblib")
        return model.predict(dataset.ridge[split].loc[dates])
    if name == "lightgbm":
        from lightgbm import Booster

        model = Booster(model_str=Path(model_file or folder / "model.txt").read_text(encoding="utf-8"))
        return model.predict(dataset.lightgbm[split].loc[dates])

    import torch
    if name == "tqnet":
        from src.tqnet_adapter import build_tqnet as build_model
    else:
        from src.patchtst_adapter import build_patchtst as build_model
    checkpoint = torch.load(model_file or folder / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("tuning_model"):
        from src.tuning_data import apply_inputs

        normalized = apply_inputs(dataset.history.loc[:dates[-1]], checkpoint["input_recipe"])
    else:
        recipe = config["sequence_normalization"]
        history = dataset.history.loc[:dates[-1], recipe["features"]]
        clipped = history.clip(
            lower=pd.Series(recipe["clip_lower"]), upper=pd.Series(recipe["clip_upper"]), axis=1,
        )
        normalized = (clipped - pd.Series(recipe["mean"])) / pd.Series(recipe["scale"])
    windows, positions, available = build_sequence(normalized, dates, checkpoint["lookback"])
    pd.testing.assert_index_equal(available, dates, check_names=False)
    if checkpoint.get("mid_window_model"):
        from src.mid_models import build_model as build_mid_model

        spec = checkpoint["model_spec"]
        model = build_mid_model(
            name, checkpoint["num_features"], checkpoint["lookback"], spec["options"],
        ).to(device)
    elif checkpoint.get("short_window_model") or "model_spec" not in checkpoint:
        from src.short_models import build_model as build_short_model

        spec = checkpoint.get("model_spec", {})
        model = build_short_model(
            name, checkpoint["num_features"], spec.get("options", {}),
        ).to(device)
    elif checkpoint.get("tuning_model"):
        from src.tuned_models import build_model as build_tuned_model

        spec = checkpoint["model_spec"]
        model = build_tuned_model(
            name, checkpoint["num_features"], checkpoint["lookback"], spec["options"], spec.get("cycle", 1),
        ).to(device)
    elif name == "tqnet":
        cycle = checkpoint["model_spec"]["cycle"]
        model = build_model(checkpoint["num_features"], checkpoint["lookback"], cycle=cycle).to(device)
    else:
        model = build_model(checkpoint["num_features"], checkpoint["lookback"]).to(device)
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


def infer_component(component, dataset_path, device_name):
    from src.models_baselines import predict_saved

    config = saved_config(component, dataset_path)
    dataset = load_dataset(dataset_path)
    names = config["models"]
    device = verify_torch(device_name, config["cpu_threads"]) if any(
        name in ("tqnet", "patchtst") for name in names
    ) else None
    test_dates = dataset.raw["test"].index
    for task in config["tasks"]:
        for name in names:
            folder = component / task / name
            validation = read_predictions(folder / "validation_predictions.csv")
            restored = predict_saved(name, task, component.parent, dataset_path, "val", validation.index, device)
            np.testing.assert_allclose(restored, validation["y_pred_ret"], rtol=1e-5, atol=1e-8)
            values = predict_saved(name, task, component.parent, dataset_path, "test", test_dates, device)
            if not np.isfinite(values).all():
                raise ValueError(f"{name}/{task} 的测试预测含非有限值。")
            frame = pd.DataFrame({"y_pred_ret": values}, index=test_dates.rename("date"))
            path = folder / "predictions.csv"
            if path.exists():
                existing = read_predictions(path)
                pd.testing.assert_index_equal(existing.index, frame.index)
                np.testing.assert_allclose(existing["y_pred_ret"], values, rtol=1e-5, atol=1e-8)
            else:
                frame.to_csv(path, encoding="utf-8-sig", date_format="%Y-%m-%d")
            print(f"已从检查点导出 {task}/{name}，{len(frame)} 行", flush=True)
    target_path = component / "test_target_dates.csv"
    dataset.target_dates.to_csv(target_path, encoding="utf-8-sig", date_format="%Y-%m-%d")


def combine_run(run, dataset_path, prices_path):
    dataset = load_dataset(dataset_path)
    prices = load_prices(prices_path, dataset_path)
    contracts = run / "handoff"
    contracts.mkdir(parents=True, exist_ok=True)
    expected_dates = dataset.raw["test"].index
    # 验证集用于模型选择和早停，信息截止日须包含验证标签的实现日期。
    train_asof = pd.Timestamp(dataset.target_dates.index[0])
    checksums, source_models = {}, {}
    for name in MODELS:
        component = run / ("cpu" if name in ("ridge", "lightgbm") else "gpu")
        config = saved_config(component, dataset_path)
        if set(config["tasks"]) != {"close", "trade"} or name not in config["models"]:
            raise ValueError(f"{component} 缺少 {name} 的双任务结果。")
        forecasts = {}
        for task in ("close", "trade"):
            prediction_path = component / task / name / "predictions.csv"
            forecasts[task] = read_predictions(prediction_path)["y_pred_ret"]
            pd.testing.assert_index_equal(forecasts[task].index, expected_dates)
            suffix = {"ridge": "model.joblib", "lightgbm": "model.txt"}.get(name, "model.pt")
            model_path = component / task / name / suffix
            source_models[f"{task}/{name}"] = sha256(model_path.read_bytes()).hexdigest()
        output = contracts / f"{name}_predictions.csv"
        write_contract(
            output, forecasts["close"], forecasts["trade"], dataset.target_dates, prices,
            f"B_{name}_{run.name}", train_asof,
        )
        checksums[output.name] = sha256(output.read_bytes()).hexdigest()
        print(f"已生成 {output.name}，{len(expected_dates)} 行 × 10 列", flush=True)
    metadata = {
        "source_run": str(run.resolve()),
        "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
        "prices_source": str(prices_path.resolve()),
        "prices_sha256": sha256(prices_path.read_bytes()).hexdigest(),
        "train_asof_date": train_asof.strftime("%Y-%m-%d"),
        "train_asof_reason": "验证集参与模型选择和早停，验证标签可用截至 2025-10-10 收盘。",
        "last_trade_target_date": None,
        "last_trade_target_note": "2026-10-22 信号的 t+2 超出本包行情截止日，交易收益结束日期留空。",
        "source_models_sha256": source_models,
        "files_sha256": checksums,
    }
    note = (
        "# 四模型双任务预测交付\n\n"
        f"来源运行为 {run.name}。每个模型的 CSV 均为 251 行、组内约定的 10 列，"
        "信号日范围为 2025-10-10 至 2026-10-22。\n\n"
        "pred_close_ret 对应下一交易日收盘相对当日收盘的收益预测，"
        "pred_close 用信号日 close × (1 + pred_close_ret) 换算。"
        "pred_trade_ret 对应下一交易日开盘到再下一交易日开盘的未扣费收益预测。\n\n"
        "train_asof_date 为 2025-10-10，包含用于选参和早停的验证标签信息截止日。"
        "模型与预处理来自上述运行。使用 export_run.ps1 补导出时，"
        "先加载检查点复现验证预测，核对一致后生成测试预测。\n\n"
        "最后一个信号日 2026-10-22 的交易收益结束日超出清洗行情截止日，"
        "return_end_date 留空。A 评价交易收益时使用前 250 条有目标日期的信号，"
        "251 条价格预测均有目标日期。最终仓位和期末处理由统一回测执行。\n"
    )
    note_path = contracts / "交付说明.md"
    note_path.write_text(note, encoding="utf-8")
    checksums[note_path.name] = sha256(note_path.read_bytes()).hexdigest()
    (contracts / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--component", type=Path, required=True)
    infer.add_argument("--dataset", type=Path, required=True)
    infer.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    combine = commands.add_parser("combine")
    combine.add_argument("--run", type=Path, required=True)
    combine.add_argument("--dataset", type=Path, required=True)
    combine.add_argument("--prices", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "infer":
        infer_component(args.component, args.dataset, args.device)
    else:
        combine_run(args.run, args.dataset, args.prices)


if __name__ == "__main__":
    main()
