"""只用历史训练与分段验证自动调整两个时序模型的交易收益预测。"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data import build_sequence, load_dataset
from src.strategy import CostAwareStrategy, StrategyConfig
from src.training import verify_torch
from src.tuned_models import build_model
from src.tuning_data import input_columns, prepare_inputs, split_validation
from src.weighted_backtest import run_backtest


CANDIDATES = (
    dict(id="reference", inputs="full16", lr=3e-4, dropout=0.1, head_dropout=0.0,
         patch_len=8, stride=4, decay=0.0, patch_decay=1e-3, loss="mse", direction=0.0),
    dict(id="slow_regularized", inputs="full16", lr=1e-4, dropout=0.3, head_dropout=0.1,
         patch_len=16, stride=8, decay=1e-3, patch_decay=1e-2, loss="mse", direction=0.0),
    dict(id="reduced_features", inputs="reduced13", lr=1e-4, dropout=0.3, head_dropout=0.1,
         patch_len=16, stride=8, decay=1e-3, patch_decay=1e-2, loss="mse", direction=0.0),
    dict(id="robust_direction", inputs="reduced13", lr=1e-4, dropout=0.3, head_dropout=0.1,
         patch_len=16, stride=8, decay=1e-3, patch_decay=1e-2, loss="huber", direction=0.2),
    dict(id="strong_regularization", inputs="full16", lr=5e-5, dropout=0.4, head_dropout=0.2,
         patch_len=16, stride=8, decay=1e-2, patch_decay=1e-2, loss="huber", direction=0.2),
)


def metrics(truth, prediction):
    truth = np.asarray(truth)
    positive, negative = truth > 0, truth < 0
    if not positive.any() or not negative.any():
        raise ValueError("方向评分段需要同时包含上涨和下跌样本。")
    return {
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(truth) == np.sign(prediction))),
        "balanced_direction_accuracy": float(
            0.5 * (np.mean(prediction[positive] > 0) + np.mean(prediction[negative] < 0))
        ),
    }


def fit_candidate(name, candidate, arrays, labels, recipe, folder, cycle, device, seed, builder=build_model):
    import torch
    from torch import nn

    start = perf_counter()
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    options = {key: candidate[key] for key in ("dropout", "head_dropout", "patch_len", "stride")}
    for key in ("encoder_layers", "cycle"):
        if key in candidate:
            options[key] = candidate[key]
    lookback = arrays["train"][0].shape[1]
    model = builder(name, arrays["train"][0].shape[2], lookback, options, cycle).to(device)
    decay = candidate["decay"] if name == "tqnet" else candidate["patch_decay"]
    optimizer_name = candidate.get("optimizer", "Adam" if name == "tqnet" else "AdamW")
    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=candidate["lr"], weight_decay=decay)
    use_scheduler = candidate.get("use_scheduler", candidate["id"] != "reference")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-6,
    ) if use_scheduler else None
    max_epochs = candidate.get("epochs", 30 if name == "tqnet" else 20)
    train_y = labels["train"]
    mean, scale = float(train_y.mean()), max(float(train_y.std(ddof=0)), 1e-8)
    x = {
        split: (
            torch.as_tensor(values, device=device, dtype=torch.float32),
            torch.as_tensor(positions, device=device, dtype=torch.long),
        )
        for split, (values, positions, _) in arrays.items()
    }
    target = torch.as_tensor((train_y.to_numpy() - mean) / scale, device=device, dtype=torch.float32)
    raw_target = torch.as_tensor(train_y.to_numpy(), device=device, dtype=torch.float32)
    class_target = raw_target.gt(0).float()
    positive_weight = raw_target.lt(0).sum() / raw_target.gt(0).sum()
    regression = nn.MSELoss() if candidate["loss"] == "mse" else nn.SmoothL1Loss(beta=1.0)
    direction_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def predict(split):
        model.eval()
        values, positions = x[split]
        result = [
            model(values[i:i + 64], positions[i:i + 64]).cpu().numpy().astype(np.float64)
            for i in range(0, len(values), 64)
        ]
        return np.concatenate(result) * scale + mean

    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator, device=device)
        losses = []
        for rows in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            output = model(x["train"][0][rows], x["train"][1][rows])
            loss = regression(output, target[rows])
            nonzero = raw_target[rows].ne(0)
            if candidate["direction"] and nonzero.any():
                # 方向阈值对应原始收益零点，保留收益率回归输出。
                logits = output[nonzero] + mean / scale
                loss = loss + candidate["direction"] * direction_loss(logits, class_target[rows][nonzero])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
        stopped = predict("stopping")
        score = metrics(labels["stopping"], stopped)
        if not np.isfinite(score["rmse"]):
            raise RuntimeError(f"{name}/{candidate['id']} 出现无效验证误差。")
        if score["rmse"] < best:
            best, best_epoch = score["rmse"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({
            "epoch": epoch, "train_loss": torch.stack(losses).mean().item(),
            "stopping_rmse": score["rmse"],
            "stopping_balanced_direction_accuracy": score["balanced_direction_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"], "best_so_far": best_epoch == epoch,
        })
        pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False, encoding="utf-8-sig")
        if scheduler is not None:
            scheduler.step(score["rmse"])
        if epoch == 1 or epoch % 5 == 0:
            print(f"  {name}/{candidate['id']} epoch={epoch} stopping_rmse={score['rmse']:.6f}", flush=True)
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(best_state)
    # 选择段只在最佳早停检查点确定后评分，不参与梯度或早停。
    validation = predict("val")
    selection = pd.Series(validation, index=arrays["val"][2]).loc[labels["selection"].index].to_numpy()
    score = metrics(labels["selection"], selection)
    checkpoint = {
        "state_dict": best_state, "model": name, "num_features": arrays["train"][0].shape[2],
        "lookback": lookback, "model_spec": model.model_spec, "tuning_model": True,
        "input_recipe": recipe, "target_mean": mean, "target_std": scale, "best_epoch": best_epoch,
        "completed_epochs": epoch, "max_epochs": max_epochs, "candidate": candidate,
        "optimizer": {"name": optimizer_class.__name__, "learning_rate": candidate["lr"], "weight_decay": decay},
        "test_used_for_selection": False,
    }
    torch.save(checkpoint, folder / "model.pt")
    pd.DataFrame({
        "y_true_ret": labels["val"].to_numpy(), "y_pred_ret": validation,
    }, index=arrays["val"][2].rename("date")).to_csv(
        folder / "validation_predictions.csv", encoding="utf-8-sig", date_format="%Y-%m-%d",
    )
    score.update({
        "best_epoch": best_epoch, "completed_epochs": epoch, "seconds": perf_counter() - start,
        "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None,
    })
    return score, selection


def tune(source, output, dataset_path, prices_path, strategy_path, device_name):
    import torch

    dataset = load_dataset(dataset_path)
    source_config = json.loads((source / "gpu/run_config.json").read_text(encoding="utf-8"))
    digest = sha256(dataset_path.read_bytes()).hexdigest()
    if source_config["dataset_sha256"] != digest:
        raise ValueError("数据与原运行不一致。")
    output.mkdir(parents=True, exist_ok=False)
    labels = dataset.labels["trade"]
    stopping, selection = split_validation(labels["val"].index)
    known_end = dataset.history.index.to_series().shift(-2)
    stopping = stopping[known_end.loc[stopping].to_numpy() <= selection[0]]
    if not len(stopping) or not len(selection):
        raise ValueError("验证分段及边界剔除后没有样本。")
    # 所有候选用相同训练日期，与上一轮保持可比。
    baseline_features = source_config["sequence_normalization"]["features"]
    base = dataset.history[baseline_features]
    fit_dates = build_sequence(base, labels["train"].index, 336)[2]
    plan = {
        "models": ["tqnet", "patchtst"], "candidates": CANDIDATES,
        "training_rows": len(fit_dates), "training_end": fit_dates[-1].strftime("%Y-%m-%d"),
        "stopping_dates": [date.strftime("%Y-%m-%d") for date in stopping],
        "selection_dates": [date.strftime("%Y-%m-%d") for date in selection],
        "selection_criterion": "balanced direction accuracy, then frozen-policy net Sharpe, then lower RMSE",
        "label_boundary": "last stopping label endpoint must be no later than first selection signal",
        "test_already_seen_in_prior_experiments": True, "test_used_for_selection": False,
        "source_run": str(source.resolve()), "dataset_sha256": digest,
        "frozen_strategy_sha256": sha256(strategy_path.read_bytes()).hexdigest(),
    }
    (output / "search_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    device = verify_torch(device_name, 4)
    prices = pd.read_csv(
        prices_path, index_col="date", parse_dates=["date"], encoding="utf-8-sig", float_precision="round_trip",
    ).loc[:"2025-10-10"]
    strategy = StrategyConfig(**json.loads(strategy_path.read_text(encoding="utf-8"))["config"])
    results, winners = [], {}
    for name in ("tqnet", "patchtst"):
        for candidate in CANDIDATES:
            folder = output / "search" / name / candidate["id"]
            folder.mkdir(parents=True)
            columns = input_columns(dataset, candidate["inputs"])
            normalized, recipe = prepare_inputs(dataset, columns, fit_dates)
            lookback = 64 if name == "tqnet" else 336
            dates = {"train": fit_dates, "stopping": stopping, "val": labels["val"].index}
            arrays = {split: build_sequence(normalized, index, lookback) for split, index in dates.items()}
            y = {
                "train": labels["train"].loc[fit_dates], "stopping": labels["val"].loc[stopping],
                "val": labels["val"], "selection": labels["val"].loc[selection],
            }
            score, prediction = fit_candidate(
                name, candidate, arrays, y, recipe, folder, source_config["tqnet_cycle"], device, 42,
            )
            account, _, _ = run_backtest(
                prices, pd.Series(prediction, index=selection), selection[0], pd.Timestamp("2025-10-10"),
                CostAwareStrategy(strategy),
            )
            row = {
                "model": name, "candidate": candidate["id"], **score,
                "selection_net_sharpe": account["sharpe_rf0"], "selection_return": account["total_return"],
            }
            results.append(row)
            pd.DataFrame(results).to_csv(output / "tuning_trials.csv", index=False, encoding="utf-8-sig")
            print(f"{name}/{candidate['id']}: selection balancedDA={score['balanced_direction_accuracy']:.2%}, "
                  f"DA={score['direction_accuracy']:.2%}, Sharpe={account['sharpe_rf0']:.3f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        model_trials = pd.DataFrame([row for row in results if row["model"] == name]).sort_values(
            ["balanced_direction_accuracy", "selection_net_sharpe", "rmse", "candidate"],
            ascending=[False, False, True, True],
        )
        winners[name] = model_trials.iloc[0].to_dict()
    # 先落盘冻结选择，再复制对照和生成测试预测。
    (output / "selected_models.json").write_text(
        json.dumps(winners, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for component in ("cpu", "gpu"):
        shutil.copytree(source / component, output / component)
    for name, winner in winners.items():
        selected = output / "search" / name / winner["candidate"]
        destination = output / "gpu/trade" / name
        for filename in ("model.pt", "validation_predictions.csv", "training_history.csv"):
            shutil.copy2(selected / filename, destination / filename)
        (destination / "predictions.csv").unlink(missing_ok=True)
    for component in ("cpu", "gpu"):
        path = output / component / "run_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["tuning_source_run"] = str(source.resolve())
        config["tuned_trade_models"] = list(winners)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat([
        pd.read_csv(output / name / "validation_metrics.csv", encoding="utf-8-sig")
        for name in ("cpu", "gpu")
    ]).to_csv(output / "source_validation_metrics.csv", index=False, encoding="utf-8-sig")
    gpu_metrics = pd.read_csv(output / "gpu/validation_metrics.csv", encoding="utf-8-sig")
    for name, winner in winners.items():
        prediction = pd.read_csv(
            output / "gpu/trade" / name / "validation_predictions.csv",
            index_col="date", parse_dates=["date"], encoding="utf-8-sig", float_precision="round_trip",
        )
        error = prediction["y_pred_ret"] - prediction["y_true_ret"]
        mask = gpu_metrics.model.eq(name) & gpu_metrics.task.eq("trade")
        gpu_metrics.loc[mask, "val_rmse"] = np.sqrt(np.mean(error ** 2))
        gpu_metrics.loc[mask, "val_mae"] = np.mean(np.abs(error))
        gpu_metrics.loc[mask, "val_direction_accuracy"] = np.mean(
            np.sign(prediction["y_pred_ret"]) == np.sign(prediction["y_true_ret"]),
        )
        gpu_metrics.loc[mask, "best_step"] = winner["best_epoch"]
        gpu_metrics.loc[mask, "fit_seconds"] = winner["seconds"]
        gpu_metrics.loc[mask, "peak_gpu_mib"] = winner["peak_gpu_mib"]
    gpu_metrics.to_csv(output / "gpu/validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        pd.read_csv(output / "cpu/validation_metrics.csv", encoding="utf-8-sig"), gpu_metrics,
    ]).to_csv(output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"候选已冻结，输出 {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prices", type=Path, default=Path(__file__).parent / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=Path(__file__).parent / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = Path(__file__).parent
    dataset = args.dataset or next((root / "data").glob("*.zip"))
    output = args.output or root / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_tuned_{uuid4().hex[:6]}"
    tune(args.source, output, dataset, args.prices, args.strategy, args.device)


if __name__ == "__main__":
    main()
