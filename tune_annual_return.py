"""以验证期冻结策略年化收益为目标，自动调整四个交易收益模型的超参数。

目标：验证期（2024-10-11 至 2025-10-10，241 个交易信号）在已冻结
Boyd 收益-风险-成本策略下的年化收益率，含约定税费。
价格预测器不参与收益回测，作为固定对照保留。
早停与选参共用完整验证期（与 64 日探索轮相同口径，如实披露）。
选择规则：每模型按验证年化收益排序；优先选前后半段收益均不低于 -5%
且最大回撤不超过 25% 的候选；若无候选满足稳健条件，取年化收益最高者并标记。
测试集只在选参冻结后用于一次性验证，不参与任何选择。
"""

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
from src.tuned_models import build_model as build_tuned
from src.short_models import build_model as build_short
from src.tuning_data import input_columns, prepare_inputs
from src.weighted_backtest import run_backtest
from tune_trade_models import fit_candidate


VAL_END = pd.Timestamp("2025-10-10")
MAX_DRAWDOWN = 0.25
MIN_HALF_RETURN = -0.05


def short_factory(name, num_features, lookback, options, cycle):
    if lookback != 64:
        raise ValueError("64 日候选固定 64 日窗口。")
    return build_short(name, num_features, options)


def seq_candidate(cid, **overrides):
    base = dict(
        id=cid, lr=3e-4, dropout=0.1, head_dropout=0.0, patch_len=8, stride=4,
        decay=0.0, patch_decay=1e-3, loss="mse", direction=0.0,
        optimizer="AdamW", epochs=20, use_scheduler=False,
    )
    base.update(overrides)
    return base


# (candidate dict, lookback, builder, cycle)
PATCHTST_CANDIDATES = [
    (seq_candidate("ref336"), 336, "tuned", 1),
    (seq_candidate("336_lr1e4", lr=1e-4), 336, "tuned", 1),
    (seq_candidate("336_lr1e4_drop02", lr=1e-4, dropout=0.2, head_dropout=0.1), 336, "tuned", 1),
    (seq_candidate("ref64", encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_lr1e4", lr=1e-4, encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_lr5e4", lr=5e-4, encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_drop02", dropout=0.2, encoder_layers=2), 64, "short", 1),
]

TQNET_CANDIDATES = [
    (seq_candidate("ref", optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("cycle5", optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 5),
    (seq_candidate("cycle20", optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 20),
    (seq_candidate("lr1e4", lr=1e-4, optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lr5e4", lr=5e-4, optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("drop02", dropout=0.2, optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("cycle5_lr5e4", lr=5e-4, optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 5),
    (seq_candidate("huber_dir", loss="huber", direction=0.2, optimizer="Adam", decay=0.0, epochs=30),
     64, "tuned", 1),
    (seq_candidate("wd1e3", optimizer="Adam", decay=1e-3, epochs=30), 64, "tuned", 1),
    (seq_candidate("lb32_cycle5", optimizer="Adam", decay=0.0, epochs=30), 32, "tuned", 5),
]

LGBM_REF = dict(
    num_leaves=31, max_depth=4, learning_rate=0.05, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
)
LGBM_CANDIDATES = [("ref", dict(LGBM_REF))]
for label, delta in (
    ("leaves15", dict(num_leaves=15)),
    ("leaves63", dict(num_leaves=63)),
    ("lr003", dict(learning_rate=0.03)),
    ("lr01", dict(learning_rate=0.1)),
    ("minchild10", dict(min_child_samples=10)),
    ("minchild40", dict(min_child_samples=40)),
    ("lambda0", dict(reg_lambda=0.0)),
    ("lambda5", dict(reg_lambda=5.0)),
    ("colsample07", dict(colsample_bytree=0.7)),
    ("colsample10", dict(colsample_bytree=1.0)),
    ("leaves63_lr003", dict(num_leaves=63, learning_rate=0.03)),
    ("leaves15_minchild40", dict(num_leaves=15, min_child_samples=40)),
):
    params = dict(LGBM_REF)
    params.update(delta)
    LGBM_CANDIDATES.append((label, params))

RIDGE_ALPHAS = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]


def score_predictions(prices_val, prediction, frozen):
    """冻结策略在完整验证期及前后半段的回测指标。"""
    dates = prediction.index
    full, _, _ = run_backtest(
        prices_val, prediction, dates[0], VAL_END, CostAwareStrategy(frozen),
    )
    mid = len(dates) // 2
    first, second = dates[:mid], dates[mid:]
    half1, _, _ = run_backtest(
        prices_val, prediction.loc[first], first[0], second[0], CostAwareStrategy(frozen),
    )
    half2, _, _ = run_backtest(
        prices_val, prediction.loc[second], second[0], VAL_END, CostAwareStrategy(frozen),
    )
    return {
        "val_annual_return": full["annual_return"],
        "val_total_return": full["total_return"],
        "val_sharpe": full["sharpe_rf0"],
        "val_max_drawdown": full["max_drawdown"],
        "val_fees": full["total_fees"],
        "val_trades": full["trade_count"],
        "first_half_return": half1["total_return"],
        "first_half_annual": half1["annual_return"],
        "second_half_return": half2["total_return"],
        "second_half_annual": half2["annual_return"],
        "worst_half_return": min(half1["total_return"], half2["total_return"]),
    }


def save_val_predictions(folder, dates, truth, prediction):
    pd.DataFrame(
        {"y_true_ret": np.asarray(truth, dtype=float),
         "y_pred_ret": np.asarray(prediction, dtype=float)},
        index=dates.rename("date"),
    ).to_csv(folder / "validation_predictions.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")


def select_winner(trials):
    """稳健过滤 + 年化收益排序；返回 (winner行, 是否通过稳健过滤)。"""
    robust = trials[
        (trials["val_max_drawdown"] <= MAX_DRAWDOWN)
        & (trials["worst_half_return"] >= MIN_HALF_RETURN)
    ]
    pool = robust if len(robust) else trials
    ordered = pool.sort_values(
        ["val_annual_return", "val_sharpe", "val_max_drawdown", "candidate"],
        ascending=[False, False, True, True],
    )
    winner = ordered.iloc[0]
    return winner, bool(len(robust))


def tune(source, output, dataset_path, prices_path, strategy_path, device_name, cpu_threads):
    import torch

    dataset = load_dataset(dataset_path)
    source_gpu_config = json.loads((source / "gpu/run_config.json").read_text(encoding="utf-8"))
    digest = sha256(dataset_path.read_bytes()).hexdigest()
    if source_gpu_config["dataset_sha256"] != digest:
        raise ValueError("数据与原运行不一致。")
    output.mkdir(parents=True, exist_ok=False)
    (output / "search").mkdir()
    labels = dataset.labels["trade"]
    frozen = StrategyConfig(
        **json.loads(strategy_path.read_text(encoding="utf-8"))["config"]
    )
    prices = pd.read_csv(
        prices_path, index_col="date", parse_dates=["date"],
        encoding="utf-8-sig", float_precision="round_trip",
    )
    prices_val = prices.loc[:VAL_END]
    columns = input_columns(dataset, "full16")
    fit_dates = build_sequence(dataset.history[columns], labels["train"].index, 336)[2]
    normalized, recipe = prepare_inputs(dataset, columns, fit_dates)
    val_dates = labels["val"].index
    y_train = labels["train"].loc[fit_dates]
    y_val = labels["val"]
    device = verify_torch(device_name, cpu_threads)
    plan = {
        "objective": "validation annualized return under frozen strategy (fees included)",
        "source_run": str(source.resolve()),
        "dataset_sha256": digest,
        "frozen_strategy_sha256": sha256(strategy_path.read_bytes()).hexdigest(),
        "frozen_strategy": json.loads(strategy_path.read_text(encoding="utf-8"))["config"],
        "training_rows": len(fit_dates),
        "training_end": fit_dates[-1].strftime("%Y-%m-%d"),
        "validation_dates": [val_dates[0].strftime("%Y-%m-%d"), val_dates[-1].strftime("%Y-%m-%d")],
        "validation_rows": len(val_dates),
        "early_stopping": "full validation RMSE, patience 5 (same period also used for selection)",
        "selection_rule": (
            "prefer candidates with worst-half total return >= -5% and max drawdown <= 25%; "
            "rank by validation annual return, then Sharpe, then smaller drawdown"
        ),
        "test_used_for_selection": False,
        "test_already_seen_in_prior_experiments": True,
        "seed": 42,
        "device": device_name,
        "cpu_threads": cpu_threads,
        "patchtst_candidates": [c[0]["id"] for c in PATCHTST_CANDIDATES],
        "tqnet_candidates": [c[0]["id"] for c in TQNET_CANDIDATES],
        "lightgbm_candidates": [c[0] for c in LGBM_CANDIDATES],
        "ridge_candidates": [f"alpha_{a:g}" for a in RIDGE_ALPHAS],
    }
    (output / "search_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    trials = []
    trials_path = output / "tuning_trials.csv"

    def record(row):
        trials.append(row)
        pd.DataFrame(trials).to_csv(trials_path, index=False, encoding="utf-8-sig")
        print(
            f"{row['model']}/{row['candidate']}: val_annual={row.get('val_annual_return', float('nan')):.2%} "
            f"total={row.get('val_total_return', float('nan')):.2%} Sharpe={row.get('val_sharpe', float('nan')):.3f} "
            f"DD={row.get('val_max_drawdown', float('nan')):.2%} "
            f"halves=({row.get('first_half_return', float('nan')):.2%}, "
            f"{row.get('second_half_return', float('nan')):.2%})",
            flush=True,
        )

    # ---- Ridge / LightGBM（表格模型，便宜，先跑）----
    tables = {"ridge": dataset.ridge, "lightgbm": dataset.lightgbm}
    x_tr = {name: table["train"].loc[fit_dates] for name, table in tables.items()}
    x_va = {name: table["val"].loc[val_dates] for name, table in tables.items()}
    for alpha in RIDGE_ALPHAS:
        cid = f"alpha_{alpha:g}"
        folder = output / "search" / "ridge" / cid
        folder.mkdir(parents=True)
        try:
            from joblib import dump
            from sklearn.linear_model import Ridge

            start = perf_counter()
            model = Ridge(alpha=alpha)
            model.fit(x_tr["ridge"], y_train)
            dump(model, folder / "model.joblib")
            pred = model.predict(x_va["ridge"])
            seconds = perf_counter() - start
            save_val_predictions(folder, val_dates, y_val, pred)
            series = pd.Series(pred, index=val_dates)
            score = score_predictions(prices_val, series, frozen)
            error = y_val.to_numpy() - pred
            record({
                "model": "ridge", "candidate": cid, "alpha": alpha,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "direction_accuracy": float(np.mean(np.sign(y_val.to_numpy()) == np.sign(pred))),
                "seconds": seconds, **score,
            })
        except Exception as error:  # noqa: BLE001 - 单个候选失败不中断搜索
            record({"model": "ridge", "candidate": cid, "alpha": alpha, "error": repr(error)})
    for cid, params in LGBM_CANDIDATES:
        folder = output / "search" / "lightgbm" / cid
        folder.mkdir(parents=True)
        try:
            from lightgbm import LGBMRegressor, early_stopping

            start = perf_counter()
            model = LGBMRegressor(
                objective="regression", n_estimators=500, random_state=42,
                n_jobs=cpu_threads, verbosity=-1, device_type="cpu", **params,
            )
            model.fit(
                x_tr["lightgbm"], y_train, eval_set=[(x_va["lightgbm"], y_val)],
                eval_metric="l2", callbacks=[early_stopping(50, verbose=False)],
            )
            best = model.best_iteration_
            text = model.booster_.model_to_string(num_iteration=best)
            (folder / "model.txt").write_bytes(text.encode("utf-8"))
            pred = model.predict(x_va["lightgbm"])
            seconds = perf_counter() - start
            save_val_predictions(folder, val_dates, y_val, pred)
            series = pd.Series(pred, index=val_dates)
            score = score_predictions(prices_val, series, frozen)
            error = y_val.to_numpy() - pred
            record({
                "model": "lightgbm", "candidate": cid, **params,
                "best_iteration": best,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "direction_accuracy": float(np.mean(np.sign(y_val.to_numpy()) == np.sign(pred))),
                "seconds": seconds, **score,
            })
        except Exception as error:  # noqa: BLE001
            record({"model": "lightgbm", "candidate": cid, **params, "error": repr(error)})

    # ---- TQNet / PatchTST（时序模型）----
    for name, candidates in (("tqnet", TQNET_CANDIDATES), ("patchtst", PATCHTST_CANDIDATES)):
        for candidate, lookback, builder_kind, cycle in candidates:
            cid = candidate["id"]
            folder = output / "search" / name / cid
            folder.mkdir(parents=True)
            try:
                dates = {"train": fit_dates, "stopping": val_dates, "val": val_dates}
                arrays = {split: build_sequence(normalized, index, lookback)
                          for split, index in dates.items()}
                labels_in = {"train": y_train, "stopping": y_val, "val": y_val, "selection": y_val}
                builder = short_factory if builder_kind == "short" else build_tuned
                score, prediction = fit_candidate(
                    name, candidate, arrays, labels_in, recipe, folder, cycle, device, 42,
                    builder=builder,
                )
                if builder_kind == "short":
                    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
                    checkpoint["short_window_model"] = True
                    torch.save(checkpoint, folder / "model.pt")
                series = pd.Series(prediction, index=val_dates)
                backtest = score_predictions(prices_val, series, frozen)
                record({"model": name, "candidate": cid, "lookback": lookback,
                        "cycle": cycle if name == "tqnet" else None,
                        **{k: candidate[k] for k in ("lr", "dropout", "head_dropout", "patch_len",
                                                     "stride", "decay", "patch_decay", "loss",
                                                     "direction", "optimizer", "epochs")},
                        "rmse": score["rmse"],
                        "direction_accuracy": score["direction_accuracy"],
                        "balanced_direction_accuracy": score["balanced_direction_accuracy"],
                        "best_epoch": score["best_epoch"],
                        "completed_epochs": score["completed_epochs"],
                        "seconds": score["seconds"], **backtest})
            except Exception as error:  # noqa: BLE001
                record({"model": name, "candidate": cid, "lookback": lookback,
                        "cycle": cycle if name == "tqnet" else None, "error": repr(error)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- 按规则选参并冻结 ----
    table = pd.DataFrame(trials)
    ok = table[table["error"].isna()] if "error" in table else table
    winners = {}
    for name in ("ridge", "lightgbm", "tqnet", "patchtst"):
        subset = ok[ok["model"] == name]
        if subset.empty:
            raise RuntimeError(f"{name} 没有成功完成的候选。")
        winner, robust = select_winner(subset)
        winners[name] = {**winner.to_dict(), "robust_filter_passed": robust}
        print(f"选中 {name}: {winner['candidate']} "
              f"val_annual={winner['val_annual_return']:.2%} robust={robust}", flush=True)
    (output / "selected_models.json").write_text(
        json.dumps(winners, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(f"候选已冻结，输出 {output.resolve()}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prices", type=Path, default=root / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=root / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output or root / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_annualret_{uuid4().hex[:6]}"
    tune(args.source, output, args.dataset or next((root / "data").glob("*.zip")),
         args.prices, args.strategy, args.device, args.cpu_threads)


if __name__ == "__main__":
    main()
