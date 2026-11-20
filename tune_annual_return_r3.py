"""第三轮年化收益调参：针对第二轮的问题，把多种子与集成做透。

第二轮留下的问题：
1. 单种子验证不可靠已证实（同配置三种子验证年化可差 40-60pp），但第二轮
   只试了 3 个种子，且选参规则仍以单种子分数为准，系统性地偏向幸运种子。
   本轮改规则：时序模型与 LightGBM 的最终排名只用集成（多种子平均）的
   验证年化，单种子分数只用于发现候选，不用于最终排名。
2. 集成没做透：PatchTST 只试了 3 配置集成（未加种子维度），TQNet/LightGBM
   只试了 3 种子（未加配置维度）。本轮做 5 种子与 15 成员（3 配置 × 5 种子）集成。
3. PatchTST 的三种子集成没做测试诊断；本轮的 15 成员集成包含它。
4. 学习率调度、方向损失强度、PatchTST 细粒度 patch、96 日窗口未试过。
5. Ridge 的验证峰在 7500-10000 之间，第二轮的 α7500 验证赢了但测试输了，
   说明 2pp 以内的验证差距是噪声——本轮 Ridge 换配置须超过 α10000 达 2pp 才算数。
"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data import build_sequence, load_dataset
from src.strategy import CostAwareStrategy, StrategyConfig
from src.training import verify_torch
from src.tuned_models import build_model as build_tuned
from src.tuning_data import input_columns, prepare_inputs
from tune_annual_return import (
    LGBM_CANDIDATES as R1_LGBM, LGBM_REF, PATCHTST_CANDIDATES as R1_PATCHTST,
    TQNET_CANDIDATES as R1_TQNET, VAL_END, score_predictions, select_winner,
    save_val_predictions, seq_candidate, short_factory,
)
from tune_annual_return_r2 import (
    LGBM_R2, PATCHTST_R2, RIDGE_R2, TQNET_R2, mid_factory,
)
from tune_trade_models import fit_candidate


TQNET_R3 = [
    (seq_candidate("lr5e4_sched", lr=5e-4, optimizer="Adam", decay=0.0, epochs=30,
                   use_scheduler=True), 64, "tuned", 1),
    (seq_candidate("lr5e4_dir04", lr=5e-4, direction=0.4, optimizer="Adam",
                   decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lr5e4_huber", lr=5e-4, loss="huber", optimizer="Adam",
                   decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lb96_lr5e4", lr=5e-4, optimizer="Adam", decay=0.0, epochs=30), 96, "tuned", 1),
]
PATCHTST_R3 = [
    (seq_candidate("64_lr5e4_sched", lr=5e-4, encoder_layers=2, use_scheduler=True), 64, "short", 1),
    (seq_candidate("64_lr5e4_patch4", lr=5e-4, patch_len=4, stride=2, encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_lr6e4", lr=6e-4, encoder_layers=2), 64, "short", 1),
]
LGBM_R3 = [
    ("minchild50", dict(min_child_samples=50)),
    ("minchild40_sub06", dict(min_child_samples=40, subsample=0.6)),
]
RIDGE_R3 = [8500.0]
MODEL_FILES = {"ridge": "model.joblib", "lightgbm": "model.txt",
               "tqnet": "model.pt", "patchtst": "model.pt"}
ALL_SEEDS = (42, 43, 44, 45, 46)
RIDGE_NOISE_PP = 0.02


def build_registry():
    """(model, candidate) -> 重训所需的规格。"""
    specs = {}
    for candidate, lookback, builder_kind, cycle in R1_TQNET:
        specs[("tqnet", candidate["id"])] = (candidate, lookback, builder_kind, cycle)
    for candidate, lookback, builder_kind, cycle in R1_PATCHTST:
        specs[("patchtst", candidate["id"])] = (candidate, lookback, builder_kind, cycle)
    for cid, params in R1_LGBM:
        specs[("lightgbm", cid)] = params
    for alpha in (100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0):
        specs[("ridge", f"alpha_{alpha:g}")] = alpha
    for candidate, lookback, builder_kind, cycle in TQNET_R2:
        specs[("tqnet", candidate["id"])] = (candidate, lookback, builder_kind, cycle)
    for candidate, lookback, builder_kind, cycle in PATCHTST_R2:
        specs[("patchtst", candidate["id"])] = (candidate, lookback, builder_kind, cycle)
    for cid, delta in LGBM_R2:
        params = dict(LGBM_REF)
        params.update(delta)
        specs[("lightgbm", cid)] = params
    for alpha in RIDGE_R2:
        specs[("ridge", f"alpha_{alpha:g}")] = alpha
    for candidate, lookback, builder_kind, cycle in TQNET_R3:
        specs[("tqnet", candidate["id"])] = (candidate, lookback, builder_kind, cycle)
    for candidate, lookback, builder_kind, cycle in PATCHTST_R3:
        specs[("patchtst", candidate["id"])] = (candidate, lookback, builder_kind, cycle)
    for cid, delta in LGBM_R3:
        params = dict(LGBM_REF)
        params.update(delta)
        specs[("lightgbm", cid)] = params
    for alpha in RIDGE_R3:
        specs[("ridge", f"alpha_{alpha:g}")] = alpha
    return specs


def tune_r3(r1_run, r2_run, output, dataset_path, prices_path, strategy_path,
            device_name, cpu_threads):
    import torch

    dataset = load_dataset(dataset_path)
    r1_config = json.loads((r1_run / "gpu/run_config.json").read_text(encoding="utf-8"))
    digest = sha256(dataset_path.read_bytes()).hexdigest()
    if r1_config["dataset_sha256"] != digest:
        raise ValueError("数据与第一轮运行不一致。")
    output.mkdir(parents=True, exist_ok=False)
    search = output / "search_r3"
    search.mkdir()
    r1_search = r1_run / "search"
    r2_search = r2_run / "search_r2"
    r1_trials = pd.read_csv(r1_run / "tuning_trials.csv", encoding="utf-8-sig")
    r2_trials = pd.read_csv(r2_run / "tuning_trials_r2.csv", encoding="utf-8-sig")
    labels = dataset.labels["trade"]
    frozen = StrategyConfig(**json.loads(strategy_path.read_text(encoding="utf-8"))["config"])
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
    specs = build_registry()
    plan = {
        "round": 3,
        "round1_run": str(r1_run.resolve()),
        "round2_run": str(r2_run.resolve()),
        "objective": "validation annualized return under frozen strategy (fees included)",
        "selection_rule": ("sequence and lightgbm models are ranked ONLY by ensemble "
                           "(multi-seed averaged) validation annual return with the "
                           "robustness filter (worst-half >= -5%, max drawdown <= 25%); "
                           "single-seed scores are used for candidate discovery only. "
                           "Ridge (deterministic) keeps alpha_10000 unless a challenger "
                           "beats it by more than 2pp on validation."),
        "round2_problems": [
            "single-seed validation unreliable (same config varied 20-88% across seeds), only 3 seeds tried",
            "ensembling incomplete: PatchTST config-only, TQNet/LightGBM seed-only (3 seeds)",
            "PatchTST seed ensemble never got a test diagnostic",
            "scheduler, direction-loss strength, fine patches, 96-day window never tried",
            "Ridge alpha_7500 validation win (+1.34pp) reversed on test: sub-2pp validation gaps are noise",
        ],
        "new_candidates": {
            "tqnet": [c[0]["id"] for c in TQNET_R3],
            "patchtst": [c[0]["id"] for c in PATCHTST_R3],
            "lightgbm": [c[0] for c in LGBM_R3],
            "ridge": [f"alpha_{a:g}" for a in RIDGE_R3],
        },
        "seeds": list(ALL_SEEDS),
        "test_used_for_selection": False,
        "seed": 42,
        "device": device_name,
        "cpu_threads": cpu_threads,
    }
    (output / "search_plan_r3.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    trials = []
    trials_path = output / "tuning_trials_r3.csv"

    def record(row):
        trials.append(row)
        pd.DataFrame(trials).to_csv(trials_path, index=False, encoding="utf-8-sig")
        annual = row.get("val_annual_return", float("nan"))
        print(f"{row['model']}/{row['candidate']}: val_annual={annual:.2%} "
              f"Sharpe={row.get('val_sharpe', float('nan')):.3f} "
              f"DD={row.get('val_max_drawdown', float('nan')):.2%} "
              f"halves=({row.get('first_half_return', float('nan')):.2%}, "
              f"{row.get('second_half_return', float('nan')):.2%})", flush=True)

    def direction_accuracy(pred):
        return float(np.mean(np.sign(y_val.to_numpy()) == np.sign(pred)))

    def rmse_of(pred):
        error = y_val.to_numpy() - pred
        return float(np.sqrt(np.mean(error ** 2)))

    def load_predictions(folder):
        frame = pd.read_csv(
            Path(folder) / "validation_predictions.csv", index_col="date",
            parse_dates=["date"], encoding="utf-8-sig", float_precision="round_trip",
        )
        return frame["y_pred_ret"].to_numpy(dtype=float)

    def score_and_record(model, candidate, kind, prediction, folder, extra=None):
        save_val_predictions(folder, val_dates, y_val, prediction)
        series = pd.Series(prediction, index=val_dates)
        backtest = score_predictions(prices_val, series, frozen)
        record({"model": model, "candidate": candidate, "kind": kind,
                "rmse": rmse_of(prediction), "direction_accuracy": direction_accuracy(prediction),
                **(extra or {}), **backtest})
        return backtest["val_annual_return"]

    arrays_cache = {}

    def arrays_for(lookback):
        if lookback not in arrays_cache:
            dates = {"train": fit_dates, "stopping": val_dates, "val": val_dates}
            arrays_cache[lookback] = {split: build_sequence(normalized, index, lookback)
                                      for split, index in dates.items()}
        return arrays_cache[lookback]

    def run_sequence(name, candidate, lookback, builder_kind, cycle, seed, folder):
        labels_in = {"train": y_train, "stopping": y_val, "val": y_val, "selection": y_val}
        builder = {"tuned": build_tuned, "short": short_factory, "mid": mid_factory}[builder_kind]
        score, prediction = fit_candidate(
            name, candidate, arrays_for(lookback), labels_in, recipe, folder,
            cycle, device, seed, builder=builder,
        )
        if builder_kind in ("short", "mid"):
            checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
            checkpoint["short_window_model" if builder_kind == "short" else "mid_window_model"] = True
            torch.save(checkpoint, folder / "model.pt")
        return score, prediction

    tables = {"ridge": dataset.ridge, "lightgbm": dataset.lightgbm}
    x_tr = {name: table["train"].loc[fit_dates] for name, table in tables.items()}
    x_va = {name: table["val"].loc[val_dates] for name, table in tables.items()}

    def run_lgbm(params, seed, folder):
        from lightgbm import LGBMRegressor, early_stopping

        model = LGBMRegressor(
            objective="regression", n_estimators=500, random_state=seed,
            n_jobs=cpu_threads, verbosity=-1, device_type="cpu", **params,
        )
        model.fit(
            x_tr["lightgbm"], y_train, eval_set=[(x_va["lightgbm"], y_val)],
            eval_metric="l2", callbacks=[early_stopping(50, verbose=False)],
        )
        text = model.booster_.model_to_string(num_iteration=model.best_iteration_)
        (folder / "model.txt").write_bytes(text.encode("utf-8"))
        return model.predict(x_va["lightgbm"]), model.best_iteration_

    def train_one(model, candidate, seed):
        """在 search_r3 下训练指定种子；返回预测数组。"""
        folder = search / model / (candidate if seed == 42 else f"{candidate}_s{seed}")
        folder.mkdir(parents=True, exist_ok=True)
        spec = specs[(model, candidate)]
        if model == "ridge":
            from joblib import dump
            from sklearn.linear_model import Ridge

            fitted = Ridge(alpha=spec)
            fitted.fit(x_tr["ridge"], y_train)
            dump(fitted, folder / "model.joblib")
            return fitted.predict(x_va["ridge"])
        if model == "lightgbm":
            pred, _ = run_lgbm(spec, seed, folder)
            return pred
        candidate_dict, lookback, builder_kind, cycle = spec
        _, pred = run_sequence(model, candidate_dict, lookback, builder_kind, cycle, seed, folder)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return pred

    def seed42_folder(model, candidate):
        for base in (r1_search, r2_search, search):
            folder = base / model / candidate
            if (folder / "validation_predictions.csv").exists():
                return folder
        raise FileNotFoundError(f"找不到 {model}/{candidate} 的种子 42 预测。")

    def seed_folder(model, candidate, seed):
        if seed == 42:
            return seed42_folder(model, candidate)
        folder = search / model / f"{candidate}_s{seed}"
        if (folder / "validation_predictions.csv").exists():
            return folder
        folder = r2_search / model / f"{candidate}_s{seed}"
        if (folder / "validation_predictions.csv").exists():
            return folder
        return None

    # ---- Phase A: 新候选（种子 42，发现用）----
    for alpha in RIDGE_R3:
        cid = f"alpha_{alpha:g}"
        try:
            pred = train_one("ridge", cid, 42)
            score_and_record("ridge", cid, "r3_refine", pred, search / "ridge" / cid,
                             {"alpha": alpha})
        except Exception as error:  # noqa: BLE001
            record({"model": "ridge", "candidate": cid, "kind": "r3_refine",
                    "alpha": alpha, "error": repr(error)})
    for cid, _ in LGBM_R3:
        try:
            pred = train_one("lightgbm", cid, 42)
            score_and_record("lightgbm", cid, "r3_refine", pred, search / "lightgbm" / cid,
                             {**specs[("lightgbm", cid)]})
        except Exception as error:  # noqa: BLE001
            record({"model": "lightgbm", "candidate": cid, "kind": "r3_refine",
                    "error": repr(error)})
    for name, candidates in (("tqnet", TQNET_R3), ("patchtst", PATCHTST_R3)):
        for candidate, lookback, builder_kind, cycle in candidates:
            cid = candidate["id"]
            try:
                folder = search / name / cid
                folder.mkdir(parents=True, exist_ok=True)
                score, pred = run_sequence(name, candidate, lookback, builder_kind,
                                           cycle, 42, folder)
                score_and_record(
                    name, cid, "r3_refine", pred, folder,
                    {"lookback": lookback, "cycle": cycle if name == "tqnet" else None,
                     **{k: candidate[k] for k in ("lr", "dropout", "head_dropout", "patch_len",
                                                  "stride", "decay", "patch_decay", "loss",
                                                  "direction", "optimizer", "epochs")},
                     "balanced_direction_accuracy": score["balanced_direction_accuracy"],
                     "best_epoch": score["best_epoch"],
                     "completed_epochs": score["completed_epochs"],
                     "seconds": score["seconds"]})
            except Exception as error:  # noqa: BLE001
                record({"model": name, "candidate": cid, "kind": "r3_refine",
                        "lookback": lookback,
                        "cycle": cycle if name == "tqnet" else None, "error": repr(error)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- Phase B: 每模型前三单配置补齐 5 个种子 ----
    singles = pd.concat([
        r1_trials.assign(kind="r1"),
        r2_trials[r2_trials["kind"] == "r2_refine"],
        pd.DataFrame(trials),
    ], ignore_index=True)
    singles_ok = singles[singles["error"].isna()] if "error" in singles else singles
    top_configs = {}
    for name in ("tqnet", "patchtst", "lightgbm"):
        ordered = singles_ok[singles_ok["model"] == name].sort_values(
            "val_annual_return", ascending=False)
        distinct, seen = [], []
        for candidate in ordered["candidate"]:
            try:
                pred = load_predictions(seed42_folder(name, candidate))
            except FileNotFoundError:
                continue
            if any(np.array_equal(pred, other) for other in seen):
                continue
            seen.append(pred)
            distinct.append(candidate)
            if len(distinct) == 3:
                break
        top_configs[name] = distinct
        print(f"{name} 前三单配置: {distinct}", flush=True)
    for name, configs in top_configs.items():
        for candidate in configs:
            for seed in (43, 44, 45, 46):
                if seed_folder(name, candidate, seed) is not None:
                    continue
                try:
                    pred = train_one(name, candidate, seed)
                    folder = search / name / f"{candidate}_s{seed}"
                    save_val_predictions(folder, val_dates, y_val, pred)
                    print(f"补种子 {name}/{candidate} seed={seed} 完成", flush=True)
                except Exception as error:  # noqa: BLE001
                    print(f"补种子 {name}/{candidate} seed={seed} 失败: {error!r}", flush=True)

    # ---- Phase C: 集成（5 种子 + 15 成员）----
    def average_predictions(folders):
        return np.mean([load_predictions(folder) for folder in folders], axis=0)

    ensemble_rows = []
    for name, configs in top_configs.items():
        for candidate in configs:
            folders = [seed_folder(name, candidate, seed) for seed in ALL_SEEDS]
            if any(folder is None for folder in folders):
                print(f"跳过 {name}/{candidate} 的 5 种子集成：种子不全", flush=True)
                continue
            folder = search / name / f"{candidate}_seedens5"
            folder.mkdir(parents=True, exist_ok=True)
            averaged = average_predictions(folders)
            annual = score_and_record(
                name, f"{candidate}_seedens5", "seed5_ensemble", averaged, folder,
                {"members": ",".join(f"{candidate}@{seed}" for seed in ALL_SEEDS)})
            ensemble_rows.append({"model": name, "candidate": f"{candidate}_seedens5",
                                  "kind": "seed5_ensemble", "val_annual_return": annual,
                                  "members": [(candidate, seed) for seed in ALL_SEEDS]})
        all_folders, member_names = [], []
        for candidate in configs:
            for seed in ALL_SEEDS:
                folder = seed_folder(name, candidate, seed)
                if folder is None:
                    break
                all_folders.append(folder)
                member_names.append(f"{candidate}@{seed}")
            else:
                continue
            break
        if len(all_folders) == 15:
            folder = search / name / "ens15"
            folder.mkdir(parents=True, exist_ok=True)
            averaged = average_predictions(all_folders)
            annual = score_and_record(name, "ens15", "ens15", averaged, folder,
                                      {"members": ",".join(member_names)})
            ensemble_rows.append({"model": name, "candidate": "ens15", "kind": "ens15",
                                  "val_annual_return": annual,
                                  "members": [(c, s) for c in configs for s in ALL_SEEDS]})
    # 第二轮最终集成作为对照候选（验证分数已知，检查点已存在）
    r2_final_ensembles = {
        "tqnet": ("lr5e4_seedens", r2_search / "tqnet/lr5e4_seedens",
                  [("lr5e4", 42), ("lr5e4", 43), ("lr5e4", 44)]),
        "lightgbm": ("leaves15_minchild40_seedens",
                     r2_search / "lightgbm/leaves15_minchild40_seedens",
                     [("leaves15_minchild40", 42), ("leaves15_minchild40", 43),
                      ("leaves15_minchild40", 44)]),
        "patchtst": ("ens_top3", r2_search / "patchtst/ens_top3", None),
    }
    r2_selected = json.loads((r2_run / "selected_models_r2.json").read_text(encoding="utf-8"))
    for name, (candidate, folder, members) in r2_final_ensembles.items():
        if not (folder / "validation_predictions.csv").exists():
            continue
        if name == "patchtst":
            annual = float(r2_selected["patchtst"]["val_annual_return"])
            checkpoints = r2_selected["patchtst"]["checkpoints"]
        else:
            row = r2_trials[(r2_trials["model"] == name)
                            & (r2_trials["candidate"] == candidate)]
            annual = float(row.iloc[0]["val_annual_return"])
            checkpoints = [{"folder": str(seed_folder(name, c, s)),
                            "file": MODEL_FILES[name]} for c, s in members]
        ensemble_rows.append({"model": name, "candidate": candidate, "kind": "r2_final_ensemble",
                              "val_annual_return": annual, "folder": str(folder),
                              "checkpoints": checkpoints,
                              "members": members})

    # ---- Phase D: 选参（集成排名；Ridge 2pp 噪声门槛）----
    table = pd.DataFrame(trials)
    ok = table[table["error"].isna()] if "error" in table else table
    winners = {}
    for name in ("tqnet", "patchtst", "lightgbm"):
        pool = ok[(ok["model"] == name)
                  & (ok["kind"].isin(["seed5_ensemble", "ens15"]))]
        rows = []
        for _, row in pool.iterrows():
            rows.append({"candidate": row["candidate"], "kind": row["kind"],
                         "val_annual_return": row["val_annual_return"],
                         "val_sharpe": row["val_sharpe"],
                         "val_max_drawdown": row["val_max_drawdown"],
                         "worst_half_return": row["worst_half_return"],
                         "first_half_return": row["first_half_return"],
                         "second_half_return": row["second_half_return"],
                         "val_total_return": row["val_total_return"],
                         "rmse": row["rmse"],
                         "direction_accuracy": row["direction_accuracy"]})
        for entry in ensemble_rows:
            if entry["model"] == name and entry["kind"] == "r2_final_ensemble":
                pred = load_predictions(entry["folder"])
                backtest = score_predictions(prices_val, pd.Series(pred, index=val_dates), frozen)
                rows.append({"candidate": entry["candidate"], "kind": entry["kind"],
                             "rmse": rmse_of(pred),
                             "direction_accuracy": direction_accuracy(pred), **backtest})
        frame = pd.DataFrame(rows)
        winner, robust = select_winner(frame)
        entry = next(e for e in ensemble_rows
                     if e["model"] == name and e["candidate"] == winner["candidate"])
        if winner["kind"] == "r2_final_ensemble":
            checkpoints = entry["checkpoints"]
            folder = entry["folder"]
        else:
            folder = search / name / winner["candidate"]
            checkpoints = []
            for candidate, seed in entry["members"]:
                member_folder = seed_folder(name, candidate, seed)
                checkpoints.append({"folder": str(member_folder), "file": MODEL_FILES[name]})
        winners[name] = {
            "model": name, "kind": winner["kind"], "candidate": winner["candidate"],
            "source_kind": winner["kind"], "folder": str(folder), "checkpoints": checkpoints,
            "val_annual_return": float(winner["val_annual_return"]),
            "val_total_return": float(winner["val_total_return"]),
            "val_sharpe": float(winner["val_sharpe"]),
            "val_max_drawdown": float(winner["val_max_drawdown"]),
            "first_half_return": float(winner["first_half_return"]),
            "second_half_return": float(winner["second_half_return"]),
            "rmse": float(winner["rmse"]),
            "direction_accuracy": float(winner["direction_accuracy"]),
            "seconds": None, "robust_filter_passed": bool(robust),
        }
        print(f"选中 {name}: {winner['candidate']} ({winner['kind']}) "
              f"val_annual={winner['val_annual_return']:.2%} robust={robust}", flush=True)
    # Ridge：确定性模型，挑战者须超过 α10000 达 2pp 才换
    ridge_pool = singles_ok[singles_ok["model"] == "ridge"].sort_values(
        "val_annual_return", ascending=False)
    base = float(r1_trials[(r1_trials["model"] == "ridge")
                           & (r1_trials["candidate"] == "alpha_10000")].iloc[0]["val_annual_return"])
    best = ridge_pool.iloc[0]
    if best["candidate"] == "alpha_10000" or best["val_annual_return"] - base <= RIDGE_NOISE_PP:
        chosen, annual = "alpha_10000", base
    else:
        chosen, annual = best["candidate"], float(best["val_annual_return"])
    folder = seed42_folder("ridge", chosen)
    pred = load_predictions(folder)
    backtest = score_predictions(prices_val, pd.Series(pred, index=val_dates), frozen)
    winners["ridge"] = {
        "model": "ridge", "kind": "single", "candidate": chosen, "source_kind": "ridge_2pp_rule",
        "folder": str(folder),
        "checkpoints": [{"folder": str(folder), "file": "model.joblib"}],
        "val_annual_return": float(backtest["val_annual_return"]),
        "val_total_return": float(backtest["val_total_return"]),
        "val_sharpe": float(backtest["val_sharpe"]),
        "val_max_drawdown": float(backtest["val_max_drawdown"]),
        "first_half_return": float(backtest["first_half_return"]),
        "second_half_return": float(backtest["second_half_return"]),
        "rmse": rmse_of(pred), "direction_accuracy": direction_accuracy(pred),
        "seconds": None, "robust_filter_passed": bool(
            backtest["val_max_drawdown"] <= 0.25 and backtest["worst_half_return"] >= -0.05),
    }
    print(f"选中 ridge: {chosen} val_annual={annual:.2%}", flush=True)
    (output / "selected_models_r3.json").write_text(
        json.dumps(winners, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(f"候选已冻结，输出 {output.resolve()}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prices", type=Path, default=root / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=root / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output or root / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_annualret3_{uuid4().hex[:6]}"
    tune_r3(args.r1_run, args.r2_run, output,
            args.dataset or next((root / "data").glob("*.zip")),
            args.prices, args.strategy, args.device, args.cpu_threads)


if __name__ == "__main__":
    main()
