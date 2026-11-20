"""第二轮年化收益调参：优胜点附近加密 + 组合 + 集成 + 多种子稳健性。

针对第一轮的问题：
1. 单种子 42，时序模型结果有运气成分——优胜配置换两个种子重跑，种子平均好才算数。
2. 单变量单轮搜索，没试组合，也没在优胜点附近加密——本轮加密 Ridge alpha、
   LightGBM min_child_samples，并在 lr 5e-4 优胜点上叠加正则/方向损失/dropout。
3. 验证期只有一个上涨行情——选参规则不变（验证年化 + 两半段稳健过滤），
   但用多种子平均与集成降低单次选择的方差；PatchTST 的窗口之谜补试 128 日。
4. 第一轮没试集成——每模型取验证前 2/前 3 名预测平均，不用重训。
5. 早停与选参共用验证期（已披露），本轮不变，保持可比。
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
from src.mid_models import build_model as build_mid
from src.strategy import CostAwareStrategy, StrategyConfig
from src.training import verify_torch
from src.tuned_models import build_model as build_tuned
from src.tuning_data import input_columns, prepare_inputs
from tune_annual_return import (
    LGBM_REF, VAL_END, score_predictions, select_winner,
    save_val_predictions, seq_candidate, short_factory,
)
from tune_trade_models import fit_candidate


def mid_factory(name, num_features, lookback, options, cycle):
    return build_mid(name, num_features, lookback, options)


RIDGE_R2 = [7500.0, 15000.0, 20000.0]
LGBM_R2 = [
    ("minchild60", dict(min_child_samples=60)),
    ("minchild80", dict(min_child_samples=80)),
    ("minchild40_depth3", dict(min_child_samples=40, max_depth=3)),
    ("minchild40_lr003", dict(min_child_samples=40, learning_rate=0.03)),
    ("minchild40_col07", dict(min_child_samples=40, colsample_bytree=0.7)),
    ("minchild40_lambda05", dict(min_child_samples=40, reg_lambda=0.5)),
]
TQNET_R2 = [
    (seq_candidate("lr7e4", lr=7e-4, optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lr1e3", lr=1e-3, optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lr5e4_wd1e3", lr=5e-4, optimizer="Adam", decay=1e-3, epochs=30), 64, "tuned", 1),
    (seq_candidate("lr5e4_huberdir", lr=5e-4, loss="huber", direction=0.2,
                   optimizer="Adam", decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lr5e4_drop02", lr=5e-4, dropout=0.2, optimizer="Adam",
                   decay=0.0, epochs=30), 64, "tuned", 1),
    (seq_candidate("lb128_lr5e4", lr=5e-4, optimizer="Adam", decay=0.0, epochs=30), 128, "tuned", 1),
]
PATCHTST_R2 = [
    (seq_candidate("64_lr7e4", lr=7e-4, encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_lr1e3", lr=1e-3, encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_lr5e4_drop02", lr=5e-4, dropout=0.2, encoder_layers=2), 64, "short", 1),
    (seq_candidate("64_lr5e4_layers3", lr=5e-4, encoder_layers=3), 64, "short", 1),
    (seq_candidate("128_lr5e4", lr=5e-4, encoder_layers=2), 128, "mid", 1),
    (seq_candidate("128_lr3e4", lr=3e-4, encoder_layers=2), 128, "mid", 1),
]
R1_WINNERS = {"ridge": "alpha_10000", "lightgbm": "leaves15_minchild40",
              "tqnet": "lr5e4", "patchtst": "64_lr5e4"}
R1_WINNER_SPECS = {
    # (candidate dict, lookback, builder_kind, cycle)；lightgbm 为参数 dict
    "tqnet": (seq_candidate("lr5e4", lr=5e-4, optimizer="Adam", decay=0.0, epochs=30),
              64, "tuned", 1),
    "patchtst": (seq_candidate("64_lr5e4", lr=5e-4, encoder_layers=2), 64, "short", 1),
    "lightgbm": {**LGBM_REF, "num_leaves": 15, "min_child_samples": 40},
}
MODEL_FILES = {"ridge": "model.joblib", "lightgbm": "model.txt",
               "tqnet": "model.pt", "patchtst": "model.pt"}
SEEDS = (43, 44)


def tune_r2(r1_run, output, dataset_path, prices_path, strategy_path, device_name, cpu_threads):
    import torch

    dataset = load_dataset(dataset_path)
    r1_config = json.loads((r1_run / "gpu/run_config.json").read_text(encoding="utf-8"))
    digest = sha256(dataset_path.read_bytes()).hexdigest()
    if r1_config["dataset_sha256"] != digest:
        raise ValueError("数据与第一轮运行不一致。")
    output.mkdir(parents=True, exist_ok=False)
    search = output / "search_r2"
    search.mkdir()
    r1_search = r1_run / "search"
    r1_trials = pd.read_csv(r1_run / "tuning_trials.csv", encoding="utf-8-sig")
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
    plan = {
        "round": 2,
        "round1_run": str(r1_run.resolve()),
        "objective": "validation annualized return under frozen strategy (fees included)",
        "selection_rule": ("prefer candidates with worst-half total return >= -5% and max "
                           "drawdown <= 25%; rank by validation annual return, then Sharpe, "
                           "then smaller drawdown; a 3-seed average replaces its single-seed "
                           "winner when it scores higher on validation"),
        "round1_problems": [
            "single seed 42: sequence results carry seed luck, never re-checked",
            "one-at-a-time single round: no combinations, no refinement near winners",
            "validation is a single rally regime: PatchTST 64-day validation edge did not transfer to test",
            "PatchTST window not fully explored: 336 and 64 tested, 128 never tried",
            "no ensembling tried in round 1",
            "early stopping and selection share the validation period (disclosed)",
        ],
        "ridge_r2": [f"alpha_{a:g}" for a in RIDGE_R2],
        "lightgbm_r2": [c[0] for c in LGBM_R2],
        "tqnet_r2": [c[0]["id"] for c in TQNET_R2],
        "patchtst_r2": [c[0]["id"] for c in PATCHTST_R2],
        "ensembles": "per model, top-2 and top-3 round-1 candidates by validation annual return, deduplicated",
        "multiseed": "seeds 43, 44 for round-1 winners and round-2 winners (sequence + lightgbm)",
        "test_used_for_selection": False,
        "seed": 42,
        "device": device_name,
        "cpu_threads": cpu_threads,
    }
    (output / "search_plan_r2.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    trials = []
    trials_path = output / "tuning_trials_r2.csv"
    specs = {}  # (model, candidate) -> retraining spec

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

    def score_and_record(model, candidate, kind, prediction, extra):
        save_val_predictions(search / model / candidate, val_dates, y_val, prediction)
        series = pd.Series(prediction, index=val_dates)
        backtest = score_predictions(prices_val, series, frozen)
        record({"model": model, "candidate": candidate, "kind": kind,
                "rmse": rmse_of(prediction), "direction_accuracy": direction_accuracy(prediction),
                **extra, **backtest})

    # ---- Phase A: 优胜点附近加密 + 组合（种子 42）----
    tables = {"ridge": dataset.ridge, "lightgbm": dataset.lightgbm}
    x_tr = {name: table["train"].loc[fit_dates] for name, table in tables.items()}
    x_va = {name: table["val"].loc[val_dates] for name, table in tables.items()}
    for alpha in RIDGE_R2:
        cid = f"alpha_{alpha:g}"
        folder = search / "ridge" / cid
        folder.mkdir(parents=True)
        specs[("ridge", cid)] = alpha
        try:
            from joblib import dump
            from sklearn.linear_model import Ridge

            start = perf_counter()
            model = Ridge(alpha=alpha)
            model.fit(x_tr["ridge"], y_train)
            dump(model, folder / "model.joblib")
            pred = model.predict(x_va["ridge"])
            score_and_record("ridge", cid, "r2_refine", pred,
                             {"alpha": alpha, "seconds": perf_counter() - start})
        except Exception as error:  # noqa: BLE001
            record({"model": "ridge", "candidate": cid, "kind": "r2_refine",
                    "alpha": alpha, "error": repr(error)})
    for cid, delta in LGBM_R2:
        folder = search / "lightgbm" / cid
        folder.mkdir(parents=True)
        params = dict(LGBM_REF)
        params.update(delta)
        specs[("lightgbm", cid)] = params
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
            score_and_record("lightgbm", cid, "r2_refine", pred,
                             {**params, "best_iteration": best,
                              "seconds": perf_counter() - start})
        except Exception as error:  # noqa: BLE001
            record({"model": "lightgbm", "candidate": cid, "kind": "r2_refine",
                    **params, "error": repr(error)})
    for name, candidates in (("tqnet", TQNET_R2), ("patchtst", PATCHTST_R2)):
        for candidate, lookback, builder_kind, cycle in candidates:
            cid = candidate["id"]
            folder = search / name / cid
            folder.mkdir(parents=True)
            specs[(name, cid)] = (candidate, lookback, builder_kind, cycle)
            try:
                score, prediction = run_sequence(
                    name, candidate, lookback, builder_kind, cycle, 42, folder)
                score_and_record(
                    name, cid, "r2_refine", prediction,
                    {"lookback": lookback, "cycle": cycle if name == "tqnet" else None,
                     **{k: candidate[k] for k in ("lr", "dropout", "head_dropout", "patch_len",
                                                  "stride", "decay", "patch_decay", "loss",
                                                  "direction", "optimizer", "epochs")},
                     "balanced_direction_accuracy": score["balanced_direction_accuracy"],
                     "best_epoch": score["best_epoch"],
                     "completed_epochs": score["completed_epochs"],
                     "seconds": score["seconds"]})
            except Exception as error:  # noqa: BLE001
                record({"model": name, "candidate": cid, "kind": "r2_refine",
                        "lookback": lookback,
                        "cycle": cycle if name == "tqnet" else None, "error": repr(error)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- Phase B: 第一轮前 2/前 3 名预测平均（不用重训）----
    def load_r1_predictions(model, candidate):
        frame = pd.read_csv(
            r1_search / model / candidate / "validation_predictions.csv",
            index_col="date", parse_dates=["date"],
            encoding="utf-8-sig", float_precision="round_trip",
        )
        return frame["y_pred_ret"].to_numpy(dtype=float)

    for name in ("ridge", "lightgbm", "tqnet", "patchtst"):
        ordered = r1_trials[r1_trials["model"] == name].sort_values(
            "val_annual_return", ascending=False)
        distinct = []
        seen = []
        for candidate in ordered["candidate"]:
            pred = load_r1_predictions(name, candidate)
            if any(np.array_equal(pred, other) for other in seen):
                continue
            seen.append(pred)
            distinct.append((candidate, pred))
            if len(distinct) == 3:
                break
        for size in (2, 3):
            if len(distinct) < size:
                continue
            members = distinct[:size]
            cid = f"ens_top{size}"
            folder = search / name / cid
            folder.mkdir(parents=True)
            averaged = np.mean([pred for _, pred in members], axis=0)
            score_and_record(name, cid, "r1_ensemble", averaged,
                             {"members": ",".join(candidate for candidate, _ in members)})
    # ---- Phase C: 多种子（43、44）重跑第一轮与第二轮优胜者 ----
    r2_table = pd.DataFrame(trials)
    r2_ok = r2_table[r2_table["error"].isna()] if "error" in r2_table else r2_table
    r2_winners = {}
    for name in ("ridge", "lightgbm", "tqnet", "patchtst"):
        subset = r2_ok[(r2_ok["model"] == name) & (r2_ok["kind"] == "r2_refine")]
        if subset.empty:
            continue
        winner, _ = select_winner(subset)
        r2_winners[name] = winner["candidate"]
        print(f"第二轮 {name} 优胜: {winner['candidate']} "
              f"val_annual={winner['val_annual_return']:.2%}", flush=True)

    def run_lgbm_seed(params, seed, folder):
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
        return model.predict(x_va["lightgbm"])

    seed_targets = []  # (model, candidate, seed42_folder, spec)
    for name in ("tqnet", "patchtst", "lightgbm"):
        seed_targets.append((name, R1_WINNERS[name], r1_search / name / R1_WINNERS[name],
                             R1_WINNER_SPECS[name]))
        if name in r2_winners:
            cid = r2_winners[name]
            seed_targets.append((name, cid, search / name / cid, specs[(name, cid)]))
    for name, cid, seed42_folder, spec in seed_targets:
        seed_preds = [load_r1_predictions(name, cid) if seed42_folder.parent.parent == r1_search
                      else pd.read_csv(
                          seed42_folder / "validation_predictions.csv", index_col="date",
                          parse_dates=["date"], encoding="utf-8-sig",
                          float_precision="round_trip")["y_pred_ret"].to_numpy(dtype=float)]
        ok = True
        for seed in SEEDS:
            folder = search / name / f"{cid}_s{seed}"
            folder.mkdir(parents=True)
            try:
                if name == "lightgbm":
                    pred = run_lgbm_seed(spec, seed, folder)
                else:
                    candidate, lookback, builder_kind, cycle = spec
                    _, pred = run_sequence(name, candidate, lookback, builder_kind,
                                           cycle, seed, folder)
                seed_preds.append(pred)
                save_val_predictions(folder, val_dates, y_val, pred)
                print(f"多种子 {name}/{cid} seed={seed} 完成", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"多种子 {name}/{cid} seed={seed} 失败: {error!r}", flush=True)
                ok = False
                break
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not ok or len(seed_preds) != 3:
            continue
        ens_folder = search / name / f"{cid}_seedens"
        ens_folder.mkdir(parents=True)
        averaged = np.mean(seed_preds, axis=0)
        score_and_record(name, f"{cid}_seedens", "seed_ensemble", averaged,
                         {"members": f"{cid}@42,{cid}@43,{cid}@44",
                          "base_candidate": cid})

    # ---- Phase D: 选参 ----
    table = pd.DataFrame(trials)
    ok = table[table["error"].isna()] if "error" in table else table
    winners = {}
    for name in ("ridge", "lightgbm", "tqnet", "patchtst"):
        r1_row = r1_trials[(r1_trials["model"] == name)
                           & (r1_trials["candidate"] == R1_WINNERS[name])].iloc[0].to_dict()
        r1_row["kind"] = "r1_winner"
        pool = pd.DataFrame([r1_row])
        pool = pd.concat([pool, ok[ok["model"] == name]], ignore_index=True)
        # 集成与种子集成只在优于单模型时才考虑：先对单模型选参，再比较集成
        singles = pool[pool["kind"].isin(["r1_winner", "r2_refine"])]
        winner, robust = select_winner(singles)
        final_kind = winner["kind"]
        final_candidate = winner["candidate"]
        if final_kind == "r1_winner":
            final_folder = r1_search / name / final_candidate
            checkpoints = [{"folder": str(final_folder), "file": MODEL_FILES[name]}]
        else:
            final_folder = search / name / final_candidate
            checkpoints = [{"folder": str(final_folder), "file": MODEL_FILES[name]}]
        ensembles = pool[pool["kind"].isin(["r1_ensemble", "seed_ensemble"])]
        if not ensembles.empty:
            ens_winner, ens_robust = select_winner(ensembles)
            if ens_winner["val_annual_return"] > winner["val_annual_return"]:
                winner, robust = ens_winner, ens_robust
                final_kind = winner["kind"]
                final_candidate = winner["candidate"]
                final_folder = search / name / final_candidate
                member_names = winner["members"].split(",")
                checkpoints = []
                for member in member_names:
                    if "@" in member:
                        base, seed = member.split("@")
                        if seed == "42":
                            member_folder = (r1_search / name / base
                                             if (r1_search / name / base).exists()
                                             else search / name / base)
                        else:
                            member_folder = search / name / f"{base}_s{seed}"
                    else:
                        member_folder = r1_search / name / member
                    checkpoints.append({"folder": str(member_folder),
                                        "file": MODEL_FILES[name]})
        entry = {
            "model": name,
            "kind": ("single" if final_kind in ("r1_winner", "r2_refine") else final_kind),
            "candidate": final_candidate,
            "source_kind": final_kind,
            "folder": str(final_folder),
            "checkpoints": checkpoints,
            "val_annual_return": float(winner["val_annual_return"]),
            "val_total_return": float(winner["val_total_return"]),
            "val_sharpe": float(winner["val_sharpe"]),
            "val_max_drawdown": float(winner["val_max_drawdown"]),
            "first_half_return": float(winner["first_half_return"]),
            "second_half_return": float(winner["second_half_return"]),
            "rmse": float(winner["rmse"]),
            "direction_accuracy": float(winner["direction_accuracy"]),
            "seconds": float(winner["seconds"]) if pd.notna(winner.get("seconds")) else None,
            "robust_filter_passed": bool(robust),
            "r1_winner_annual": float(r1_row["val_annual_return"]),
            "r2_winner": r2_winners.get(name),
        }
        winners[name] = entry
        print(f"选中 {name}: {final_candidate} ({final_kind}) "
              f"val_annual={winner['val_annual_return']:.2%} robust={robust}", flush=True)
    (output / "selected_models_r2.json").write_text(
        json.dumps(winners, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(f"候选已冻结，输出 {output.resolve()}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prices", type=Path, default=root / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=root / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output or root / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_annualret2_{uuid4().hex[:6]}"
    tune_r2(args.r1_run, output, args.dataset or next((root / "data").glob("*.zip")),
            args.prices, args.strategy, args.device, args.cpu_threads)


if __name__ == "__main__":
    main()
