"""第三轮测试诊断：解释为什么验证优胜的配置在测试上反而不如第二轮最终集。

诊断项（均为事后诊断，不用于选参）：
1. PatchTST ens15（3 配置 × 5 种子）：验证只有 +40.59%，但种子平均可能在
   测试上帮 PatchTST——第二轮没做过这个诊断。
2. PatchTST 三个 5 种子集成（ens15 的子集，免费）。
3. TQNet lr5e4 的 s45/s46 单独测试：5 种子集成测试 +5.84% 远低于 3 种子
   +16.19%，看是不是新加的两个种子拖了后腿。
4. LightGBM 2×2：minchild40/minchild50 × 3 种子/5 种子，分离“换配置”和
   “加种子”两个因素（minchild40 的 s45/s46 本脚本现训）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.strategy import CostAwareStrategy, StrategyConfig
from src.weighted_backtest import run_backtest
from verify_annualret_r3 import restore_any
from export_predictions import saved_config

ROOT = Path(__file__).parent
R1 = ROOT / "runs/20261119_201122_annualret_770f6c"
R2 = ROOT / "runs/20261119_231541_annualret2_5619ec"
R3 = ROOT / "runs/20261120_102224_annualret3_2e5131"
TEST_START = pd.Timestamp("2025-10-10")
TEST_END = pd.Timestamp("2026-10-23")


def seed_folder(model, candidate, seed):
    if seed == 42:
        for base in (R1 / "search", R2 / "search_r2", R3 / "search_r3"):
            folder = base / model / candidate
            if (folder / "validation_predictions.csv").exists():
                return folder
        raise FileNotFoundError(f"{model}/{candidate} seed 42")
    for base in (R3 / "search_r3", R2 / "search_r2"):
        folder = base / model / f"{candidate}_s{seed}"
        if (folder / "validation_predictions.csv").exists():
            return folder
    raise FileNotFoundError(f"{model}/{candidate} seed {seed}")


def main():
    import torch

    dataset = load_dataset(next((ROOT / "data").glob("*.zip")))
    test_dates = dataset.raw["test"].index
    prices = pd.read_csv(ROOT / "data/clean_prices.csv", index_col="date",
                         parse_dates=["date"], encoding="utf-8-sig",
                         float_precision="round_trip")
    # 测试 trade 标签由行情按目标日期算出（最后一行无 return_end_date，不计分）
    scored_dates = dataset.target_dates.loc[
        dataset.target_dates["return_end_date"].notna()].index
    opening = prices.loc[dataset.target_dates.loc[scored_dates, "return_start_date"], "open"]
    ending = prices.loc[dataset.target_dates.loc[scored_dates, "return_end_date"], "open"]
    y_test = (ending.to_numpy() / opening.to_numpy() - 1)
    y_test_index = scored_dates
    frozen = StrategyConfig(**json.loads((ROOT / "config/strategy_frozen.json").read_text())["config"])
    config = saved_config(R3 / "cpu", next((ROOT / "data").glob("*.zip")))
    device = torch.device("cpu")
    results = {}

    def score(name, prediction):
        series = pd.Series(prediction, index=test_dates)
        metrics, _, _ = run_backtest(prices, series, TEST_START, TEST_END,
                                     CostAwareStrategy(frozen))
        pred_scored = series.loc[y_test_index].to_numpy()
        da = float(np.mean(np.sign(y_test) == np.sign(pred_scored)))
        results[name] = {**metrics, "direction_accuracy": da}
        print(f"{name}: test_total={metrics['total_return']:.2%} "
              f"annual={metrics['annual_return']:.2%} DD={metrics['max_drawdown']:.2%} "
              f"Sharpe={metrics['sharpe']:.2f} trades={metrics['trade_count']} DA={da:.2%}",
              flush=True)

    def restore(model, candidate, seed):
        folder = seed_folder(model, candidate, seed)
        return restore_any(model, folder, config, dataset, "test", test_dates, device)

    # 1-2. PatchTST ens15 及其三个 5 种子子集
    pt_configs = ["64_lr6e4", "64_lr7e4", "64_lr5e4_sched"]
    pt_members = {}
    for candidate in pt_configs:
        pt_members[candidate] = [restore("patchtst", candidate, seed)
                                 for seed in (42, 43, 44, 45, 46)]
        score(f"patchtst/{candidate}_seedens5_test",
              np.mean(pt_members[candidate], axis=0))
    all15 = np.concatenate([pt_members[c] for c in pt_configs])
    score("patchtst/ens15_test", np.mean(all15, axis=0))
    # 验证复现检查：ens15 的验证平均须与搜索时保存的一致
    val_dates = dataset.labels["trade"]["val"].index
    val_avg = np.mean([
        restore_any("patchtst", seed_folder("patchtst", c, s), config, dataset,
                    "val", val_dates, device)
        for c in pt_configs for s in (42, 43, 44, 45, 46)], axis=0)
    saved = pd.read_csv(R3 / "search_r3/patchtst/ens15/validation_predictions.csv",
                        index_col="date", parse_dates=["date"], encoding="utf-8-sig",
                        float_precision="round_trip")["y_pred_ret"].to_numpy()
    np.testing.assert_allclose(val_avg, saved, rtol=1e-5, atol=1e-8)
    print("ens15 验证复现检查通过", flush=True)

    # 3. TQNet lr5e4 s45/s46 单独测试
    for seed in (45, 46):
        score(f"tqnet/lr5e4_s{seed}_test", restore("tqnet", "lr5e4", seed))

    # 4. LightGBM 2×2：先现训 minchild40 的 s45/s46
    from lightgbm import LGBMRegressor, early_stopping
    from tune_annual_return import LGBM_CANDIDATES
    from src.data import build_sequence
    from src.tuning_data import input_columns, prepare_inputs

    params = dict(next(p for cid, p in LGBM_CANDIDATES if cid == "leaves15_minchild40"))
    labels = dataset.labels["trade"]
    columns = input_columns(dataset, "full16")
    fit_dates = build_sequence(dataset.history[columns], labels["train"].index, 336)[2]
    x_tr = dataset.lightgbm["train"].loc[fit_dates]
    x_va = dataset.lightgbm["val"].loc[labels["val"].index]
    y_tr = labels["train"].loc[fit_dates]
    y_va = labels["val"]
    for seed in (45, 46):
        folder = R3 / "search_r3/lightgbm" / f"leaves15_minchild40_s{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        model = LGBMRegressor(objective="regression", n_estimators=500,
                              random_state=seed, n_jobs=2, verbosity=-1,
                              device_type="cpu", **params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], eval_metric="l2",
                  callbacks=[early_stopping(50, verbose=False)])
        (folder / "model.txt").write_bytes(
            model.booster_.model_to_string(num_iteration=model.best_iteration_).encode("utf-8"))
        val_pred = model.predict(x_va)
        pd.DataFrame({"y_true_ret": y_va, "y_pred_ret": val_pred},
                     index=labels["val"].index.rename("date")).to_csv(
            folder / "validation_predictions.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"现训 lightgbm/leaves15_minchild40 seed={seed} 完成", flush=True)
    lg40 = [restore("lightgbm", "leaves15_minchild40", s) for s in (42, 43, 44, 45, 46)]
    score("lightgbm/minchild40_seedens5_test", np.mean(lg40, axis=0))
    lg50 = [restore("lightgbm", "minchild50", s) for s in (42, 43, 44)]
    score("lightgbm/minchild50_seedens3_test", np.mean(lg50, axis=0))

    (R3 / "diagnostics_r3.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"诊断完成，写入 {R3 / 'diagnostics_r3.json'}", flush=True)


if __name__ == "__main__":
    main()
