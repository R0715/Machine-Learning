"""组装最终推荐配置（第二轮方法输出 + 种子集成诊断后的综合选择）。

最终配置及依据（详见调参报告第二轮）：
- ridge：第一轮 alpha_10000（确定性模型，无种子方差；第二轮 alpha_7500 的
  验证增益 +1.34pp 未迁移到测试）。
- lightgbm：leaves15_minchild40 三种子集成（种子 42/43/44 预测平均）。
- tqnet：lr5e4 三种子集成（种子 42/43/44 预测平均）。
- patchtst：第一轮前三名集成 ens_top3（第二轮验证选参的优胜者）。
种子集成是在验证集上证明单种子分数不可靠（同配置三种子的验证年化
可差 40-60pp）后，采用的标准方差缩减做法；其测试诊断结果如实报告，
不作为选参依据的替代说明。
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from backtest_research import run as run_backtest_research
from evaluate_test import evaluate as evaluate_test
from export_predictions import combine_run, saved_config
from src.data import load_dataset
from verify_annualret_r2 import restore_any


def assemble_final(r1_run, r2_run, output, dataset_path, prices_path, strategy_path, device_name):
    import torch

    if output.exists():
        raise ValueError(f"输出目录已存在：{output}")
    shutil.copytree(r2_run, output)
    for stale in ("handoff", "research_strategy_backtest", "test_metrics.csv",
                  "validation_metrics.csv", "source_validation_metrics.csv"):
        target = output / stale
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    dataset = load_dataset(dataset_path)
    test_dates = dataset.raw["test"].index
    device = torch.device(device_name)

    # ridge：用第一轮 alpha_10000 的完整结果替换
    ridge_dest = output / "cpu/trade/ridge"
    ridge_src = r1_run / "cpu/trade/ridge"
    for name in ("model.joblib", "validation_predictions.csv", "predictions.csv"):
        shutil.copy2(ridge_src / name, ridge_dest / name)

    # lightgbm / tqnet：三种子集成
    seed_members = {
        "lightgbm": [r1_run / "search/lightgbm/leaves15_minchild40",
                     r2_run / "search_r2/lightgbm/leaves15_minchild40_s43",
                     r2_run / "search_r2/lightgbm/leaves15_minchild40_s44"],
        "tqnet": [r1_run / "search/tqnet/lr5e4",
                  r2_run / "search_r2/tqnet/lr5e4_s43",
                  r2_run / "search_r2/tqnet/lr5e4_s44"],
    }
    seed_val = {
        "lightgbm": r2_run / "search_r2/lightgbm/leaves15_minchild40_seedens/validation_predictions.csv",
        "tqnet": r2_run / "search_r2/tqnet/lr5e4_seedens/validation_predictions.csv",
    }
    for name, members in seed_members.items():
        group = "cpu" if name == "lightgbm" else "gpu"
        destination = output / group / "trade" / name
        for stale in destination.iterdir():
            if stale.name not in ("validation_predictions.csv",):
                stale.unlink()
        manifest = {"kind": "seed_ensemble", "candidate": members[0].name, "members": []}
        for index, member in enumerate(members):
            ext = ".txt" if name == "lightgbm" else ".pt"
            src = member / ("model.txt" if name == "lightgbm" else "model.pt")
            dest_name = f"member_{index}{ext}"
            shutil.copy2(src, destination / dest_name)
            manifest["members"].append({"file": dest_name, "source_folder": str(member)})
        # handoff 清单要求每个任务文件夹有标准模型文件：用 member_0（种子 42）
        # 的检查点充当，ensemble.json 中注明它只是成员之一。
        first = members[0]
        canonical = "model.txt" if name == "lightgbm" else "model.pt"
        shutil.copy2(first / canonical, destination / canonical)
        manifest["canonical_model_file"] = f"{canonical} 为 member_0（种子 42）的副本，仅供清单哈希使用"
        (destination / "ensemble.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(seed_val[name], destination / "validation_predictions.csv")
        # 读回验证（平均预测须与存档一致）并生成测试预测
        config = saved_config(output / group, dataset_path)
        validation = pd.read_csv(
            destination / "validation_predictions.csv", encoding="utf-8-sig",
            index_col="date", parse_dates=["date"], float_precision="round_trip",
        )
        val_preds = [restore_any(name, member, config, dataset, "val", validation.index, device)
                     for member in members]
        np.testing.assert_allclose(np.mean(val_preds, axis=0),
                                   validation["y_pred_ret"], rtol=1e-5, atol=1e-8)
        test_preds = [restore_any(name, member, config, dataset, "test", test_dates, device)
                      for member in members]
        values = np.mean(test_preds, axis=0)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 种子集成的测试预测含非有限值。")
        pd.DataFrame({"y_pred_ret": values}, index=test_dates.rename("date")).to_csv(
            destination / "predictions.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"已生成 {name} 种子集成测试预测，{len(values)} 行", flush=True)

    # patchtst：保留第二轮 ens_top3（检查点、ensemble.json、predictions.csv 已在目录内）
    # 更新运行配置与验证指标
    final_selection = {
        "ridge": {"candidate": "alpha_10000", "source": "round1", "kind": "single",
                  "basis": "round-2 refinement gain (+1.34pp validation) did not transfer to test"},
        "lightgbm": {"candidate": "leaves15_minchild40_seedens", "source": "round1+round2 seeds",
                     "kind": "seed_ensemble",
                     "basis": "seed-variance reduction; validation seed range 51-64%"},
        "tqnet": {"candidate": "lr5e4_seedens", "source": "round1+round2 seeds",
                  "kind": "seed_ensemble",
                  "basis": "seed-variance reduction; validation seed range 43-88%"},
        "patchtst": {"candidate": "ens_top3", "source": "round2", "kind": "r1_ensemble",
                     "basis": "round-2 validation winner (+87.28%, best first-half balance)"},
        "disclosure": ("seed ensembles were NOT the validation winners (they scored below the "
                       "lucky seed-42 singles on validation); they are recommended on the "
                       "principled ground of seed-variance reduction, and their test results "
                       "are reported as a diagnostic, not as selection evidence."),
    }
    (output / "final_selection.json").write_text(
        json.dumps(final_selection, ensure_ascii=False, indent=2), encoding="utf-8")
    for group in ("cpu", "gpu"):
        path = output / group / "run_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["tuning_round"] = "final"
        config["final_selection"] = "see final_selection.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    r1_val = pd.read_csv(r1_run / "validation_metrics.csv", encoding="utf-8-sig")
    for group in ("cpu", "gpu"):
        path = output / group / "validation_metrics.csv"
        frame = pd.read_csv(r2_run / group / "validation_metrics.csv", encoding="utf-8-sig")
        for name in ("ridge", "lightgbm", "tqnet", "patchtst"):
            if ("cpu" if name in ("ridge", "lightgbm") else "gpu") != group:
                continue
            mask = frame.model.eq(name) & frame.task.eq("trade")
            if name == "ridge":
                row = r1_val[r1_val.model.eq("ridge") & r1_val.task.eq("trade")].iloc[0]
                for col in ("val_rmse", "val_mae", "val_direction_accuracy", "fit_seconds"):
                    frame.loc[mask, col] = row[col]
            elif name in ("lightgbm", "tqnet"):
                prediction = pd.read_csv(
                    output / group / "trade" / name / "validation_predictions.csv",
                    index_col="date", parse_dates=["date"],
                    encoding="utf-8-sig", float_precision="round_trip",
                )
                error = prediction["y_pred_ret"] - prediction["y_true_ret"]
                frame.loc[mask, "val_rmse"] = np.sqrt(np.mean(error ** 2))
                frame.loc[mask, "val_mae"] = np.mean(np.abs(error))
                frame.loc[mask, "val_direction_accuracy"] = np.mean(
                    np.sign(prediction["y_pred_ret"]) == np.sign(prediction["y_true_ret"]),
                )
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    pd.concat([
        pd.read_csv(output / group / "validation_metrics.csv", encoding="utf-8-sig")
        for group in ("cpu", "gpu")
    ]).to_csv(output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        pd.read_csv(r2_run / group / "validation_metrics.csv", encoding="utf-8-sig")
        for group in ("cpu", "gpu")
    ]).to_csv(output / "source_validation_metrics.csv", index=False, encoding="utf-8-sig")
    combine_run(output, dataset_path, prices_path)
    evaluate_test(output, prices_path)
    run_backtest_research(output, prices_path, output / "research_strategy_backtest", strategy_path)
    print(f"最终配置已组装并验证，输出 {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    from datetime import datetime
    from uuid import uuid4
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prices", type=Path, default=root / "data/clean_prices.csv")
    parser.add_argument("--strategy", type=Path, default=root / "config/strategy_frozen.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    output = args.output or root / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_annualret_final_{uuid4().hex[:6]}"
    assemble_final(args.r1_run, args.r2_run, output,
                   args.dataset or next((root / "data").glob("*.zip")),
                   args.prices, args.strategy, args.device)


if __name__ == "__main__":
    main()
