"""检查整理后的数据、最终预测、模型归档和来源，不训练或改写结果。"""

from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "runs/20261119_233827_annualret_final_cd62b2"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify():
    checked = 0
    source = json.loads((ROOT / "docs/delivery_20261120/provenance.json").read_text(encoding="utf-8"))
    for item in source["files"]:
        expected = item.get("organized_sha256", item["sha256"])
        assert digest(ROOT / item["path"]) == expected, item["path"]
        checked += 1
    revised = ROOT / "dataset/revised"
    manifest = json.loads((revised / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        assert digest(revised / name) == expected, name
        checked += 1
    archive = ROOT / "data/第二阶段_统一建模数据_修订版.zip"
    assert digest(archive) == source["source_archives"][1]["sha256"]
    with zipfile.ZipFile(archive) as file:
        for name in file.namelist():
            if not name.endswith("/"):
                assert file.read(name) == (revised / name).read_bytes(), name
                checked += 1
    assert digest(ROOT / "data/clean_prices.csv") == digest(ROOT / "outputs/tables/clean_prices.csv")
    for split, count in [("train", 3465), ("val", 242), ("test", 251)]:
        raw = pd.read_csv(revised / split / f"X_{split}.csv", index_col="date", parse_dates=["date"])
        assert raw.shape == (count, 17)
        assert raw.index.is_unique and raw.index.is_monotonic_increasing
        for suffix, width in [("ridge", 13), ("lightgbm", 16)]:
            features = pd.read_csv(revised / "processed" / split / f"X_{split}_{suffix}.csv", index_col="date", parse_dates=["date"])
            assert features.shape == (count, width)
            assert features.index.equals(raw.index)
            assert np.isfinite(features.to_numpy()).all()
    delivery = ROOT / "outputs/preds"
    pred_manifest = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))
    assert pred_manifest["dataset_sha256"] == digest(archive)
    assert pred_manifest["prices_sha256"] == digest(ROOT / "data/clean_prices.csv")
    for name, expected in pred_manifest["files_sha256"].items():
        assert digest(delivery / name) == expected, name
        assert (delivery / name).read_bytes() == (FINAL / "handoff" / name).read_bytes(), name
        checked += 1
    expected_columns = ["model_id", "date", "train_asof_date", "close", "price_target_date",
                        "pred_close_ret", "pred_close", "return_start_date", "return_end_date", "pred_trade_ret"]
    dates = pd.read_csv(revised / "test/X_test.csv")["date"]
    for model in ("ridge", "lightgbm", "tqnet", "patchtst"):
        frame = pd.read_csv(delivery / f"{model}_predictions.csv", float_precision="round_trip")
        assert list(frame.columns) == expected_columns and len(frame) == 251
        assert frame.date.equals(dates) and frame.date.is_unique
        assert frame.train_asof_date.eq("2025-10-10").all()
        assert frame.return_end_date.isna().sum() == 1 and pd.isna(frame.return_end_date.iloc[-1])
        assert np.isfinite(frame[["close", "pred_close_ret", "pred_close", "pred_trade_ret"]].to_numpy()).all()
        np.testing.assert_allclose(frame.pred_close, frame.close * (1 + frame.pred_close_ret), atol=1e-8)
        for task in ("close", "trade"):
            group = "cpu" if model in ("ridge", "lightgbm") else "gpu"
            folder = FINAL / group / task / model
            ensemble = folder / "ensemble.json"
            if ensemble.exists():
                for member in json.loads(ensemble.read_text(encoding="utf-8"))["members"]:
                    assert (folder / member["file"]).is_file()
    for name in ("test_metrics.csv", "validation_metrics.csv", "final_selection.json"):
        assert (ROOT / "outputs/metrics" / name).read_bytes() == (FINAL / name).read_bytes(), name
    assert (ROOT / "outputs/metrics/strategy_metrics.csv").read_bytes() == (FINAL / "research_strategy_backtest/strategy_metrics.csv").read_bytes()
    metrics = pd.read_csv(ROOT / "outputs/metrics/strategy_metrics.csv").set_index("model")
    expected_assets = {"ridge": 106742.27656118639, "lightgbm": 129582.50657368197,
                       "tqnet": 116192.5086056799, "patchtst": 101223.83513184862}
    for name, assets in expected_assets.items():
        assert abs(metrics.loc[name, "final_equity"] - assets) < 1e-7
    print(f"PASS: {checked} source/hash/archive checks; revised splits and processed inputs aligned; 4 final prediction tables (251 x 10); ensemble members and original results intact.")


if __name__ == "__main__":
    verify()
