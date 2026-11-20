"""复用 A 的价格任务，补齐交易任务、预处理和时序窗口。"""

import argparse
from hashlib import sha256
from importlib.metadata import version as package_version
import json
from operator import index as integer_index
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .features import build_features


SPLITS = ("train", "val", "test")
DROP_LGBM = ("amount_rel_20",)
DROP_RIDGE = ("ret_1", "amount_rel_20", "true_range_pct_mean_14", "close_vs_sma_20")
LONG_COLUMNS = ("ret_60", "ret_120", "close_vs_sma_60", "short_long_vol_ratio")
REGIME_START = "2020-08-24"
HALF_LIFE = 756
LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def read_frame(archive, name):
    with archive.open(name) as source:
        return pd.read_csv(
            source, encoding="utf-8-sig", index_col="date", parse_dates=["date"],
            float_precision="round_trip",
        )


def load_inputs(stage1, stage2):
    with ZipFile(stage1) as archive:
        names = [name for name in archive.namelist() if name.endswith("/outputs/tables/clean_prices.csv")]
        if len(names) != 1:
            raise ValueError("第一阶段包应包含唯一的 outputs/tables/clean_prices.csv")
        prices = read_frame(archive, names[0])

    with ZipFile(stage2) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, digest in manifest["files_sha256"].items():
            if sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"第二阶段文件校验失败：{name}")
        x = {split: read_frame(archive, f"{split}/X_{split}.csv") for split in SPLITS}
        y = {split: read_frame(archive, f"{split}/y_{split}.csv") for split in ("train", "val")}
    return prices, x, y, manifest["dataset_version"]


def label_and_check(prices, x, price_labels):
    """仅在训练和验证所允许的历史内计算带答案的标签。"""
    first_validation = x["val"].index[0]
    cutoff = x["test"].index[0]
    if not (x["train"].index[-1] < first_validation <= x["val"].index[-1] < cutoff):
        raise ValueError("训练、验证、测试信号日必须顺序排列且互不重叠。")
    dates = prices.index.to_series()
    target_dates = pd.DataFrame({
        "price_target_date": dates.shift(-1),
        "return_start_date": dates.shift(-1),
        "return_end_date": dates.shift(-2),
    })
    recent = prices.loc[:cutoff]
    close_ret = recent["close"].shift(-1) / recent["close"] - 1
    gross_open_ret = recent["open"].shift(-2) / recent["open"].shift(-1) - 1
    trade_labels = {}
    for split in ("train", "val"):
        index = x[split].index
        targets = target_dates.loc[index]
        if split == "train":
            valid_close = targets["price_target_date"] < first_validation
            valid_trade = targets["return_end_date"] < first_validation
        else:
            valid_close = targets["price_target_date"] <= cutoff
            valid_trade = targets["return_end_date"] <= cutoff
        if not valid_close.all():
            raise ValueError(f"{split} 的原有价格标签跨越边界。")
        pd.testing.assert_index_equal(index, price_labels[split].index)
        np.testing.assert_allclose(price_labels[split]["y_true_ret"], close_ret.loc[index],
                                   rtol=1e-12, atol=1e-13)
        trade = targets.loc[valid_trade, ["return_start_date", "return_end_date"]].copy()
        trade["label_trade_ret"] = gross_open_ret.loc[trade.index]
        if not np.isfinite(trade["label_trade_ret"]).all():
            raise ValueError(f"{split} 的交易标签含缺失值。")
        trade_labels[split] = trade
    return trade_labels, target_dates.loc[x["test"].index]


def add_long_features(history, close):
    history = history.copy()
    history["ret_60"] = close / close.shift(60) - 1
    history["ret_120"] = close / close.shift(120) - 1
    history["close_vs_sma_60"] = close / close.rolling(60).mean() - 1
    long_vol = history["ret_vol_20"]
    history["short_long_vol_ratio"] = history["ret_vol_5"] / long_vol.where(long_vol > 0)
    return history


def build_windows(history, signal_dates, lookback):
    """返回 (样本、交易日、特征) 的数组，每个窗口以对应信号日结尾。"""
    lookback = integer_index(lookback)
    if lookback < 1:
        raise ValueError("lookback 必须为正整数。")
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("历史索引必须唯一且按日期升序。")
    signal_dates = pd.DatetimeIndex(signal_dates)
    positions = history.index.get_indexer(signal_dates)
    invalid = np.flatnonzero(positions < lookback - 1)
    if invalid.size:
        raise ValueError(f"{signal_dates[invalid[0]].date()} 缺少完整历史窗口。")
    rows = positions[:, None] - np.arange(lookback - 1, -1, -1)
    windows = history.to_numpy(dtype=np.float64)[rows]
    if not np.isfinite(windows).all():
        raise ValueError("时序窗口含空值，请检查历史预热期和所选字段。")
    return windows


def fit_ridge_recipe(train):
    from sklearn.preprocessing import StandardScaler

    columns = train.columns.drop(list(DROP_RIDGE)).tolist()
    train = train[columns]
    lower, upper = train.quantile(0.01), train.quantile(0.99)
    scaler = StandardScaler().fit(train.clip(lower=lower, upper=upper, axis=1))
    return {
        "columns": columns,
        "clip_lower": lower.to_dict(),
        "clip_upper": upper.to_dict(),
        "mean": dict(zip(columns, scaler.mean_.tolist())),
        "scale": dict(zip(columns, scaler.scale_.tolist())),
    }


def apply_ridge_recipe(features, recipe):
    """用存档的训练集参数转换任意日期，保留原有 NaN 预热期。"""
    columns = recipe["columns"]
    clipped = features[columns].clip(lower=pd.Series(recipe["clip_lower"]),
                                     upper=pd.Series(recipe["clip_upper"]), axis=1)
    return (clipped - pd.Series(recipe["mean"])) / pd.Series(recipe["scale"])


def recency_weights(training_dates, trading_dates):
    positions = trading_dates.get_indexer(training_dates)
    weights = np.exp2((positions - positions[-1]) / HALF_LIFE)
    return pd.Series(weights / weights.mean(), index=training_dates, name="sample_weight")


def validation_comparison(features, x, price_labels, trade_labels, preprocessing_train, ridge_recipe):
    """只调用训练与验证数据。列组合和权重均为预先定义的对照。"""
    from lightgbm import LGBMRegressor, early_stopping
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rows = []
    for task, labels, field in (
        ("close_to_close", price_labels, "y_true_ret"),
        ("open_to_open", trade_labels, "label_trade_ret"),
    ):
        yt = labels["train"][field]
        yv = labels["val"][field]
        a, b = x["train"].loc[yt.index], x["val"].loc[yv.index]

        def record(name, predictions, train_rows, best_iteration=None):
            delta = yv.to_numpy() - predictions
            direction = np.mean(np.sign(predictions) == np.sign(yv.to_numpy())) if train_rows else None
            rows.append({
                "task": task, "variant": name, "train_rows": train_rows, "val_rows": len(yv),
                "rmse": np.sqrt(np.mean(delta ** 2)), "mae": np.mean(np.abs(delta)),
                "direction_accuracy": direction,
                "best_iteration": best_iteration,
            })

        record("zero", np.zeros(len(yv)), 0)
        # Ridge 只对比截尾，列和正则强度相同，避免把多个变量的改动归给截尾。
        ridge_columns = ridge_recipe["columns"]
        scaler = StandardScaler().fit(preprocessing_train[ridge_columns])
        no_clip = Ridge(alpha=5000).fit(scaler.transform(a[ridge_columns]), yt)
        record("ridge_13_standard_only", no_clip.predict(scaler.transform(b[ridge_columns])), len(yt))
        ridge = Ridge(alpha=5000).fit(apply_ridge_recipe(a, ridge_recipe), yt)
        record("ridge_13_winsor_standard", ridge.predict(apply_ridge_recipe(b, ridge_recipe)), len(yt))

        lightgbm_columns = a.columns.drop(list(DROP_LGBM)).tolist()
        available = features.loc[yt.index, list(LONG_COLUMNS)].notna().all(axis=1)
        matched_dates = yt.index[available]
        cases = (
            ("lgbm_16", lightgbm_columns, yt.index, None),
            ("lgbm_13", ridge_columns, yt.index, None),
            ("lgbm_16_recency", lightgbm_columns, yt.index, recency_weights(yt.index, features.index)),
            ("lgbm_16_matched", lightgbm_columns, matched_dates, None),
            ("lgbm_20_with_long", lightgbm_columns + list(LONG_COLUMNS), matched_dates, None),
        )
        for name, columns, dates, weights in cases:
            model = LGBMRegressor(**LGBM_PARAMS)
            xt = features.loc[dates, columns]
            xv = features.loc[yv.index, columns]
            model.fit(xt, yt.loc[dates], sample_weight=weights, eval_set=[(xv, yv)],
                      callbacks=[early_stopping(50, verbose=False)])
            record(name, model.predict(xv), len(xt), model.best_iteration_)
    return pd.DataFrame(rows)


def write_package(output, stage1, stage2):
    prices, x, y, version = load_inputs(stage1, stage2)
    history = build_features(prices)
    for frame in x.values():
        pd.testing.assert_frame_equal(frame, history.loc[frame.index], rtol=1e-12, atol=1e-13)
    trade, test_dates = label_and_check(prices, x, y)
    history = add_long_features(history, prices["close"]).loc[:x["test"].index[-1]]

    preprocessing_train = x["train"].loc[trade["train"].index]
    recipe = fit_ridge_recipe(preprocessing_train)
    lgbm_columns = x["train"].columns.drop(list(DROP_LGBM)).tolist()
    train_std, val_std = x["train"].std(), x["val"].std()
    drift = pd.DataFrame({
        "train_mean": x["train"].mean(), "train_std": train_std,
        "recent_train_mean": x["train"].loc[REGIME_START:].mean(),
        "val_mean": x["val"].mean(), "val_std": val_std, "val_train_std_ratio": val_std / train_std,
    }).rename_axis("feature")
    comparisons = validation_comparison(history, x, y, trade, preprocessing_train, recipe)
    trade_dates = trade["train"].index
    weights = pd.concat([
        recency_weights(x["train"].index, history.index).rename("close_sample_weight"),
        recency_weights(trade_dates, history.index).rename("trade_sample_weight"),
    ], axis=1)
    corr = x["train"].corr().rename_axis("feature")
    vif = pd.DataFrame({
        "raw_17": pd.Series(np.diag(np.linalg.inv(corr)), index=corr.columns),
        "ridge_13": pd.Series(np.diag(np.linalg.inv(corr.loc[recipe["columns"], recipe["columns"]])),
                              index=recipe["columns"]),
    }).rename_axis("feature")
    splits = pd.DataFrame([
        {"split": split, "signal_count": len(x[split]), "signal_start": x[split].index[0].date(),
         "signal_end": x[split].index[-1].date(), "price_labels_in_package": len(y[split]) if split in y else 0,
         "trade_labels_in_package": len(trade[split]) if split in trade else 0}
        for split in SPLITS
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix="modeling-data-", suffix=".zip", delete=False) as file:
        temp = Path(file.name)
    try:
        with ZipFile(stage2) as original, ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            hashes = {}

            def put(name, data):
                archive.writestr(name, data)
                hashes[name] = sha256(data).hexdigest()

            def csv(name, data, index=True):
                put(name, data.to_csv(date_format="%Y-%m-%d", index=index).encode("utf-8-sig"))

            for split in SPLITS:
                original_x = f"{split}/X_{split}.csv"
                put(original_x, original.read(original_x))
                csv(f"processed/{split}/X_{split}_ridge.csv", apply_ridge_recipe(x[split], recipe))
                csv(f"processed/{split}/X_{split}_lightgbm.csv", x[split][lgbm_columns])
            for split in ("train", "val"):
                original_y = f"{split}/y_{split}.csv"
                put(original_y, original.read(original_y))
                csv(f"{split}/y_trade.csv", trade[split])
            csv("test/target_dates.csv", test_dates)
            csv("history/features_history.csv", history)
            csv("train/recency_weights.csv", weights)
            csv("audit/feature_correlation.csv", corr)
            csv("audit/vif.csv", vif)
            csv("audit/train_val_drift.csv", drift)
            csv("audit/validation_comparison.csv", comparisons, index=False)
            put("split_summary.csv", original.read("split_summary.csv"))
            csv("audit/split_summary_dual.csv", splits, index=False)
            config = {
                "fit_start": str(preprocessing_train.index[0].date()),
                "fit_end": str(preprocessing_train.index[-1].date()),
                "fit_rows": len(preprocessing_train),
                "lightgbm_columns": lgbm_columns,
                "ridge": recipe,
                "long_candidate_columns": list(LONG_COLUMNS),
                "recency_half_life_trading_days": HALF_LIFE,
                "validation_models": {"lightgbm": LGBM_PARAMS, "early_stopping_rounds": 50, "ridge_alpha": 5000},
            }
            put("preprocessing.json", json.dumps(config, indent=2, allow_nan=False).encode("utf-8"))
            root = Path(__file__).resolve().parents[1]
            put("README.md", (root / "README.md").read_bytes())
            for filename in ("features.py", "modeling_data.py"):
                put(f"src/{filename}", (root / "src" / filename).read_bytes())
            manifest = {
                "dataset_version": f"{version}_dual_v2",
                "source_stage1_sha256": sha256(stage1.read_bytes()).hexdigest(),
                "source_stage2_sha256": sha256(stage2.read_bytes()).hexdigest(),
                "libraries": {name: package_version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
                "files_sha256": hashes,
            }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        with ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise OSError("输出压缩包校验失败。")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(splits.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(f"已生成 {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("第二阶段_统一建模数据_修订版.zip"))
    args = parser.parse_args()
    if args.output.resolve() in {args.stage1.resolve(), args.stage2.resolve()}:
        parser.error("输出路径须与两个输入文件不同。")
    write_package(args.output, args.stage1, args.stage2)


if __name__ == "__main__":
    main()
