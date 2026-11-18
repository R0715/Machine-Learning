"""A 第二阶段：接收 B 的原始特征，配未来收益标签，按时间切分。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from src.config import Config
from src.dataset import build_dataset
from src.features import build_features

ROOT = Path(__file__).resolve().parent


def read_table(path):
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"],
                       float_precision="round_trip").set_index("date")


def write_json(value, path):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon",type=int,default=1)
    parser.add_argument("--val-start",default="2024-10-11")
    parser.add_argument("--output",type=Path,default=ROOT/"dataset")
    args = parser.parse_args()
    cfg = Config(horizon=args.horizon,val_start=args.val_start)
    prices_path, features_path = ROOT/"outputs/tables/clean_prices.csv", ROOT/"outputs/tables/raw_features.csv"
    prices, features = read_table(prices_path), read_table(features_path)
    # 在导出之前校验 B 的 CSV 与其生成代码是否相符，避免拿错版本。
    pd.testing.assert_frame_equal(features,build_features(prices),check_exact=False,rtol=1e-12,atol=1e-14)
    data = build_dataset(prices,features,cfg)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    summary, assigned = [], pd.Series("",index=features.index,dtype="str")
    for split in ("train","val","test"):
        x,y = getattr(data,f"X_{split}"),getattr(data,f"y_{split}")
        folder = output/split
        folder.mkdir(exist_ok=True)
        metadata = data.metadata.loc[x.index]
        x.to_csv(folder/f"X_{split}.csv",encoding="utf-8-sig",index_label="date")
        y.to_csv(folder/f"y_{split}.csv",encoding="utf-8-sig",index_label="date")
        metadata.to_csv(folder/f"metadata_{split}.csv",encoding="utf-8-sig",index_label="date")
        assigned.loc[x.index] = split
        summary.append({"split":split,"rows":len(x),"feature_columns":len(x.columns),
                        "signal_start":str(x.index[0].date()),"signal_end":str(x.index[-1].date()),
                        "target_start":str(metadata.target_date.min().date()),
                        "target_end":str(metadata.target_date.max().date()),
                        "known_labels":int(y.notna().sum())})
    excluded = data.metadata.loc[assigned.eq("")].copy()
    reasons=[]
    for date in excluded.index:
        if features.loc[date].isna().any():
            reason="warmup_incomplete_features"
        elif date >= pd.Timestamp(cfg.end):
            reason="end_date_no_next_execution"
        elif date < pd.Timestamp(cfg.val_start):
            reason="label_crosses_train_val_boundary"
        elif date < pd.Timestamp(cfg.cutoff):
            reason="label_crosses_historical_cutoff"
        else:
            reason="outside_selected_period"
        reasons.append(reason)
    excluded["reason"] = reasons
    excluded.to_csv(output/"excluded_rows.csv",encoding="utf-8-sig",index_label="date")
    pd.DataFrame(summary).to_csv(output/"split_summary.csv",encoding="utf-8-sig",index=False)
    # 所有行应恰好进入一个集合或排除清单，不能重叠，也不能不明原因丢失。
    assert sum(r["rows"] for r in summary)+len(excluded)==len(features)
    audit = {**data.audit,"splits":summary,"excluded_rows":len(excluded),
             "excluded_by_reason":excluded.reason.value_counts().to_dict(),
             "feature_order":features.columns.tolist(),"csv_matches_B_code":True,
             "preprocessing":"raw_features_only_no_scaling_no_selection",
             "test_label_policy":"仅供最终评估，禁止用于模型训练、选参、特征筛选和策略调参",
             "target_formula":"close[t+H]/close[t]-1; H counts trading rows",
             "return_warning":"标签为收盘到收盘收益，不等于按次日开盘成交的账户收益",
             "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
             "config":asdict(cfg),"source_sha256":{
                 "clean_prices.csv":hashlib.sha256(prices_path.read_bytes()).hexdigest(),
                 "raw_features.csv":hashlib.sha256(features_path.read_bytes()).hexdigest()}}
    write_json(audit,output/"split_audit.json")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"排除 {len(excluded)} 行：{audit['excluded_by_reason']}")
    print(f"输出位置：{output}")


if __name__ == "__main__":
    main()
