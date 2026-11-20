"""创业板指日线原始特征。每行按该交易日收盘时可用的数据计算。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close"]
ACTIVITY_COLUMNS = ["volume", "amount"]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """接收 A 的清洗行情，返回同日期、同顺序的无量纲特征。

    滚动窗口按交易日计数，样本不足的行保留 NaN。特征表可按日期与标签对齐。
    """
    index = prices.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
    ):
        raise ValueError("行情必须使用无缺失、唯一且升序的 DatetimeIndex。")

    columns = PRICE_COLUMNS + ACTIVITY_COLUMNS
    missing_columns = [column for column in columns if column not in prices.columns]
    if missing_columns:
        raise ValueError(f"行情缺少列：{', '.join(missing_columns)}")
    data = prices.loc[:, columns].astype(float)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("行情数值必须有限。")
    if (data[PRICE_COLUMNS] <= 0).any().any() or (data[ACTIVITY_COLUMNS] < 0).any().any():
        raise ValueError("价格必须为正，成交量和成交额必须非负。")

    opening, high, low, close, volume, amount = (data[column] for column in columns)
    body_high = np.maximum(opening, close)
    body_low = np.minimum(opening, close)
    if (high < body_high).any() or (low > body_low).any():
        raise ValueError("最高价和最低价必须覆盖开盘价与收盘价。")

    prev_close = close.shift(1)
    daily_return = close / prev_close - 1
    day_range = high - low

    # 动量趋势：收益率及相对均线偏离。
    features = {
        "ret_1": daily_return,
        "ret_5": close / close.shift(5) - 1,
        "ret_20": close / close.shift(20) - 1,
        "close_vs_sma_20": close / close.rolling(20).mean() - 1,
    }

    # 震荡：简单滚动 RSI 和随机指标 %K，均表示为 0 到 1 的比例。
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(14).sum()
    losses = (-changes.clip(upper=0)).rolling(14).sum()
    movement = gains + losses
    rsi = (gains / movement.where(movement > 0)).mask(movement.eq(0), 0.5)
    range_low = low.rolling(14).min()
    range_width = high.rolling(14).max() - range_low
    stochastic = ((close - range_low) / range_width.where(range_width > 0)).mask(
        range_width.eq(0), 0.5
    )
    features.update(rsi_sma_14=rsi, stochastic_k_14=stochastic)

    # 波动率：收盘收益率的标准差及平均相对真实振幅。
    true_range = np.maximum(high, prev_close) - np.minimum(low, prev_close)
    features.update(
        ret_vol_5=daily_return.rolling(5).std(ddof=0),
        ret_vol_20=daily_return.rolling(20).std(ddof=0),
        true_range_pct_mean_14=(true_range / prev_close).rolling(14).mean(),
    )

    # 量价：当日量额相对前 20 日均值，加上近 10 日带方向的成交量占比。
    prev_volume_mean = volume.shift(1).rolling(20).mean()
    prev_amount_mean = amount.shift(1).rolling(20).mean()
    signed_volume = np.sign(changes) * volume
    volume_sum = volume.rolling(10).sum()
    features.update(
        volume_rel_20=volume / prev_volume_mean.where(prev_volume_mean > 0) - 1,
        amount_rel_20=amount / prev_amount_mean.where(prev_amount_mean > 0) - 1,
        signed_volume_share_10=signed_volume.rolling(10).sum() / volume_sum.where(volume_sum > 0),
    )

    # K 线结构：隔夜跳空、日内涨跌、振幅及上下影线。
    nonzero_range = day_range.where(day_range > 0)
    features.update(
        gap_ret=opening / prev_close - 1,
        intraday_ret=close / opening - 1,
        range_vs_open=day_range / opening,
        upper_wick_share=((high - body_high) / nonzero_range).mask(day_range.eq(0), 0.0),
        lower_wick_share=((body_low - low) / nonzero_range).mask(day_range.eq(0), 0.0),
    )
    return pd.DataFrame(features, index=index.rename("date"))


def main() -> None:
    parser = argparse.ArgumentParser(description="从第一阶段清洗行情生成 B 的原始特征表。")
    parser.add_argument("--input", type=Path, default=Path("outputs/tables/clean_prices.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/raw_features.csv"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("输入与输出路径必须不同。")

    prices = pd.read_csv(
        args.input, encoding="utf-8-sig", parse_dates=["date"], float_precision="round_trip"
    ).set_index("date")
    features = build_features(prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"已生成 {len(features):,} 行 × {features.shape[1]} 列特征：{args.output}")


if __name__ == "__main__":
    main()
