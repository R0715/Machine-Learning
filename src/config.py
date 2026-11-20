from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # 课程规定的边界；val_start、horizon 和均线周期为可调整的小组约定。
    cutoff: str = "2025-10-10"
    end: str = "2026-10-23"
    val_start: str = "2024-10-11"
    horizon: int = 1
    initial_cash: float = 100_000.0
    short_window: int = 5
    long_window: int = 20
    # 小组规定的股票模拟税费；不另叠加已包含在佣金中的规费。
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    transfer_fee_rate: float = 0.00001
    stamp_tax_rate: float = 0.0005

