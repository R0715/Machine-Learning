"""Use training-only autocorrelation to select TQNet's common cycle."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


NOTEBOOK_URL = (
    "https://github.com/ACAT-SCUT/TQNet/blob/"
    "15e19cb23483ed52398566c4baa959168cfffa57/acf_plot.ipynb"
)
NOTEBOOK_SHA256 = "e71bc1ef6414742715b4c04c9871b30cea3f044047e3746bc71241679054886f"


def _acf(values, max_lag):
    centered = values - values.mean(axis=0)
    energy = np.sum(centered ** 2, axis=0)
    result = np.zeros((max_lag + 1, values.shape[1]), dtype=np.float64)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        numerator = np.sum(centered[lag:] * centered[:-lag], axis=0)
        result[lag] = np.divide(numerator, energy, out=np.zeros_like(energy), where=energy > 0)
    return result


def _thresholds(acf, samples, critical):
    # Bartlett's approximate ACF error scale, made more conservative for the
    # many lags/channels examined. This is a screening rule, not a cycle proof.
    previous_squared = np.zeros_like(acf)
    previous_squared[2:] = np.cumsum(acf[1:-1] ** 2, axis=0)
    return critical * np.sqrt((1.0 + 2.0 * previous_squared) / samples)


def _peak(acf, threshold, lag, channel):
    positions = np.arange(max(2, lag - 1), min(len(acf) - 2, lag + 1) + 1)
    valid = (
        (acf[positions, channel] > threshold[positions, channel])
        & (acf[positions, channel] > acf[positions - 1, channel])
        & (acf[positions, channel] >= acf[positions + 1, channel])
    )
    return bool(valid.any())


def select_cycle(train_features: pd.DataFrame, output_dir: Path) -> int:
    """Select a stable cycle or return W=1 when no common cycle is supported.

    The caller passes only the actual training signal dates. Validation and
    test features are never loaded here. Official TQNet's notebook plots raw
    training ACF; its visual choice is automated here with a conservative,
    separately documented persistence filter and split-half confirmation.
    """
    if (
        not isinstance(train_features.index, pd.DatetimeIndex)
        or train_features.empty or train_features.index.hasnans
        or not train_features.index.is_unique or not train_features.index.is_monotonic_increasing
    ):
        raise ValueError("ACF requires nonempty training features with unique, ordered trading dates.")
    values = train_features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or len(values) < 32:
        raise ValueError("ACF requires finite training features and at least 32 trading days.")
    active = np.var(values, axis=0) > 0
    if not active.any():
        raise ValueError("ACF needs at least one nonconstant training feature.")
    max_lag = min(252, len(values) // 4)
    raw_acf = _acf(values, max_lag)
    differences = np.diff(values[:, active], axis=0)
    middle = len(differences) // 2
    blocks = (differences, differences[:middle], differences[middle:])
    acfs = [_acf(block, max_lag) for block in blocks]
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * max_lag * int(active.sum()) * len(blocks)))
    thresholds = [_thresholds(acf, len(block), critical) for acf, block in zip(acfs, blocks)]
    required_support = max(1, int(np.ceil(active.sum() / 2)))
    candidates = []
    for period in range(2, (max_lag - 1) // 2 + 1):
        supporting = [
            channel for channel in range(int(active.sum()))
            if all(
                _peak(acf, threshold, period, channel) and _peak(acf, threshold, 2 * period, channel)
                for acf, threshold in zip(acfs, thresholds)
            )
        ]
        strength = float(np.median(acfs[0][period, supporting])) if supporting else 0.0
        candidates.append({
            "period": period, "supporting_channels": len(supporting),
            "median_differenced_acf": strength,
            "eligible": len(supporting) >= required_support,
        })
    scoring = pd.DataFrame(candidates)
    eligible = scoring.loc[scoring["eligible"]].sort_values(
        ["supporting_channels", "median_differenced_acf", "period"],
        ascending=[False, False, True],
    )
    cycle = int(eligible.iloc[0]["period"]) if len(eligible) else 1
    table = pd.DataFrame(raw_acf, columns=[f"raw__{name}" for name in train_features.columns])
    active_columns = train_features.columns[active]
    for name, acf in zip(("difference_full", "difference_first_half", "difference_second_half"), acfs):
        for channel, column in enumerate(active_columns):
            table[f"{name}__{column}"] = acf[:, channel]
    table.index.name = "lag"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "acf_values.csv", encoding="utf-8-sig")
    scoring.to_csv(output_dir / "cycle_candidates.csv", index=False, encoding="utf-8-sig")
    report = {
        "selected_cycle": cycle,
        "periodic_evidence_found": bool(len(eligible)),
        "fallback_reason": None if len(eligible) else "No stable repeated peak shared by at least half the channels in both training halves. W=1 supplies a constant temporal query without an assumed periodic shift.",
        "training_start": train_features.index[0].strftime("%Y-%m-%d"),
        "training_end": train_features.index[-1].strftime("%Y-%m-%d"),
        "training_rows": len(train_features),
        "training_columns": train_features.columns.tolist(),
        "training_data_sha256": sha256(train_features.to_csv(date_format="%Y-%m-%d").encode("utf-8")).hexdigest(),
        "source_notebook": NOTEBOOK_URL,
        "source_notebook_sha256": NOTEBOOK_SHA256,
        "source_method": "The official notebook displays biased demeaned ACF of selected raw training channels. It does not prescribe an automatic peak-selection algorithm.",
        "automation": "Compute all raw channel ACFs. Use first differences to reduce the mechanical persistence of rolling indicators. Require positive local peaks within +/-1 lag at W and 2W in the full training sample and both halves. Choose most supported candidate, then largest median ACF, then smaller W.",
        "max_acf_lag": max_lag,
        "candidate_period_min": 2,
        "candidate_period_max": (max_lag - 1) // 2,
        "required_channel_support": required_support,
        "critical_normal_value": critical,
        "threshold_method": "Approximate Bartlett error scale with 0.05 Bonferroni screening across lags, channels and three blocks. Correlated features mean channel votes are not independent.",
        "interpretation": "A conservative training-only heuristic for this dataset, not a hypothesis test establishing a true financial cycle.",
    }
    (output_dir / "cycle_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        lags = np.arange(max_lag + 1)
        axes[0].plot(lags, np.median(raw_acf[:, active], axis=1), color="#245b83")
        axes[0].set_ylabel("Median raw ACF")
        axes[1].plot(lags, np.median(acfs[0], axis=1), color="#245b83")
        axes[1].plot(lags, np.median(thresholds[0], axis=1), "--", color="#9b5639", label="screening threshold")
        axes[1].set_ylabel("Median differenced ACF")
        axes[1].set_xlabel("Lag in trading days")
        axes[1].legend()
        for axis in axes:
            axis.set_xlim(1, max_lag)
            axis.grid(alpha=0.2)
            if cycle > 1:
                axis.axvline(cycle, color="#9b5639", alpha=0.6)
        fig.suptitle(f"Training-only cycle selection: W={cycle}")
        fig.tight_layout()
        fig.savefig(output_dir / "acf.png", dpi=150)
        plt.close(fig)
    return cycle
