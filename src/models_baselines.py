"""四基准模型的检查点预测入口，识别单模型与本地成员集成。"""

import json
from pathlib import Path

import numpy as np

from .data import load_dataset


MODELS = ("ridge", "lightgbm", "tqnet", "patchtst")


def predict_saved(name, task, run, dataset_path, split="test", dates=None, device=None):
    """返回与信号日期同序的一维收益预测，不读取测试答案。

    trade 预测下一开盘到再下一开盘收益。close 预测下一收盘相对当前收盘收益。
    ensemble.json 存在时始终平均全部成员，不能用 model.pt/txt 替代集成。
    """
    from export_predictions import restore_predict, saved_config

    if name not in MODELS or task not in ("close", "trade") or split not in ("train", "val", "test"):
        raise ValueError("模型、任务或数据分组无效。")
    run, dataset_path = Path(run), Path(dataset_path)
    component = run / ("cpu" if name in ("ridge", "lightgbm") else "gpu")
    folder = component / task / name
    config = saved_config(component, dataset_path)
    dataset = load_dataset(dataset_path)
    if dates is None:
        dates = dataset.raw[split].index if split == "test" else dataset.labels[task][split].index
    ensemble_file = folder / "ensemble.json"
    if ensemble_file.exists():
        members = json.loads(ensemble_file.read_text(encoding="utf-8"))["members"]
        if not members:
            raise ValueError("集成成员不能为空。")
        files = []
        for member in members:
            path = (folder / member["file"]).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError(f"集成成员无效：{member['file']}")
            files.append(path)
    else:
        files = [folder / {"ridge": "model.joblib", "lightgbm": "model.txt"}.get(name, "model.pt")]
    predictions = [restore_predict(name, folder, config, dataset, split, dates, device, model_file=file)
                   for file in files]
    result = np.mean(predictions, axis=0)
    if result.shape != (len(dates),) or not np.isfinite(result).all():
        raise ValueError("预测长度错误或包含非有限值。")
    return result
