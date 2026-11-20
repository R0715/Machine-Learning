"""四种模型的训练和检查点保存。"""

from dataclasses import dataclass
import json
from time import perf_counter

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    validation: np.ndarray
    test: np.ndarray | None
    best_step: int | None
    seconds: float
    peak_gpu_mib: float | None = None


def fit_tabular(name, x_train, y_train, x_val, y_val, x_test, output, seed):
    start = perf_counter()
    if name == "ridge":
        from joblib import dump
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=5000.0)
        model.fit(x_train, y_train)
        dump(model, output / "model.joblib")
        best_step = None
    else:
        from lightgbm import LGBMRegressor, early_stopping

        model = LGBMRegressor(
            objective="regression", n_estimators=500, learning_rate=0.05,
            num_leaves=31, max_depth=4, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbosity=-1, device_type="cpu",
        )
        model.fit(
            x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l2",
            callbacks=[early_stopping(50, verbose=False)],
        )
        best_step = model.best_iteration_
        # LightGBM's Windows native file writer can reject Unicode paths.
        # Write via Python and preserve the model format's original line endings.
        model_text = model.booster_.model_to_string(num_iteration=best_step)
        (output / "model.txt").write_bytes(model_text.encode("utf-8"))
    validation = model.predict(x_val)
    test = model.predict(x_test) if x_test is not None else None
    return FitResult(validation, test, best_step, perf_counter() - start)


def verify_torch(device_name, cpu_threads):
    import torch

    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"此 Python 的 PyTorch {torch.__version__} 无法使用 CUDA。"
                "本机可运行 run_5060.ps1，或显式指定 --device cpu。"
            )
        try:
            probe = torch.ones((16, 16), device=device, requires_grad=True)
            (probe @ probe).mean().backward()
            torch.cuda.synchronize(device)
        except RuntimeError as error:
            raise RuntimeError(f"CUDA 前后向自检失败：{error}") from error
        print(f"GPU {torch.cuda.get_device_name(device)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    return device


def fit_sequence(name, sequence, y_train, y_val, output, device, args):
    import torch
    from torch import nn

    is_cuda = device.type == "cuda"
    # 相同 seed 在相同环境下复现实验。跨不同 CPU/GPU 的浮点结果允许有差异。
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()

    train_windows, _, _ = sequence["train"]
    num_features, lookback = train_windows.shape[-1], train_windows.shape[1]
    if name == "tqnet":
        from .tqnet_adapter import build_tqnet

        cycle = int(args.tqnet_cycle)
        model = build_tqnet(num_features, lookback, cycle=cycle).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0)
        max_epochs = args.tqnet_epochs
        model_spec = {
            "cycle": cycle, "lookback": lookback, "num_features": num_features,
            "d_model": 32, "n_heads": 4, "attention_dropout": 0.5,
            "output_dropout": 0.1, "use_revin": False,
            "output_head": "linear across channels to scalar return",
        }
    else:
        from .patchtst_adapter import build_patchtst

        model = build_patchtst(num_features, lookback).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        max_epochs = args.epochs
        model_spec = {
            "lookback": lookback, "num_features": num_features, "encoder_layers": 3,
            "d_model": 32, "n_heads": 4, "d_ff": 64, "dropout": 0.1,
            "patch_len": min(8, lookback), "stride": max(1, min(8, lookback) // 2),
            "use_revin": False, "output_head": "linear across channels to scalar return",
        }
    optimizer_spec = {
        "name": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "betas": list(optimizer.param_groups[0]["betas"]),
        "eps": optimizer.param_groups[0]["eps"],
    }
    objective = nn.MSELoss()
    input_tensors = {
        split: (torch.as_tensor(windows, dtype=torch.float32, device=device),
                torch.as_tensor(positions, dtype=torch.int64, device=device))
        for split, (windows, positions, _) in sequence.items()
    }
    x_train, positions = input_tensors["train"]
    target_mean = float(y_train.mean())
    target_std = max(float(y_train.std(ddof=0)), 1e-8)
    target = torch.as_tensor(
        (y_train.to_numpy() - target_mean) / target_std, dtype=torch.float32, device=device,
    )
    best_loss, best_epoch, best_state = float("inf"), 0, None
    batch_size = args.batch_size
    history = []
    print(
        f"  {name} optimizer={optimizer_spec['name']} max_epochs={max_epochs} "
        f"lookback={lookback}" + (f" cycle={cycle}" if name == "tqnet" else " encoder_layers=3"),
        flush=True,
    )

    @torch.no_grad()
    def predict(split):
        model.eval()
        values, dates = input_tensors[split]
        batches = [
            model(values[start:start + batch_size], dates[start:start + batch_size])
            for start in range(0, len(values), batch_size)
        ]
        return torch.cat(batches).cpu().numpy().astype(np.float64) * target_std + target_mean

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=device, generator=generator)
        losses = []
        for rows in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x_train[rows], positions[rows])
            loss = objective(outputs, target[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())

        prediction = predict("val")
        rmse = float(np.sqrt(np.mean((prediction - y_val.to_numpy()) ** 2)))
        if not np.isfinite(rmse):
            raise RuntimeError(f"{name} 第 {epoch} 轮出现非有限验证误差。")
        if rmse < best_loss:
            best_loss, best_epoch = rmse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_loss = torch.stack(losses).mean().item()
        history.append({
            "epoch": epoch, "train_mse_scaled": train_loss, "val_rmse": rmse,
            "is_best_so_far": epoch == best_epoch,
        })
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
        if epoch == 1 or epoch % 5 == 0 or epoch == max_epochs:
            print(f"  {name} epoch={epoch:02d} train_mse_scaled={train_loss:.6f} val_rmse={rmse:.6f}", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"  早停，最佳 epoch={best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    validation = predict("val")
    test = predict("test") if "test" in input_tensors else None
    torch.save({
        "state_dict": best_state,
        "model": name,
        "num_features": num_features,
        "lookback": lookback,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "max_epochs": max_epochs,
        "optimizer": optimizer_spec,
        "model_spec": model_spec,
        "training_args": json.loads(json.dumps(vars(args), default=str)),
    }, output / "model.pt")
    print(f"  {name} completed_epochs={epoch} best_epoch={best_epoch}", flush=True)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if is_cuda else None
    return FitResult(validation, test, best_epoch, perf_counter() - start, peak)
