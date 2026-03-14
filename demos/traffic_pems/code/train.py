"""REMX-TF: Training loop — single phase, no FAISS."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import REMXConfig
from model import REMXModel, masked_mae_loss
from data import load_data


def set_seed(seed):
    import random; random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, scaler, device, config):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_traffic = batch["x_traffic"].to(device)
            x_weather = batch["x_weather"].to(device)
            x_incident = batch["x_incident"].to(device)
            tod_idx = batch["tod_idx"].to(device)
            dow_idx = batch["dow_idx"].to(device)
            y = batch["y"].to(device)
            pred = model(x_traffic, x_weather, x_incident, tod_idx, dow_idx)
            all_preds.append(pred.cpu()); all_targets.append(y.cpu())
    preds = torch.cat(all_preds); targets = torch.cat(all_targets)
    preds_np = preds.numpy(); targets_np = targets.numpy()
    if scaler is not None:
        # scaler.mean/std shape: (1, N, 1), pred shape: (samples, N, T_out, 1)
        mean = scaler.mean.reshape(1, -1, 1, 1)
        std = scaler.std.reshape(1, -1, 1, 1)
        preds_np = preds_np * std + mean
        targets_np = targets_np * std + mean
    mask = targets_np != 0.0
    mae = np.abs(preds_np - targets_np)[mask].mean()
    rmse = np.sqrt(((preds_np - targets_np) ** 2)[mask].mean())
    mape = (np.abs(preds_np - targets_np) / (np.abs(targets_np) + 1e-8))[mask].mean() * 100
    return {"mae_avg": float(mae), "rmse_avg": float(rmse), "mape_avg": float(mape)}


def train_epoch(model, loader, optimizer, config, device, epoch):
    model.train()
    total_loss = 0.0; n_batches = 0
    for i, batch in enumerate(loader):
        x_traffic = batch["x_traffic"].to(device)
        x_weather = batch["x_weather"].to(device)
        x_incident = batch["x_incident"].to(device)
        tod_idx = batch["tod_idx"].to(device)
        dow_idx = batch["dow_idx"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        pred = model(x_traffic, x_weather, x_incident, tod_idx, dow_idx)
        loss = masked_mae_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
        if (i + 1) % config.log_interval == 0:
            print(f"  Ep{epoch} B{i+1}/{len(loader)}: loss={total_loss/n_batches:.4f}", flush=True)
    return {"loss": total_loss / n_batches}


def train(config, run_name="default"):
    set_seed(config.seed)
    device = config.get_device()
    print(f"Device: {device} | {config.dataset} d={config.d_model} mem={config.memory_size}", flush=True)
    out_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    config.save(os.path.join(out_dir, "config.json"))

    train_loader, val_loader, test_loader, scaler = load_data(config)
    model = REMXModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader), eta_min=config.lr_min)

    best_val_mae = float("inf"); best_state = None; patience_counter = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, config, device, epoch)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, scaler, device, config)
        elapsed = time.time() - t0
        print(f"Ep{epoch}/{config.epochs}: loss={train_metrics['loss']:.4f} | "
              f"val_MAE={val_metrics['mae_avg']:.4f} RMSE={val_metrics['rmse_avg']:.4f} "
              f"MAPE={val_metrics['mape_avg']:.2f}% | {elapsed:.1f}s", flush=True)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["mae_avg"] < best_val_mae:
            best_val_mae = val_metrics["mae_avg"]; patience_counter = 0
            if config.save_best:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch}", flush=True); break

    # Test
    print("\n=== TEST EVALUATION ===", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state); model = model.to(device)
    test_metrics = evaluate(model, test_loader, scaler, device, config)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)

    results = {
        "run_name": run_name, "config": config.to_dict(), "n_params": n_params,
        "test_metrics": test_metrics, "best_val_mae": best_val_mae, "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS04")
    parser.add_argument("--output_dir", default="outputs/")
    parser.add_argument("--run_name", default="remx_tf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--memory_size", type=int, default=128)
    args = parser.parse_args()
    cfg = REMXConfig(dataset=args.dataset, output_dir=args.output_dir,
                     seed=args.seed, batch_size=args.batch_size,
                     epochs=args.epochs, memory_size=args.memory_size)
    train(cfg, run_name=args.run_name)
