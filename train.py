"""
train.py — Training loop for HAZOP GNN+T5 model
================================================
Trains a model to generate HAZOP analysis text (deviation, causes, consequences,
safeguards, recommendations) for process safety hazard analysis.

Usage:
    python train.py \
        --data_dir ./data \
        --output_dir ./checkpoints \
        --t5_model t5-small \
        --batch_size 8 \
        --epochs 50 \
        --lr 1e-4
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import HAZOPDataset, hazop_collate_fn, OUTPUT_KEYS
from model import HAZOPModel
from evaluate import evaluate_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train HAZOP GNN+T5 model for process safety analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --data_dir ./data
  python train.py --data_dir ./data --batch_size 16 --t5_model t5-base
  python train.py --data_dir ./data --epochs 100 --patience 7
        """)
    p.add_argument("--data_dir", type=str, default="./data",
                   help="Path to data directory containing system_* folders")
    p.add_argument("--output_dir", type=str, default="./checkpoints",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--t5_model", type=str, default="t5-small",
                   help="T5 model name (t5-small, t5-base, t5-large)")
    p.add_argument("--gnn_hidden", type=int, default=128,
                   help="GNN hidden dimension")
    p.add_argument("--gnn_layers", type=int, default=2,
                   help="Number of GNN layers")
    p.add_argument("--gnn_out", type=int, default=64,
                   help="GNN output dimension")
    p.add_argument("--gw_embed_dim", type=int, default=32,
                   help="Guideword embedding dimension")
    p.add_argument("--max_output_len", type=int, default=128,
                   help="Max tokens per output field")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for training")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience")
    p.add_argument("--train_split", type=float, default=0.70,
                   help="Fraction of systems for training")
    p.add_argument("--val_split", type=float, default=0.15,
                   help="Fraction of systems for validation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--freeze_t5_epochs", type=int, default=5,
                   help="Number of epochs to freeze T5 weights")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = no multiprocessing)")
    p.add_argument("--grad_accumulate_steps", type=int, default=1,
                   help="Gradient accumulation steps (for effective larger batches)")
    return p.parse_args()


def train_one_epoch(model, loader, optimiser, device, tokenizer,
                    max_output_len, output_keys, grad_accumulate_steps=1):
    """
    Run one training epoch.

    Args:
        model: HAZOPModel
        loader: DataLoader
        optimiser: AdamW optimizer
        device: torch.device
        tokenizer: T5Tokenizer
        max_output_len: int
        output_keys: list of field names
        grad_accumulate_steps: int for gradient accumulation

    Returns:
        mean loss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    accumulation_count = 0

    pbar = tqdm(loader, desc="Training Batches", leave=False)

    for batch_idx, batch in enumerate(pbar):

        # Move batch to device
        batch_on_device = {}
        for k, v in batch.items():
            if k == "target_texts":
                batch_on_device[k] = v  # dict of lists
            elif isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device)
            else:
                batch_on_device[k] = v.to(device)

        # Forward pass
        loss = model(batch_on_device, tokenizer, max_output_len, output_keys)

        # Scale loss for gradient accumulation
        loss = loss / grad_accumulate_steps
        loss.backward()

        accumulation_count += 1

        # Optimizer step after accumulation
        if accumulation_count % grad_accumulate_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()
            accumulation_count = 0

        # Track total loss (rescale back)
        total_loss += loss.item() * grad_accumulate_steps
        n_batches += 1

        # Update tqdm progress bar
        current_loss = loss.item() * grad_accumulate_steps
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Old-style batch logging (every 5 batches)
        if batch_idx % 5 == 0:
            log.info(f"[Batch {batch_idx+1}/{len(loader)}] Loss: {current_loss:.4f}")

    # Final gradient update if leftover batches
    if accumulation_count > 0:
        log.info(f"Final optimizer step with remaining {accumulation_count} batches")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        optimiser.zero_grad()

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    
    # ── Setup ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 70)
    log.info("HAZOP GNN+T5 Training")
    log.info("=" * 70)
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        log.info("CUDA Memory: %.1f GB",
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save hyperparameters
    hparams_path = os.path.join(args.output_dir, "hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Saved hyperparameters to: %s", hparams_path)

    # ── Load tokenizer ───────────────────────────────────────────────────
    log.info("Loading T5 tokenizer: %s", args.t5_model)
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)

    # ── Load dataset ─────────────────────────────────────────────────────
    log.info("Loading dataset from: %s", args.data_dir)
    full_dataset = HAZOPDataset(args.data_dir, tokenizer)

    # ── Train/val/test split by system (not instance!) ──────────────────
    system_ids = full_dataset.system_ids
    n_total = len(system_ids)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    n_test = n_total - n_train - n_val

    torch.manual_seed(args.seed)
    perm = torch.randperm(n_total).tolist()
    train_ids = [system_ids[i] for i in perm[:n_train]]
    val_ids = [system_ids[i] for i in perm[n_train:n_train + n_val]]
    test_ids = [system_ids[i] for i in perm[n_train + n_val:]]

    log.info("System split — train:%d  val:%d  test:%d", n_train, n_val, n_test)

    train_ds = full_dataset.subset(train_ids)
    val_ds = full_dataset.subset(val_ids)
    test_ds = full_dataset.subset(test_ids)

    log.info("Instance split — train:%d  val:%d  test:%d",
             len(train_ds), len(val_ds), len(test_ds))

    # ── Create DataLoaders ───────────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=hazop_collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"))

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=hazop_collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"))

    # ── Model setup ──────────────────────────────────────────────────────
    node_feat_dim = full_dataset.node_feat_dim
    num_guidewords = full_dataset.num_guidewords
    output_keys = full_dataset.output_keys

    log.info("Node features: %d  Guidewords: %d  Output fields: %s",
             node_feat_dim, num_guidewords, output_keys)

    model = HAZOPModel(
        node_feat_dim=node_feat_dim,
        gnn_hidden=args.gnn_hidden,
        gnn_out=args.gnn_out,
        gnn_layers=args.gnn_layers,
        num_guidewords=num_guidewords,
        gw_embed_dim=args.gw_embed_dim,
        t5_model_name=args.t5_model,
        output_keys=output_keys,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Total parameters: %s  Trainable: %s",
             f"{total_params:,}", f"{trainable_params:,}")

    # ── Optimizer and scheduler ──────────────────────────────────────────
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2, verbose=True)

    # ── Training loop ────────────────────────────────────────────────────
    if args.freeze_t5_epochs > 0:
        log.info("Phase 1: Freezing T5 for %d epochs (train GNN + prefix only)",
                 args.freeze_t5_epochs)
        model.freeze_t5(True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    log.info("=" * 70)
    log.info("Starting training (%d epochs, batch_size=%d)",
             args.epochs, args.batch_size)
    log.info("=" * 70)

    for epoch in range(1, args.epochs + 1):

        # Switch to joint training
        if epoch == args.freeze_t5_epochs + 1 and args.freeze_t5_epochs > 0:
            log.info("=" * 70)
            log.info("Phase 2: Unfreezing T5 — joint training begins")
            log.info("=" * 70)
            model.freeze_t5(False)

        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimiser, device,
            tokenizer, args.max_output_len, output_keys,
            grad_accumulate_steps=args.grad_accumulate_steps)

        # Validation
        val_loss, field_losses = evaluate_epoch(
            model, val_loader, device, tokenizer,
            args.max_output_len, output_keys)

        scheduler.step(val_loss)

        # Logging
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            **{f"val_{k}": round(v, 4) for k, v in field_losses.items()},
        }
        history.append(row)

        field_str = "  ".join(f"{k}={v:.3f}" for k, v in field_losses.items())
        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  %s",
            epoch, args.epochs, train_loss, val_loss, field_str)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)
            log.info("  → Saved best checkpoint (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            log.info("  → No improvement. Patience: %d/%d",
                     patience_counter, args.patience)
            if patience_counter >= args.patience:
                log.info("  → Early stopping triggered")
                break

    # ── Save training history ────────────────────────────────────────────
    history_path = os.path.join(args.output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training complete. Saved history to: %s", history_path)
    log.info("Best validation loss: %.4f", best_val_loss)

    # ── Test evaluation ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Evaluating on test set …")
    log.info("=" * 70)

    ckpt = torch.load(
        os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=hazop_collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"))

    test_loss, test_field_losses = evaluate_epoch(
        model, test_loader, device, tokenizer,
        args.max_output_len, output_keys)

    field_str = "  ".join(f"{k}={v:.4f}" for k, v in test_field_losses.items())
    log.info("Test loss: %.4f  %s", test_loss, field_str)

    test_results_path = os.path.join(args.output_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump({"test_loss": test_loss, **test_field_losses}, f, indent=2)
    log.info("Saved test results to: %s", test_results_path)
    
    log.info("=" * 70)
    log.info("Training pipeline complete!")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
