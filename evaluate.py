"""
evaluate.py — Validation / test loop
======================================
Runs a full pass over a DataLoader without gradient updates and
returns the mean loss, both overall and per output field.
"""

import torch
from typing import Dict, List, Tuple


@torch.no_grad()
def evaluate_epoch(model, loader, device, tokenizer,
                   max_output_len: int,
                   output_keys: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Run evaluation over a DataLoader and return (mean_loss, per_field_losses).

    Args:
        model: HAZOPModel
        loader: DataLoader
        device: torch.device
        tokenizer: T5Tokenizer
        max_output_len: int max tokens per field
        output_keys: list of field names

    Returns:
        (mean_total_loss, {field: mean_field_loss})
    """
    model.eval()

    total_loss = 0.0
    field_totals = {k: 0.0 for k in output_keys}
    n_batches = 0

    for batch in loader:
        # Move batch to device (except target_texts which are strings)
        batch_on_device = {}
        for k, v in batch.items():
            if k == "target_texts":
                batch_on_device[k] = v  # dict of lists of strings
            elif isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device)
            else:
                # PyGBatch — move all attributes
                batch_on_device[k] = v.to(device)

        loss, field_losses = model(
            batch_on_device, tokenizer, max_output_len, output_keys,
            return_field_losses=True)

        total_loss += loss.item()
        for k, v in field_losses.items():
            field_totals[k] += v
        n_batches += 1

    n = max(n_batches, 1)
    mean_total = total_loss / n
    mean_fields = {k: v / n for k, v in field_totals.items()}

    return mean_total, mean_fields
