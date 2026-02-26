"""Debug: run the training loop manually with full error capture."""
import traceback
import sys
import os

# Force stdout for all output
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")

try:
    import torch
    from app.services.train_gnn import load_dataset_by_source, _samples_to_pyg
    from app.services.gnn_engine import FloorPlanGNN, compute_loss
    from torch.optim import Adam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load a small subset to debug
    print("Loading 50 samples from combined...")
    samples = load_dataset_by_source("combined", "data", synthetic_samples=20, max_samples=10)
    print(f"Loaded {len(samples)} total samples")

    # Split
    split = int(len(samples) * 0.8)
    train_s = samples[:split]
    val_s = samples[split:]
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    # Convert to PyG (on CPU)
    cpu = torch.device("cpu")
    print("Converting to PyG...")
    train_data = _samples_to_pyg(train_s, cpu)
    val_data = _samples_to_pyg(val_s, cpu)
    print(f"PyG: train={len(train_data)}, val={len(val_data)}")

    if not train_data:
        print("ERROR: No training data after conversion!")
        sys.exit(1)

    # Model
    print("Creating model...")
    model = FloorPlanGNN().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    # One training step
    print("Running one training step...")
    model.train()
    data, gt_bboxes = train_data[0]
    data = data.to(device)
    gt_bboxes = gt_bboxes.to(device)
    print(f"  Input: x={data.x.shape}, edges={data.edge_index.shape}, gt={gt_bboxes.shape}")

    optimizer.zero_grad()
    bbox_pred, door_pred = model(data.x, data.edge_index, data.edge_attr)
    print(f"  Output: bbox={bbox_pred.shape}, door={door_pred.shape}")

    losses = compute_loss(bbox_pred, gt_bboxes)
    print(f"  Loss: {losses['total'].item():.4f}")

    losses["total"].backward()
    optimizer.step()
    print("  Backward + step: OK ✓")

    # Run 2 full epochs
    print("\nRunning 2 full epochs...")
    for epoch in range(2):
        model.train()
        train_losses = []
        for i, (data, gt_bboxes) in enumerate(train_data):
            data = data.to(device)
            gt_bboxes = gt_bboxes.to(device)
            optimizer.zero_grad()
            bbox_pred, door_pred = model(data.x, data.edge_index, data.edge_attr)
            losses = compute_loss(bbox_pred, gt_bboxes)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(losses["total"].item())

        avg = sum(train_losses) / len(train_losses)
        print(f"  Epoch {epoch+1}: train_loss={avg:.4f}")

    print("\n✅ Training works!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
