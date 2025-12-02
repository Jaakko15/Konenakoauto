# train_custom.py
# Fine-tune YOLO (Ultralytics) with tuned hypers, resume support, and end-of-run validation.
# Works on Windows paths. Requires: pip install ultralytics

from ultralytics import YOLO
import argparse
import os
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO with tuned hyperparameters")
    p.add_argument("--model", type=str, required=True,
                   help=r'Path to starting weights (e.g. "C:\...\runs\detect\train3\weights\best.pt") or a model name like "yolo11m.pt"')
    p.add_argument("--data", type=str, required=True,
                   help=r'Path to data.yaml (use absolute paths inside YAML for train/val/test)')
    p.add_argument("--epochs", type=int, default=60, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=1280, help="Training image size")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--device", type=str, default="0", help='GPU id or "cpu"')
    p.add_argument("--cache", type=str, default="disk", choices=["disk","ram","False","false"],
                   help="Caching strategy: 'disk' recommended on 32GB RAM systems")
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers (0–4 on Windows)")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint in the run")
    p.add_argument("--val_each_epoch", action="store_true", help="Run validation every epoch (default: only at end)")
    p.add_argument("--save_period", type=int, default=5, help="Save checkpoint every N epochs (-1 disables)")
    p.add_argument("--run_name", type=str, default=None, help="Optional custom run name")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--fast", action="store_true",
                   help="Faster prototyping: imgsz=960, epochs=40, batch auto-kept")
    return p.parse_args()

def main():
    args = parse_args()

    if args.fast:
        args.imgsz = 960
        args.epochs = min(args.epochs, 40)

    # Build a clean run name
    run_name = args.run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load model (can be a .pt path from your previous training, or a model string like 'yolo11m.pt')
    model = YOLO(args.model)

    # IMPORTANT: for best results with your 4 classes (car/biker/pedestrian/truck),
    # export a 4-class dataset or ensure your data.yaml 'names' contains only those 4.
    # Training with 11 classes is fine; you can filter at inference.

    overrides = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=args.cache,            # 'disk' avoids RAM blowups
        workers=args.workers,        # keep low on Windows
        name=run_name,
        deterministic=True,          # stable runs
        patience=15,                 # early stop patience
        save_period=args.save_period,
        close_mosaic=10,             # turn off mosaic near the end
        mosaic=1.0,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.10, scale=0.50, shear=2.0,
        label_smoothing=0.05,
        lr0=0.005, lrf=0.01,         # gentle LR for fine-tuning
        optimizer="AdamW",
        val=args.val_each_epoch,     # if False: we’ll run a final val after training
        seed=args.seed,
        # iou=0.7  # (optional) stronger NMS during training/val
    )

    if args.resume:
        # Resume full state (optimizer/epoch). Model should point to the last run's last.pt.
        print(">> Resuming training...")
        model.train(resume=True, **overrides)
    else:
        print(">> Starting/continuing training from:", args.model)
        model.train(**overrides)

    # End-of-run validation (if not validated each epoch)
    if not args.val_each_epoch:
        print("\n>> Running final validation...")
        model.val(data=args.data, plots=True)  # saves PR curves, confusion matrix, etc.

    print("\nDone. Check results in:  C:\\Users\\<you>\\...\\runs\\detect\\<run_name>\\")
    print("Use weights/best.pt or weights/last.pt for inference.")

if __name__ == "__main__":
    main()
