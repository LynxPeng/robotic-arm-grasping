"""Train GR-ConvNet on local Jacquard V2 shards with Trackio."""
import argparse
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "robotic-grasping"
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inference.models.grconvnet3 import GenerativeResnet
from inference.post_process import post_process_output
from utils.data.grasp_data import GraspDatasetBase
from utils.data.jacquard_data import JacquardDataset
from utils.dataset_processing import evaluation
from utils.dataset_processing.image import DepthImage, Image


def object_split(labels, seed):
    objects = sorted({path.parent.name for path in labels})
    if len(objects) < 2:
        raise ValueError("At least two distinct objects are required")
    random.Random(seed).shuffle(objects)
    val_objects = set(objects[:max(1, len(objects) // 10)])
    return ([p for p in labels if p.parent.name not in val_objects],
            [p for p in labels if p.parent.name in val_objects])


class ShardedJacquard(JacquardDataset):
    def __init__(self, labels, augment=False, input_size=224):
        GraspDatasetBase.__init__(self, output_size=input_size, include_depth=True, include_rgb=True,
                                  random_rotate=augment, random_zoom=augment)
        self.grasp_files = list(map(str, labels))
        self.depth_files = [str(p.with_name(p.name.removesuffix("grasps.txt") + "perfect_depth.tiff")) for p in labels]
        self.rgb_files = [str(p.with_name(p.name.removesuffix("grasps.txt") + "RGB.png")) for p in labels]


class RawJacquard(ShardedJacquard):
    """Decode on workers and leave image geometry and normalization to CUDA."""

    def __getitem__(self, idx):
        turn = random.randrange(4)
        zoom = np.random.uniform(0.5, 1.0)
        depth = DepthImage.from_tiff(self.depth_files[idx]).img
        rgb = Image.from_file(self.rgb_files[idx]).img
        if depth.shape != (1024, 1024) or rgb.shape != (1024, 1024, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Unexpected Jacquard image layout for {self.grasp_files[idx]}")
        rectangles = self.get_gtbb(idx, turn * np.pi / 2, zoom)
        pos, angle, width = rectangles.draw((self.output_size, self.output_size))
        half_size = self.output_size / 2
        width = np.clip(width, 0.0, half_size) / half_size
        targets = tuple(map(self.numpy_to_torch, (pos, np.cos(2 * angle), np.sin(2 * angle), width)))
        return (depth.astype(np.float32, copy=False), rgb, turn, zoom), targets, idx, turn, zoom


def fused_rgbd(batch, device, output_size):
    """Sample the rotated center crop directly, then average 2x2 subpixels."""
    depth, rgb, turn, zoom = batch
    depth = depth.to(device, non_blocking=True).unsqueeze(1)
    depth = (depth - depth.mean(dim=(2, 3), keepdim=True)).clamp_(-1, 1)
    rgb = rgb.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    source = torch.cat((depth, rgb), dim=1)

    size = depth.shape[-1]
    sample_size = output_size * 2
    axis = (torch.arange(sample_size, device=device, dtype=torch.float32) + 0.5) * (2 / sample_size) - 1
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    zoom = zoom.to(device, dtype=torch.float64, non_blocking=True)
    margin = torch.floor((1 - zoom) * size / 2)
    crop_fraction = (1 - 2 * margin / size).float().view(-1, 1, 1)
    # Inverse map for skimage's positive (counterclockwise) image rotations.
    turn = turn.to(device, non_blocking=True)
    cos = ((turn == 0).float() - (turn == 2).float()).view(-1, 1, 1)
    sin = ((turn == 1).float() - (turn == 3).float()).view(-1, 1, 1)
    grid = torch.stack(((cos * xx - sin * yy) * crop_fraction,
                        (sin * xx + cos * yy) * crop_fraction), dim=-1)
    sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sampled = F.avg_pool2d(sampled, 2)
    rgb = sampled[:, 1:]
    return torch.cat((sampled[:, :1], rgb - rgb.mean(dim=(1, 2, 3), keepdim=True)), dim=1)


def configure_worker(_):
    import cv2

    cv2.setNumThreads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=6, help="PyTorch CPU compute threads")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--channel-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-scenes", type=int, help="Fixed random subset size for screening")
    parser.add_argument("--val-scenes", type=int, help="Fixed random subset size for screening")
    parser.add_argument("--exclude-screening-val", action="store_true",
                        help="Exclude the fixed 96 validation scenes used for LR screening")
    parser.add_argument("--max-batches", type=int, help="Limit optimizer steps per epoch")
    parser.add_argument("--stage", default="train")
    parser.add_argument("--group", default="jacquard-v2")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--gpu-augment", action="store_true",
                        help="Fuse training image geometry on GPU; validation stays upstream")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Require positive epochs and batch size, and nonnegative workers")
    if args.threads < 1 or args.input_size < 32 or args.input_size % 4 or args.channel_size < 1:
        parser.error("Use positive threads and channels, and input size >= 32 divisible by 4")
    for limit in (args.train_scenes, args.val_scenes, args.max_batches):
        if limit is not None and limit < 1:
            parser.error("Scene and batch limits must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be finite and positive")
    if args.gpu_augment and args.device != "cuda":
        parser.error("--gpu-augment requires --device cuda")

    seed = 123
    configure_worker(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = args.data.resolve()
    labels = sorted(data.glob("Jacquard*Dataset_*/*/*_grasps.txt"))
    train_files, val_files = object_split(labels, seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")

    if args.train_scenes:
        train_files = random.Random(seed + 1).sample(train_files, min(args.train_scenes, len(train_files)))
    if args.exclude_screening_val:
        screening = set(random.Random(seed + 2).sample(val_files, min(96, len(val_files))))
        val_files = [label for label in val_files if label not in screening]
    if not val_files:
        parser.error("No validation scenes remain")
    if args.val_scenes:
        val_files = random.Random(seed + 2).sample(val_files, min(args.val_scenes, len(val_files)))
    if args.smoke:
        train_files = train_files[:args.batch_size]
        val_files = val_files[:2]
    device = torch.device(args.device)
    train_dataset = RawJacquard if args.gpu_augment else ShardedJacquard
    train_data = train_dataset(train_files, augment=True, input_size=args.input_size)
    val_data = ShardedJacquard(val_files, input_size=args.input_size)
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0, worker_init_fn=configure_worker)
    # Full-resolution input queues are larger; keep only one prefetched batch per worker.
    train_options = {**loader_options, "prefetch_factor": 1} if args.gpu_augment and args.workers else loader_options
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **train_options)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, **loader_options)
    model = GenerativeResnet(input_channels=4, dropout=False, channel_size=args.channel_size)
    model = model.to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = 1 if args.smoke else args.epochs
    stage = "smoke" if args.smoke else args.stage
    steps = min(args.max_batches or len(train_loader), len(train_loader))
    name = f"jacquard-v2-{args.device}-{stage}-lr{args.lr:g}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    folder = RESULTS / "training-v2" / name
    config = dict(data=str(data), seed=seed, train_scenes=len(train_files), val_scenes=len(val_files),
                  epochs=epochs, batch_size=args.batch_size, workers=args.workers, input_size=args.input_size,
                  channel_size=args.channel_size, threads=args.threads, device=args.device, stage=stage,
                  exclude_screening_val=args.exclude_screening_val, gpu_augment=args.gpu_augment,
                  optimizer="Adam", learning_rate=args.lr, initialization="random", total_batches_per_epoch=steps)
    if device.type == "cuda":
        config["gpu"] = torch.cuda.get_device_name()
    if args.benchmark:
        started = time.perf_counter()
        for i, (x, targets, *_) in enumerate(islice(train_loader, steps), 1):
            if args.gpu_augment:
                x = fused_rgbd(x, device, args.input_size)
            optimizer.zero_grad(set_to_none=True)
            result = compute_batch(model, x, targets, device)
            result["loss"].backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"Benchmark batch {i}/{steps}, elapsed={time.perf_counter() - started:.3f}s", flush=True)
        return
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run {name} | {folder} | {config}", flush=True)

    os.environ.setdefault("TRACKIO_DIR", str(RESULTS / "trackio"))
    import trackio

    run = trackio.init(project="grconvnet-jacquard", name=name, group=args.group,
                       config=config, embed=False, auto_log_gpu=device.type == "cuda",
                       gpu_log_interval=5, auto_log_cpu=True, cpu_log_interval=5)
    best = -1.0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            started = time.monotonic()
            train_sums = {}
            seen = 0
            for batch, (x, targets, *_meta) in enumerate(islice(train_loader, steps), 1):
                if args.gpu_augment:
                    x = fused_rgbd(x, device, args.input_size)
                optimizer.zero_grad(set_to_none=True)
                result = compute_batch(model, x, targets, device)
                loss = result["loss"]
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite training loss at epoch {epoch}, batch {batch}")
                loss.backward()
                optimizer.step()
                accumulate_losses(train_sums, result, len(x))
                seen += len(x)
                if batch == 1 or batch % 25 == 0 or batch == steps:
                    run.log({"train/batch_loss": value, "train/epoch": epoch, "train/batch": batch,
                             "train/samples_per_second": seen / (time.monotonic() - started),
                             **{f"train/batch_{key}": val.item() for key, val in result["losses"].items()}})
                    print(f"Epoch {epoch} batch {batch}/{steps} loss={value:.6f}", flush=True)

            train_seconds = time.monotonic() - started
            model.eval()
            val_sums = {}
            matched = 0
            with torch.inference_mode():
                for x, targets, indices, rotations, zooms in val_loader:
                    result = compute_batch(model, x, targets, device)
                    accumulate_losses(val_sums, result, len(x))
                    prediction = {k: v.cpu() for k, v in result["pred"].items()}
                    for i, (index, rot, zoom) in enumerate(zip(indices, rotations, zooms)):
                        quality, angle, width = post_process_output(
                            *(prediction[key][i] for key in ("pos", "cos", "sin", "width")))
                        matched += evaluation.calculate_iou_match(
                            quality, angle, val_data.get_gtbb(int(index), float(rot), float(zoom)),
                            grasp_width=width * (val_data.output_size / 2 / 150))
            rate = matched / len(val_data)
            metrics = {"epoch": epoch,
                       **{f"train/{key}": val / seen for key, val in train_sums.items()},
                       **{f"val/{key}": val / len(val_data) for key, val in val_sums.items()},
                       "val/iou_match_rate": rate, "val/matched": matched, "val/scenes": len(val_data),
                       "val/failed": len(val_data) - matched, "train/scenes": seen,
                       "train/learning_rate": args.lr, "train/seconds": train_seconds,
                       "train/samples_per_second": seen / train_seconds,
                       "epoch_seconds": time.monotonic() - started}
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, "metrics": metrics}
            pending = folder / "checkpoint.tmp"
            torch.save(state, pending)
            pending.replace(folder / "latest.pt")
            if rate > best:
                torch.save(state, pending)
                pending.replace(folder / "best.pt")
                best = rate
            run.log(metrics)
            with (folder / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
    finally:
        run.finish()


def compute_batch(model, x, targets, device):
    return model.compute_loss(
        x.to(device, non_blocking=True, memory_format=torch.channels_last),
        [target.to(device, non_blocking=True) for target in targets],
    )


def accumulate_losses(totals, result, size):
    losses = {"loss": result["loss"], **result["losses"]}
    for key, value in losses.items():
        totals[key] = totals.get(key, 0.0) + value.detach().item() * size


if __name__ == "__main__":
    main()
