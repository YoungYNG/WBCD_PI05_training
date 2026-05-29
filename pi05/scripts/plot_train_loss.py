#!/usr/bin/env python3
"""Plot OpenPI training loss from a log file.

Usage:
    python scripts/plot_train_loss.py /path/to/train.log

The script extracts lines like:
    Step 100: grad_norm=..., loss=..., param_norm=...

It writes a CSV and PNG next to the log by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


STEP_RE = re.compile(
    r"Step\s+(?P<step>\d+):\s+"
    r"grad_norm=(?P<grad_norm>[0-9.eE+-]+),\s+"
    r"loss=(?P<loss>[0-9.eE+-]+),\s+"
    r"param_norm=(?P<param_norm>[0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Training log path.")
    parser.add_argument("--out-png", type=Path, default=None, help="Output PNG path. Defaults next to log.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path. Defaults next to log.")
    parser.add_argument(
        "--avg-window",
        type=int,
        default=0,
        help="Trailing moving-average window in points. 0 chooses an automatic window.",
    )
    parser.add_argument("--title", default=None, help="Plot title. Defaults to the log filename.")
    return parser.parse_args()


def parse_log(log_path: Path) -> list[dict[str, float | int]]:
    by_step: dict[int, dict[str, float | int]] = {}
    text = log_path.read_text(errors="replace")
    for match in STEP_RE.finditer(text):
        step = int(match.group("step"))
        by_step[step] = {
            "step": step,
            "loss": float(match.group("loss")),
            "grad_norm": float(match.group("grad_norm")),
            "param_norm": float(match.group("param_norm")),
        }
    return [by_step[step] for step in sorted(by_step)]


def write_csv(rows: list[dict[str, float | int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "grad_norm", "param_norm"])
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: list[float], window: int) -> list[float]:
    averaged = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def plot_loss(rows: list[dict[str, float | int]], out_png: Path, title: str, avg_window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]

    if avg_window <= 0:
        avg_window = min(50, max(5, len(rows) // 40))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5.8), dpi=170)
    plt.plot(steps, losses, linewidth=1.1, alpha=0.6, label="loss")
    plt.scatter(steps, losses, s=7, alpha=0.65)

    if len(rows) >= 5:
        avg = moving_average(losses, avg_window)
        plt.plot(steps, avg, linewidth=2.2, label=f"trailing avg {avg_window} pts")

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    args = parse_args()
    log_path = args.log_path.expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    out_png = args.out_png or log_path.with_name(f"{log_path.stem}_loss_curve.png")
    out_csv = args.out_csv or log_path.with_name(f"{log_path.stem}_loss_points.csv")
    title = args.title or f"Training Loss - {log_path.name}"

    rows = parse_log(log_path)
    if not rows:
        raise RuntimeError(f"No training loss records found in {log_path}")

    write_csv(rows, out_csv)
    plot_loss(rows, out_png, title, args.avg_window)

    first = rows[0]
    last = rows[-1]
    print(f"points={len(rows)}")
    print(f"first_step={first['step']} first_loss={float(first['loss']):.6f}")
    print(f"last_step={last['step']} last_loss={float(last['loss']):.6f}")
    print(f"csv={out_csv}")
    print(f"png={out_png}")


if __name__ == "__main__":
    main()
