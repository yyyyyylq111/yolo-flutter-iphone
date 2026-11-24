#!/usr/bin/env python3
"""Generate A's CPU and resident-memory chart from mobile_resources.csv."""
import argparse
import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv_file.open(encoding="utf-8-sig", newline="")))
    rows = [r for r in rows if r.get("phase", "").startswith("after_")]
    labels = [r.get("image_id", "") for r in rows]

    def values(key):
        out = []
        for row in rows:
            try:
                out.append(float(row[key]) if row.get(key, "") else math.nan)
            except (TypeError, ValueError):
                out.append(math.nan)
        return out

    cpu = values("cpu_percent")
    memory = values("memory_mb")
    x = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, cpu, marker="o")
    axes[0].set_title("iPhone 15 App Process CPU Utilization")
    axes[0].set_ylabel("CPU (% of total device capacity)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, memory, marker="o")
    axes[1].set_title("iPhone 15 App Process Resident Memory")
    axes[1].set_ylabel("Resident memory (MB)")
    axes[1].set_xlabel("Measured inference run")
    axes[1].grid(True, alpha=0.3)

    if labels:
        step = max(1, len(labels) // 12)
        axes[1].set_xticks(x[::step])
        axes[1].set_xticklabels(labels[::step], rotation=45, ha="right")
    fig.tight_layout()
    output = args.csv_file.with_name("mobile_resource_chart.png")
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
