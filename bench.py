"""Latency per stage on this machine's CPU: fp32 vs int8 CLIP, 1 vs 4 threads.
This is the baseline the Hailo-10H port (Stage II) has to beat. Writes results/bench.json.
    uv run python bench.py [n_photos]
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

from closet_door.vision import Vision, load_image

DOOR = Path("../wardrobe-ai/photos")


def run(quantized: bool, threads: int, imgs) -> dict:
    v = Vision(quantized=quantized, threads=threads)
    v.analyze(imgs[0])  # warm-up
    stages: dict[str, list[float]] = {}
    for im in imgs:
        for k, ms in v.analyze(im).timings_ms.items():
            stages.setdefault(k, []).append(ms)
    med = {k: round(float(np.median(x)), 1) for k, x in stages.items()}
    p90 = {k: round(float(np.percentile(x, 90)), 1) for k, x in stages.items()}
    return {"clip": "int8" if quantized else "fp32", "threads": threads, "n": len(imgs),
            "median_ms": med, "p90_ms": p90, "median_total_ms": round(sum(med.values()), 1)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    imgs = [load_image(p, 1600) for p in sorted(DOOR.glob("*.JPG"))[:n]]
    out = {"cpu": platform.processor(), "python": platform.python_version(),
           "runs": [run(q, t, imgs) for q in (False, True) for t in (1, 4)]}
    print(json.dumps(out, indent=1))
    Path("results").mkdir(exist_ok=True)
    Path("results/bench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
