"""Accuracy of the garment-type stage against the owner's own labels (104 items).

Labels come from the owner's existing closet data (wardrobe-ai/data/library.json, labelled by
the owner in an earlier personal tool). Images there are already cut out, so this measures the
classifier alone, not the cutout stage.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from closet_door.vision import Vision

LIB = Path("../wardrobe-ai/data")
MAP = {"upperbody": "top", "wholebody_up": "outer", "lowerbody": "bottom", "shoes": "shoes", "bag": "bag"}
SLOT = {"top": "upper", "outer": "upper", "bottom": "bottom", "shoes": "shoes", "bag": "bag"}


def main(quantized: bool) -> dict:
    items = [i for i in json.loads((LIB / "library.json").read_text(encoding="utf-8")) if i["part"] in MAP]
    v = Vision(quantized=quantized)
    rows, ms = [], []
    for it in items:
        img = Image.open(LIB / "imported" / Path(it["image"]).name).convert("RGBA")
        img.thumbnail((800, 800))
        t0 = time.perf_counter()
        part, garment, conf = v.classify(v.embed(img))
        ms.append((time.perf_counter() - t0) * 1000)
        rows.append((MAP[it["part"]], part, garment))
    n = len(rows)
    exact = sum(t == p for t, p, _ in rows)
    slot = sum(SLOT[t] == SLOT[p] for t, p, _ in rows)
    confusion = Counter((t, p) for t, p, _ in rows if t != p)
    return {"model": "int8" if quantized else "fp32", "n": n,
            "part_acc": round(exact / n, 3), "slot_acc": round(slot / n, 3),
            "classify_ms_median": round(float(np.median(ms)), 1),
            "errors": {f"{t}->{p}": c for (t, p), c in confusion.most_common()}}


if __name__ == "__main__":
    out = [main(q) for q in (False, True)]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    Path("results").mkdir(exist_ok=True)
    Path("results/eval_parts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
