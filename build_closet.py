"""Index a folder of garment images into a local closet (closet.json + closet.npy).

Every item goes through the same on-device pipeline as a new garment on the door.
    uv run python build_closet.py <image folder> [out folder]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from closet_door.closet import Closet, Item
from closet_door.vision import Vision, load_image


def main(src: Path, out: Path) -> None:
    v = Vision()
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and "-thumb" not in p.name and "-garment" not in p.name)
    items, embs = [], []
    for p in files:
        img = load_image(p, 800)
        if img.mode == "RGBA":  # already cut out
            cut, mask = img, np.asarray(img.split()[-1]) > 127
        else:
            cut, mask = v.cutout(img)
        emb = v.embed(cut)
        part, garment, _ = v.classify(emb)
        cname, chex = v.dominant_color(cut, mask)
        items.append(Item(p.stem, str(p.resolve()), part, garment, cname, chex))
        embs.append(emb)
    Closet(items, np.stack(embs)).save(out)
    print(f"{len(items)} items -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "closet"))
