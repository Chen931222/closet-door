"""Local closet index + outfit suggestions. Plain rules, every score has a reason a person can read."""
from __future__ import annotations

import colorsys
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from .vision import NEUTRALS, Result

SLOT = {"top": "upper", "outer": "upper", "bottom": "bottom", "shoes": "shoes", "bag": "bag"}
WARMTH = {  # rough clo-like units per garment type
    "t-shirt": 1, "polo shirt": 1, "button-up shirt": 1.5, "long-sleeve shirt": 1.5, "sweater": 2.5,
    "sweatshirt": 2.5, "hoodie": 3, "zip-up jacket": 3, "coat": 4, "denim jacket": 3,
    "jeans": 1.5, "trousers": 1.5, "shorts": 0.5, "sweatpants": 1.5, "skirt": 1,
    "sneakers": 0, "leather shoes": 0, "sandals": 0, "bag": 0, "backpack": 0,
}


@dataclass
class Item:
    id: str
    image: str
    part: str
    garment: str
    color_name: str
    color_hex: str

    @property
    def slot(self) -> str:
        return SLOT[self.part]

    @property
    def warmth(self) -> float:
        return WARMTH.get(self.garment, 1)

    @property
    def lightness(self) -> float:
        r, g, b = (int(self.color_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hls(r, g, b)[1]


class Closet:
    def __init__(self, items: list[Item], embeddings: np.ndarray):
        self.items, self.emb = items, embeddings

    @classmethod
    def load(cls, folder: Path) -> "Closet":
        items = [Item(**d) for d in json.loads((folder / "closet.json").read_text(encoding="utf-8"))]
        return cls(items, np.load(folder / "closet.npy"))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "closet.json").write_text(json.dumps([asdict(i) for i in self.items], ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(folder / "closet.npy", self.emb)

    def similar(self, r: Result, k: int = 3) -> list[tuple[Item, float]]:
        """Items in the same slot that look like the new one: the 'you already own this' check."""
        sims = self.emb @ r.embedding
        idx = [i for i in np.argsort(-sims) if self.items[i].slot == SLOT[r.part]][:k]
        return [(self.items[i], float(sims[i])) for i in idx]

    def outfits(self, new: Item, temp_c: float, k: int = 3) -> list[dict]:
        """Build outfits that include the new item, ranked by warmth fit and colour rules."""
        by_slot: dict[str, list[Item]] = {}
        for it in self.items:
            by_slot.setdefault(it.slot, []).append(it)
        uppers = [new] if new.slot == "upper" else by_slot.get("upper", [])
        bottoms = [new] if new.slot == "bottom" else by_slot.get("bottom", [])
        shoes = [new] if new.slot == "shoes" else by_slot.get("shoes", [])
        target = max(1.5, min(6.0, (28 - temp_c) / 3.5 + 1.5))  # warmer clothes as it gets colder
        scored = []
        for u, b, s in product(uppers, bottoms, shoes or [None]):
            parts = [x for x in (u, b, s) if x]
            reasons, score = [], 0.0
            warmth = sum(x.warmth for x in parts)
            score -= abs(warmth - target)
            loud = [x for x in parts if x.color_name not in NEUTRALS]
            if len(loud) <= 1:
                score += 1
                reasons.append("one accent colour at most" if loud else "all neutrals")
            else:
                score -= len(loud) - 1
            contrast = abs(u.lightness - b.lightness)
            score += min(contrast, 0.5)  # some contrast reads better than tone-on-tone
            if contrast > 0.25:
                reasons.append("light/dark contrast between top and bottom")
            if s is not None and s.garment == "sandals" and temp_c < 18:
                score -= 1.5  # open shoes below 18°C
            reasons.append(f"warmth {warmth:g} for {temp_c:g}°C (target {target:.1f})")
            scored.append((score, [x.id for x in parts], reasons))
        scored.sort(key=lambda t: -t[0])
        picked, used = [], set()
        for s, ids, r in scored:  # no piece other than the new one repeats across suggestions
            others = set(ids) - {"new"}
            if others & used:
                continue
            picked.append({"score": round(s, 2), "items": ids, "reasons": r})
            used |= others
            if len(picked) == k:
                break
        return picked
