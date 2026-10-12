"""On-device vision: garment cutout (U2Net-p) + garment type (CLIP zero-shot) + dominant color.

Everything runs through ONNX Runtime so the same graphs can later be compiled for the
Hailo-10H on the ASUS UGen300 (Stage II). No image leaves the machine.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

MODELS = Path(__file__).resolve().parent.parent / "models"

# (prompt noun, part). Parts follow the closet schema: top / outer / bottom / shoes / bag.
GARMENTS = [
    ("t-shirt", "top"), ("polo shirt", "top"), ("button-up shirt", "top"),
    ("long-sleeve shirt", "top"), ("sweater", "top"), ("sweatshirt", "top"),
    ("hoodie", "outer"), ("zip-up jacket", "outer"), ("coat", "outer"), ("denim jacket", "outer"),
    ("jeans", "bottom"), ("trousers", "bottom"), ("shorts", "bottom"), ("sweatpants", "bottom"),
    ("skirt", "bottom"),
    ("sneakers", "shoes"), ("leather shoes", "shoes"), ("sandals", "shoes"),
    ("bag", "bag"), ("backpack", "bag"),
]
NAMED_COLORS = {
    "black": (25, 25, 25), "charcoal": (64, 64, 66), "gray": (128, 128, 128),
    "light gray": (195, 195, 192), "white": (240, 240, 236), "cream": (232, 222, 200),
    "beige": (200, 180, 150), "brown": (110, 75, 50), "navy": (30, 40, 70),
    "blue": (60, 100, 170), "light blue": (150, 180, 210), "olive": (100, 105, 60),
    "green": (50, 120, 70), "red": (170, 40, 40), "burgundy": (100, 30, 40),
    "pink": (225, 160, 170), "yellow": (225, 200, 80), "orange": (220, 120, 50),
    "purple": (110, 70, 140),
}
NEUTRALS = {"black", "charcoal", "gray", "light gray", "white", "cream", "beige", "navy", "brown"}


def _session(name: str, threads: int | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(MODELS / name), opts, providers=["CPUExecutionProvider"])


@dataclass
class Result:
    part: str
    garment: str
    confidence: float
    color_name: str
    color_hex: str
    embedding: np.ndarray
    cutout: Image.Image
    timings_ms: dict[str, float] = field(default_factory=dict)


class Vision:
    def __init__(self, quantized: bool = True, threads: int | None = None):
        self.u2net = _session("u2netp.onnx", threads)
        self.clip = _session("clip_vision_model_quantized.onnx" if quantized else "clip_vision_model.onnx", threads)
        self.quantized = quantized
        self.labels = self._label_embeddings()

    # ---------- text side: computed once, cached ----------
    def _label_embeddings(self) -> np.ndarray:
        cache = MODELS / "garment_labels.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(GARMENTS):
                return emb
        tok = Tokenizer.from_file(str(MODELS / "clip_tokenizer.json"))
        text = _session("clip_text_model.onnx")
        prompts = [f"a photo of a {g} hanging on a door" for g, _ in GARMENTS]
        enc = tok.encode_batch(prompts)
        n = max(len(e.ids) for e in enc)
        ids = np.array([e.ids + [49407] * (n - len(e.ids)) for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids}
        if "attention_mask" in [i.name for i in text.get_inputs()]:
            feeds["attention_mask"] = np.array([[1] * len(e.ids) + [0] * (n - len(e.ids)) for e in enc], dtype=np.int64)
        emb = text.run(["text_embeds"], feeds)[0]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        np.save(cache, emb)
        return emb

    # ---------- stage 1: cutout ----------
    def cutout(self, img: Image.Image) -> tuple[Image.Image, np.ndarray]:
        rgb = img.convert("RGB")
        x = np.asarray(rgb.resize((320, 320), Image.BILINEAR), dtype=np.float32)
        x = x / max(x.max(), 1e-6)
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.u2net.run(None, {self.u2net.get_inputs()[0].name: x})[0][0, 0]
        pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
        mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size, Image.BILINEAR)
        out = rgb.copy()
        out.putalpha(mask)
        return out, np.asarray(mask) > 127

    # ---------- stage 2: what is it ----------
    def embed(self, cut: Image.Image) -> np.ndarray:
        bg = Image.new("RGB", cut.size, (255, 255, 255))
        bg.paste(cut, mask=cut.split()[-1] if cut.mode == "RGBA" else None)
        bbox = cut.split()[-1].getbbox() if cut.mode == "RGBA" else None
        if bbox:
            bg = bg.crop(bbox)
        w, h = bg.size
        s = 224 / min(w, h)
        bg = bg.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = bg.size
        left, top = (w - 224) // 2, (h - 224) // 2
        bg = bg.crop((left, top, left + 224, top + 224))
        x = np.asarray(bg, dtype=np.float32) / 255.0
        x = (x - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        emb = self.clip.run(["image_embeds"], {"pixel_values": x})[0][0]
        return emb / np.linalg.norm(emb)

    def classify(self, emb: np.ndarray) -> tuple[str, str, float]:
        logits = 100.0 * self.labels @ emb
        p = np.exp(logits - logits.max())
        p /= p.sum()
        # Sum probabilities per part so near-synonyms (tee vs polo) don't split the vote.
        parts: dict[str, float] = {}
        for (g, part), pi in zip(GARMENTS, p):
            parts[part] = parts.get(part, 0.0) + float(pi)
        best = max(parts, key=parts.get)
        garment = max((i for i in range(len(GARMENTS)) if GARMENTS[i][1] == best), key=lambda i: p[i])
        return best, GARMENTS[garment][0], parts[best]

    # ---------- stage 3: color ----------
    @staticmethod
    def dominant_color(cut: Image.Image, mask: np.ndarray, k: int = 3, seed: int = 0) -> tuple[str, str]:
        px = np.asarray(cut.convert("RGB"), dtype=np.float32)[mask]
        if len(px) == 0:
            px = np.asarray(cut.convert("RGB"), dtype=np.float32).reshape(-1, 3)
        rng = np.random.default_rng(seed)
        px = px[rng.choice(len(px), size=min(4000, len(px)), replace=False)]
        centers = px[rng.choice(len(px), size=k, replace=False)]
        for _ in range(12):
            lab = np.argmin(((px[:, None] - centers[None]) ** 2).sum(-1), axis=1)
            centers = np.array([px[lab == j].mean(0) if (lab == j).any() else centers[j] for j in range(k)])
        counts = np.bincount(lab, minlength=k)
        c = centers[counts.argmax()]
        name = min(NAMED_COLORS, key=lambda n: ((np.array(NAMED_COLORS[n]) - c) ** 2).sum())
        return name, "#%02x%02x%02x" % tuple(int(v) for v in c)

    def analyze(self, img: Image.Image) -> Result:
        t = {}
        t0 = time.perf_counter()
        cut, mask = self.cutout(img)
        t["cutout"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emb = self.embed(cut)
        part, garment, conf = self.classify(emb)
        t["classify"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cname, chex = self.dominant_color(cut, mask)
        t["color"] = (time.perf_counter() - t0) * 1000
        return Result(part, garment, conf, cname, chex, emb, cut, t)


def load_image(path: str | Path, max_side: int = 1600) -> Image.Image:
    img = Image.open(path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img.thumbnail((max_side, max_side))
    return img


if __name__ == "__main__":
    import sys
    v = Vision()
    for p in sys.argv[1:]:
        r = v.analyze(load_image(p))
        print(json.dumps({"file": Path(p).name, "part": r.part, "garment": r.garment,
                          "conf": round(r.confidence, 2), "color": r.color_name,
                          "ms": {k: round(x) for k, x in r.timings_ms.items()}}))
