# Closet Door

An on-device camera for the closet door. Hang a garment on the door; the device cuts it out, names the type and main colour, checks whether you already own something like it, and suggests outfits for today's temperature. Photos never leave the device.

Entry for the ASUS UGen AI League (Everyday AI, Battlefield Lightning / UGen300). This repository is the Stage I prototype: it runs on a laptop CPU through ONNX Runtime, as a stand-in for the Hailo-10H.

## Why a device and not an app

I keep a list of my own clothes: 104 items, and every one was photographed hanging on the same door handle. Three things went wrong doing that with a phone:

1. **Clothes photos are private.** A wardrobe is a picture of your body, your home and your money. Cloud vision APIs see all of it.
2. **Colour is never the same twice.** A phone shot under warm indoor light turns light-blue denim olive (it happened in this prototype, see Limits). A fixed camera on the door sees the same light every time, so it can be calibrated once.
3. **Nobody photographs clothes one by one.** If the camera is already on the door, adding an item is just hanging it there.

## What runs (Stage I, measured)

| Stage | Model | Notes |
|---|---|---|
| Cutout | U2Net-p (4.6 MB, ONNX) | salient-object mask, 320×320 |
| Type | CLIP ViT-B/32 image encoder (int8, 87 MB, ONNX) | zero-shot over 20 garment types, grouped into top / outer / bottom / shoes / bag |
| Colour | k-means on masked pixels (numpy) | mapped to 19 named colours |
| Already own it? | cosine similarity of CLIP embeddings | against the local closet index |
| Outfits | plain rules | warmth vs temperature, at most one accent colour, light/dark contrast; every suggestion lists its reasons |

Text embeddings for the 20 garment prompts are computed once and cached, so the text encoder never runs on the device.

### Accuracy: garment slot, 103 of my own items

Labels are my own, from an earlier personal tool. Images are already cut out, so this measures the type stage alone.

| CLIP | 5-way part | 4-way slot (upper / bottom / shoes / bag) |
|---|---|---|
| fp32 | 81.6% | 97.1% |
| int8 | 79.6% | 94.2% |

Most 5-way errors are hoodies and sweatshirts that I labelled "top" and the model calls "outer". For building an outfit the slot is what matters. `results/eval_parts.json`

### Latency: 30 door photos, median ms, Intel Core Ultra 7 270K Plus (CPU only)

| CLIP | threads | cutout | type | colour | total |
|---|---|---|---|---|---|
| fp32 | 1 | 341 | 82 | 23 | 446 |
| fp32 | 4 | 111 | 35 | 24 | 171 |
| int8 | 1 | 347 | 37 | 22 | 406 |
| int8 | 4 | 112 | 24 | 23 | 159 |

Cutout is 85% of single-thread time. That is the first thing to move onto the Hailo-10H. `results/bench.json`

## Limits (found in this prototype)

- Hanging shorts were called "long-sleeve shirt" in 2 of the first 8 raw door photos I checked by eye.
- Colour under warm light is wrong for light colours: light-blue jeans came out "olive", light-gray shorts on a beige bed came out "beige". The fix planned for Stage II is a printed white reference card on the door, read on every frame.
- The cutout keeps the door handle and hanger.
- Outfit rules are mine and untested on anyone else.

## Stage II plan (UGen300, Hailo-10H)

1. Compile U2Net-p and the CLIP image encoder to HEF with the Hailo Dataflow Compiler; measure the same 30 photos against the table above. Whether both graphs compile unchanged is unknown until then; fallback for the cutout is a YOLO-seg model from the Hailo Model Zoo.
2. Re-run the 103-item accuracy check on int8 Hailo output and report the drop next to the CPU int8 number.
3. Add the white reference card and measure colour-name agreement before and after.
4. Camera on the door: trigger on "new object hanging" (frame difference), not a button.

## Run

```bash
uv sync
uv run python build_closet.py <folder of garment photos>   # builds closet/ locally
uv run uvicorn app:app --host 127.0.0.1 --port 8765        # demo UI
uv run python eval_parts.py                                # needs the labelled set
uv run python bench.py 30
```

Models go in `models/` (not in git): `u2netp.onnx` from the rembg releases, and `vision_model_quantized.onnx`, `vision_model.onnx`, `text_model.onnx`, `tokenizer.json` from `Xenova/clip-vit-base-patch32` (prefixed `clip_`).

My clothes photos and closet index are not in this repository.

## Disclosure

- Written with Claude Code (Anthropic) as a coding assistant; design decisions, data and measurements are mine and are reproducible with the scripts above.
- Models: U2Net-p (Apache-2.0, via rembg), OpenAI CLIP ViT-B/32 (MIT) in the ONNX export by Xenova.
- I previously built a personal web closet for my own items; this project is new code with a different job (an on-device camera), and uses that tool's labels only as ground truth for the accuracy check.
