"""Local demo UI. Runs on 127.0.0.1 only; the page loads nothing from the internet.
    uv run uvicorn app:app --port 8765
"""
import base64
import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

from closet_door.closet import Closet, Item
from closet_door.vision import Vision, load_image

ROOT = Path(__file__).parent
DOOR = Path(os.environ.get("DOOR_PHOTOS", ROOT.parent / "wardrobe-ai" / "photos"))
vision = Vision(quantized=True)
closet = Closet.load(ROOT / "closet")
by_id = {i.id: i for i in closet.items}
app = FastAPI()


def png_b64(img: Image.Image, side: int = 520) -> str:
    img = img.copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "ui.html").read_text(encoding="utf-8")


@app.get("/door")
def door():
    return sorted(p.name for p in DOOR.glob("*.JPG"))


@app.get("/door/{name}")
def door_photo(name: str):
    p = DOOR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    img = load_image(p, 900)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return HTMLResponse(buf.getvalue(), media_type="image/jpeg")


@app.get("/item/{item_id}")
def item_image(item_id: str):
    it = by_id.get(item_id)
    if not it:
        raise HTTPException(404)
    thumb = Path(it.image.replace("-view.webp", "-thumb.webp"))
    return FileResponse(thumb if thumb.exists() else it.image)


@app.post("/analyze")
async def analyze(name: str = Form(""), temp: float = Form(22), file: UploadFile | None = File(None)):
    t0 = time.perf_counter()
    if file is not None and file.filename:
        img = load_image(io.BytesIO(await file.read()), 1600)
    else:
        img = load_image(DOOR / Path(name).name, 1600)
    t_load = (time.perf_counter() - t0) * 1000
    r = vision.analyze(img)
    new = Item("new", "", r.part, r.garment, r.color_name, r.color_hex)
    t1 = time.perf_counter()
    sims = closet.similar(r)
    outfits = closet.outfits(new, temp)
    t_reco = (time.perf_counter() - t1) * 1000
    return {
        "cutout": png_b64(r.cutout),
        "part": r.part, "garment": r.garment, "confidence": round(r.confidence, 2),
        "color": {"name": r.color_name, "hex": r.color_hex},
        "timings_ms": {"decode": round(t_load), **{k: round(v) for k, v in r.timings_ms.items()}, "recommend": round(t_reco)},
        "similar": [{"id": it.id, "garment": it.garment, "color": it.color_name, "sim": round(s, 2)} for it, s in sims],
        "outfits": outfits,
        "closet_size": len(closet.items),
    }
