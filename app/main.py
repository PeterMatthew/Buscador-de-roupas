import os, glob, torch
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel

BASE_URL = "http://127.0.0.1:8000"
DINO_MODEL = 'facebook/dinov2-base'
YOLO_PATH = '../best.pt'
IMAGES_PATH = "../deepfashion2/train/image"
EMBEDDINGS_GLOB = "../embedding/*.npz"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory=IMAGES_PATH), name="images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo = YOLO(YOLO_PATH)
processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
dino = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

category_data = {}
for path in glob.glob(EMBEDDINGS_GLOB):
    name = os.path.basename(path).replace('_data.npz', '')
    data = np.load(path, allow_pickle=True)
    
    feats = F.normalize(torch.from_numpy(data["embeddings"]).float(), p=2, dim=1).to(device)
    category_data[name] = {"embeddings": feats, "filenames": data["strings"]}

def get_embedding(img):
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = dino(**inputs)
        return F.normalize(out.last_hidden_state.mean(dim=1), p=2, dim=1)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    img = Image.open(image.file).convert('RGB')
    res = yolo(img)[0].to('cpu')
    return [{"box": b, "category": res.names[int(c)]} 
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist())]

@app.post("/similar")
async def similar(
    image: UploadFile = File(...),
    x0: float = Form(...), y0: float = Form(...), 
    x1: float = Form(...), y1: float = Form(...),
    category: str = Form(...),
    offset: int = Form(0), limit: int = Form(12)
):
    if category not in category_data:
        raise HTTPException(400, f"Category '{category}' unknown")

    # process query
    img = Image.open(image.file).convert('RGB').crop((x0, y0, x1, y1))
    query_feat = get_embedding(img)

    # search
    db = category_data[category]
    sims = torch.mm(query_feat, db["embeddings"].t()).squeeze()
    scores, indices = torch.sort(sims, descending=True)

    # paginate
    start, end = offset, offset + limit
    p_indices, p_scores = indices[start:end].tolist(), scores[start:end].tolist()
    
    items = [{
        "image_name": db["filenames"][i],
        "score": s,
        "image_url": f"{BASE_URL}/images/{db['filenames'][i]}"
    } for s, i in zip(p_scores, p_indices)]

    total = len(db["filenames"])
    return {
        "itemSummaries": items,
        "pagination": {
            "total": total, "offset": offset, "limit": limit,
            "next_offset": end if end < total else None,
            "prev_offset": max(0, offset - limit) if offset > 0 else None
        }
    }
