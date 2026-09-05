from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import json
import sys
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import STGCN

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")
# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LABEL MAP =================
with open("label_map.json", "r") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}

# ================= MODEL =================
model = STGCN(num_classes=len(label_map))
model.load_state_dict(torch.load("stgcn_model.pth", map_location="cpu"))
model.eval()

# ================= INPUT =================
class LandmarkInput(BaseModel):
    landmarks: list

# ================= PREDICT =================
@app.post("/predict")
def predict(data: LandmarkInput):

    x = np.array(data.landmarks, dtype=np.float32)

    # must be (30, 21, 2)
    if x.shape != (30, 21, 2):
        return {"prediction": "no_hand"}

    # (30,21,2) -> (2,30,21)
    x = np.transpose(x, (2, 0, 1))

    x = torch.tensor(x, dtype=torch.float32)

    # normalize
    x = (x - x.mean()) / (x.std() + 1e-6)

    x = x.unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return {"prediction": idx_to_label[pred]}