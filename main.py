from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
import io
from pathlib import Path
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== モデル読み込み =====
BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"


def load_class_names(path: str):
    class_names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                class_names.append(name)
    return class_names


class_names = load_class_names(CLASS_NAMES_PATH)

if len(class_names) == 0:
    raise ValueError("class_names.txt にクラス名がありません")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_path = BASE_DIR / "best_efficientnetv2.pth"

weights = EfficientNet_V2_S_Weights.DEFAULT
model = models.efficientnet_v2_s(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return {
            "prediction": class_names[pred],
            "confidence": float(probs[0][pred].detach().cpu().item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
