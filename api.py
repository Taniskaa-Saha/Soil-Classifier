from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import timm
import json
import io
import os

app = FastAPI()

# ========== CONFIG ==========
project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "soil_model.pth")
classes_path = os.path.join(project_root, "classes.json")

# Load class names
with open(classes_path, "r") as f:
    classes = json.load(f)

num_classes = len(classes)

# Define transforms (same as validation)
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ========== API Endpoint ==========
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = val_tfms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = output.max(1)

        predicted_class = classes[predicted.item()]
        return JSONResponse(content={"class": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
