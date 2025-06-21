import os
import uuid
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
from torchvision import transforms
import torch

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# UPLOAD_DIR = Path("uploaded_images")
# UPLOAD_DIR.mkdir(exist_ok=True)
# DATA_FILE = Path("product_data.json")

UPLOAD_DIR = Path("../product_db/uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_FILE = Path("../product_db/product_data.json")

app = FastAPI()
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")
templates = Jinja2Templates(directory="templates")

# --- Load Model and Preprocessing ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

classes = ['drum', 'flute', 'guitar', 'tabla', 'violin']
classify_model = torch.load("../model/classify.pth", map_location=DEVICE, weights_only=False)
classify_model.eval()
classify_model.to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Gemini Model ---
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Helper Functions ---
def classify_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classify_model(input_tensor)
        predicted_idx = logits.argmax().item()
        return classes[predicted_idx]

def clean_json_string(s: str) -> str:
    """Remove Markdown-style triple backticks and language hints."""
    if s.startswith("```"):
        s = s.strip("` \n")  # remove leading/trailing backticks and newlines
        if s.startswith("json"):
            s = s[len("json"):].strip()  # remove "json" after ```
    return s

def generate_description(product_class: str, prop: str):     
    prompt = f"write a description about {product_class} with properties like {prop}. make features {prop} concise and clear and store them in a list. Return JSON with 'description' and 'features'."
    response = model.invoke(prompt)

    json_response = clean_json_string(response.content)
    try:
        return json.loads(json_response)
    except:
        return {"description": "N/A", "features": []}

def save_product(product):
    data = []
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
    data.append(product)
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    products = []
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r') as f:
            products = json.load(f)
    return templates.TemplateResponse("index.html", {"request": request, "products": products})

from fastapi import Form

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    prop: str = Form(...)
):
    ext = file.filename.split(".")[-1]
    image_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{image_id}.{ext}"
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    product_class = classify_image(image_path)
    details = generate_description(product_class, prop)

    product = {
        "id": image_id,
        "class": product_class,
        "image": f"/images/{image_path.name}",
        "description": details.get("description", "N/A"),
        "features": details.get("features", [])
        # "prop": prop  # include the additional prop here
    }
    save_product(product)
    # return templates.TemplateResponse("index.html", {"request": request, "products": product})
    return {"status": "success", "product": product}
