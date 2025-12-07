import os
import torch
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from transformers import GPT2TokenizerFast

from blip import MiniBLIP  # Ensure this module is available in the container

# Environment/config
MODEL_PATH = os.getenv("MODEL_PATH", "miniblip_epoch9.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="MiniBLIP Captioning Service", version="1.0.0")

# Globals for model/tokenizer
tokenizer = None
model = None

def load_model(path: str, tokenizer: GPT2TokenizerFast) -> MiniBLIP:
    model = MiniBLIP(
        vocab_size=len(tokenizer),
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
    )
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

@app.on_event("startup")
def startup_event():
    global tokenizer, model
    # Load tokenizer and add special tokens
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    })
    # Load model
    try:
        model = load_model(MODEL_PATH, tokenizer)
        model.to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/generate")
async def generate_caption(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    try:
        # Read image bytes
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Generate
    try:
        with torch.no_grad():
            ids = model.generate(
                img_tensor,
                tokenizer,
                max_length=32,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    return JSONResponse({"caption": text})
