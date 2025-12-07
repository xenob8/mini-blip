import random
import torch
import json

import torchinfo
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
from pr.copilot.blip import MiniBLIP
from pr.copilot.datatest import VizWizDataset, ImageGoingDataset  # Импорт класса из dataset.py

# -------------------------------
# 1. Токенизатор
# -------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
})

# -------------------------------
# 2. Модель
# -------------------------------
model = MiniBLIP(
    vocab_size=len(tokenizer),
    bos_id=tokenizer.bos_token_id,
    eos_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
)
model.load_state_dict(torch.load("../miniblip_epoch9.pt", map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
summary(
    model,
    input_data=(
        torch.randn(2, 3, 224, 224).to(device),                # изображения на GPU
        torch.randint(0, len(tokenizer), (2, 16)).to(device),  # токены на GPU
    ),
    dtypes=[torch.float32, torch.long],
    device=device,  # важно!
)

# -------------------------------
# 3. Датасет (только для изображений и аннотаций)
# -------------------------------
val_dataset = ImageGoingDataset(
    img_dir="../../vizwiz/val",
    ann_file="../../vizwiz/annotations/val.json",
    tokenizer=tokenizer,
    max_len=32,
)

# ОГРАНИЧИВАЕМ КОЛИЧЕСТВО ИЗОБРАЖЕНИЙ (пачка)
num_images = 500  # ←←← ИЗМЕНИ: None для всех, 1 для одного, 50 для пачки и т.д.
if num_images is not None:
    val_dataset.images = val_dataset.images[:num_images]  # Берем первые N

print(f"Обрабатываем {len(val_dataset)} изображений...")

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # Батч для скорости

# -------------------------------
# 4. Генерация предсказаний (батчевая)
# -------------------------------
# -------------------------------
# 4. Генерация предсказаний (батчевая) — ИСПРАВЛЕНО
# -------------------------------
predictions = []

with torch.no_grad():
    for images, im_ids in val_loader:  # input_ids и labels нам в inference не нужны
        images = images.to(device)
        gen_ids = model.generate(images, tokenizer, max_length=32)
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for i, im_id in enumerate(im_ids):
            predictions.append({
                "image_id": int(im_id),
                "caption": gen_texts[i]
            })


# -------------------------------
# 5. Сохранение в COCO-формате
# -------------------------------
with open("results_batch.json", "w") as f:
    json.dump(predictions, f, indent=2)

print(f"Готово! {len(predictions)} предсказаний сохранено в results_batch.json")

