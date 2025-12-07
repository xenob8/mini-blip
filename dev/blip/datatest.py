# dataset.py
import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Трансформации для изображений (ViT-подобные)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class VizWizDataset(Dataset):
    def __init__(self, img_dir, ann_file, tokenizer, max_len=32):
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Фильтрация плохих примеров
        self.annotations = [
            a for a in data["annotations"]
            if not a.get("is_rejected", False)
            and not a.get("is_precanned", False)
            and isinstance(a.get("caption", None), str)
            and len(a["caption"].strip()) > 0
            # and a["image_id"] < 24000
        ]

        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        # Префикс имени файлов для train
        # self.prefix = "VizWiz_train_"
        self.prefix = "VizWiz_val_"

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        img_path = os.path.join(self.img_dir, f"{self.prefix}{image_id-23431:08d}.jpg")

        # Изображение
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Текст: добавляем BOS/EOS
        caption = ann["caption"].strip()
        text = f"{self.tokenizer.bos_token} {caption} {self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)        # [max_len]
        labels = input_ids.clone()
        # Игнорируем PAD в лоссе
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return image, input_ids, labels


class ImageGoingDataset(Dataset):
    def __init__(self, img_dir, ann_file, tokenizer, max_len=32):
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Фильтрация плохих примеров
        self.images = [
            a for a in data["images"]
            # and a["image_id"] < 24000
        ]

        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        # Префикс имени файлов для train
        # self.prefix = "VizWiz_train_"
        self.prefix = "VizWiz_val_"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        dct = self.images[idx]
        image_id = dct["id"]
        img_path = os.path.join(self.img_dir, f"{self.prefix}{image_id-23431:08d}.jpg")

        # Изображение
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)




        return image, image_id


class JustEasyImage(Dataset):
    def __init__(self, img_dir, ann_file, tokenizer, max_len=32):
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Фильтрация плохих примеров
        self.images = [
            a for a in data["images"]
            # and a["image_id"] < 24000
        ]

        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        # Префикс имени файлов для train
        # self.prefix = "VizWiz_train_"
        self.prefix = "VizWiz_val_"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        dct = self.images[idx]
        image_id = dct["id"]
        img_path = os.path.join(self.img_dir, f"{self.prefix}{image_id-23431:08d}.jpg")

        # Изображение
        image = Image.open(img_path)

        return image, image_id