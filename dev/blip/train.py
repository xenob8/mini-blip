# train.py
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup

from dev.blip.blip import MiniBLIP
from dev.blip.datatest import VizWizDataset


def main():
    # 1. Токенизатор (GPT-2) + спец-токены
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    tokenizer.add_special_tokens(special_tokens)

    # 2. Датасет
    dataset = VizWizDataset(
        img_dir="../vizwiz/train",
        ann_file="../vizwiz/annotations/train.json",
        tokenizer=tokenizer,
        max_len=32,
    )

    # 3. DataLoader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # 4. Модель
    model = MiniBLIP(
        vocab_size=len(tokenizer),
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # (опционально) — заморозить ViT на первые эпохи
    for p in model.vit.parameters():
        p.requires_grad = False

    # 5. Оптимизатор + шедулер
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    total_steps = len(loader) * 10
    warmup_steps = max(100, total_steps // 20)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 6. AMP
    scaler = GradScaler()

    # 7. Обучение
    model.train()
    for epoch in range(10):
        total_loss = 0.0

        # Разморозим ViT после 2 эпох (если нужно)
        if epoch == 2:
            for p in model.vit.parameters():
                p.requires_grad = True

        for batch_idx, (images, input_ids, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                outputs = model(images, input_ids, labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"\n=== Epoch {epoch} finished, Avg Loss: {avg_loss:.4f} ===\n")

        # Сохранение модели
        torch.save(model.state_dict(), f"miniblip_epoch{epoch}.pt")

    print("Обучение завершено!")


if __name__ == "__main__":
    main()
