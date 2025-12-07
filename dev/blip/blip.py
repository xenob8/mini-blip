# pr/mini_blip.py
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, GPT2Config, GPT2LMHeadModel

# === 1. Tiny ViT ===
vit_config = ViTConfig(
    hidden_size=192,
    num_hidden_layers=6,
    num_attention_heads=4,
    intermediate_size=768,
    image_size=224,
    patch_size=16,
)
vit = ViTModel(vit_config)

# === 2. Q-Former (минимальный) ===
class QFormer(nn.Module):
    def __init__(self, dim=192, n_queries=8):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=768,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):  # x: [B, 197, 192]
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        x = torch.cat([q, x], dim=1)
        x = self.transformer(x)
        return x[:, :self.n_queries, :]  # [B, 8, 192]

# === 3. Tiny GPT-2 Decoder ===
# ВАЖНО: vocab_size и спец-токены будут заменены после инициализации по tokenizer
gpt_config = GPT2Config(
    vocab_size=50257,        # временно; потом resize по tokenizer.vocab_size
    n_positions=128,
    n_embd=256,
    n_layer=4,
    n_head=4,
    bos_token_id=None,       # выставим в модели позже
    eos_token_id=None,
    pad_token_id=None,
)
decoder = GPT2LMHeadModel(gpt_config)

# === 4. MiniBLIP ===
class MiniBLIP(nn.Module):
    def __init__(self, vocab_size, bos_id, eos_id, pad_id):
        super().__init__()
        self.vit = vit
        self.qformer = QFormer()
        self.decoder = decoder
        # Устанавливаем спец-токены
        self.decoder.config.vocab_size = vocab_size
        self.decoder.config.bos_token_id = bos_id
        self.decoder.config.eos_token_id = eos_id
        self.decoder.config.pad_token_id = pad_id
        # Resize эмбеддингов под tokenizer
        self.decoder.resize_token_embeddings(vocab_size)

        self.proj = nn.Linear(192, 256)
        self.num_image_tokens = self.qformer.n_queries

    def forward(self, images, input_ids, labels=None):
        # Vision
        vit_out = self.vit(images).last_hidden_state        # [B, 197, 192]
        img_emb = self.qformer(vit_out)                     # [B, 8, 192]
        img_emb = self.proj(img_emb)                        # [B, 8, 256]

        # Text
        text_emb = self.decoder.transformer.wte(input_ids)  # [B, T, 256]

        # Concat: image + text
        decoder_inputs = torch.cat([img_emb, text_emb], dim=1)  # [B, 8+T, 256]

        # Attention mask: все токены активны
        B, S, _ = decoder_inputs.shape
        attention_mask = torch.ones((B, S), device=decoder_inputs.device, dtype=torch.long)

        # Labels: prepend -100 для визуальных токенов
        shifted_labels = None
        if labels is not None:
            ignore_tokens = torch.full(
                (labels.shape[0], self.num_image_tokens),
                fill_value=-100,
                device=labels.device,
                dtype=labels.dtype,
            )
            shifted_labels = torch.cat([ignore_tokens, labels], dim=1)  # [B, 8+T]

        outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=attention_mask,
            labels=shifted_labels,
            use_cache=False,
        )
        return outputs

    @torch.no_grad()
    def generate(self, images, tokenizer, max_length=32, temperature=0.7, top_k=50, top_p=0.9):
        self.eval()
        vit_out = self.vit(images).last_hidden_state
        img_emb = self.qformer(vit_out)
        img_emb = self.proj(img_emb)

        device = images.device
        input_ids = torch.full((images.shape[0], 1), tokenizer.bos_token_id, device=device, dtype=torch.long)

        for _ in range(max_length):
            text_emb = self.decoder.transformer.wte(input_ids)
            decoder_inputs = torch.cat([img_emb, text_emb], dim=1)
            attention_mask = torch.ones(
                (decoder_inputs.size(0), decoder_inputs.size(1)),
                device=device, dtype=torch.long
            )

            logits = self.decoder(
                inputs_embeds=decoder_inputs,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :]  # [B, vocab]

            # Sampling
            logits = logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                mask = torch.zeros_like(probs)
                mask.scatter_(1, topk_idx, topk_vals)
                probs = mask
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > top_p).float()
                cutoff[:, 0] = 0.0
                sorted_probs = sorted_probs * (1.0 - cutoff)
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # EOS check (все батчи, если хотя бы один дошёл — продолжаем остальные)
            if ((next_token == tokenizer.eos_token_id).all().item()):
                break

        return input_ids

