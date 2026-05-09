from tokenizers import Tokenizer
import torch
import torch.nn as nn

MAX_LEN = 256

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=256,
        n_heads=4,
        n_layers=3,
        ffn_dim=1024,
        max_len=MAX_LEN,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed   = nn.Embedding(max_len,    d_model)
        self.embed_drop  = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN → more stable training
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm     = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x    = self.embed_drop(self.token_embed(input_ids) + self.pos_embed(pos))

        # src_key_padding_mask: True where position should be IGNORED
        pad_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Masked mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(x)



class InferenceModel:
    def __init__(self, model, tokenizer, label2idx):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.label2idx = label2idx
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.max_len = model.max_len
        self.compiled = False
        self.constant_bz = None
        self.device = next(model.parameters()).device

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def predict(self, texts, top_k=1):
        input_ids, attention_mask = self.encode(texts)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            if top_k == 1:
                topk_probs, topk_indices = torch.max(probs, dim=-1)
            else:
                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
        if top_k == 1:
            labels = [self.idx2label[topk_indices[i].item()] for i in range(topk_indices.shape[0])]
        else:
            labels = [[self.idx2label[idx.item()] for idx in sample_indices] for sample_indices in topk_indices]
        return labels, topk_probs.cpu().numpy()
    
    @classmethod
    def from_checkpoint(cls, ckpt_path, tokenizer_path="tokenizer.json", device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt['config']
        tok  = Tokenizer.from_file(tokenizer_path)
        m    = TransformerClassifier(**cfg).to(device)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        inst =  cls(m, tok, ckpt['label2idx'])
        inst.device = device
        return inst
    
    def encode(self, texts):
        input_ids, attn_masks = [], []
        if self.compiled and len(texts) < self.constant_bz:
            texts = texts + ["Hello, how are you?"] * (self.constant_bz - len(texts))
        for txt in texts:
            enc = self.tokenizer.encode(txt)
            ids = enc.ids[:self.max_len]
            mask = enc.attention_mask[:self.max_len]
            pad = self.max_len - len(ids)
            ids = ids + [0] * pad
            mask = mask + [0] * pad
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attn_masks.append(torch.tensor(mask, dtype=torch.long))
    
        input_ids = torch.stack(input_ids)
        attn_masks = torch.stack(attn_masks)
        return input_ids, attn_masks

    
    def compile_model(self, constant_bz=32, mode="max-autotune"):
        self.constant_bz = constant_bz   
        self.model = torch.compile(self.model, mode=mode)
        # compile the model with dummy texts of the constant batch size
        dummy_texts = ["Hello, how are you?"] * constant_bz
        dummy_input_ids, dummy_attn_masks = self.encode(dummy_texts)
        dummy_input_ids = dummy_input_ids.to(self.device)
        dummy_attn_masks = dummy_attn_masks.to(self.device)
        with torch.no_grad():
            self.model(dummy_input_ids, dummy_attn_masks)
        self.compiled = True

    def predict_compiled(self, texts, top_k=1):
        assert self.compiled, "Model is not compiled. Call compile_model() first."
        device = next(self.model.parameters()).device

        all_labels = []
        all_probs  = []

        for chunk_start in range(0, len(texts), self.constant_bz):
            chunk = texts[chunk_start : chunk_start + self.constant_bz]
            real_n = len(chunk)

            # Pad the chunk to exactly constant_bz
            if real_n < self.constant_bz:
                chunk = chunk + ["Hello, how are you?"] * (self.constant_bz - real_n)

            input_ids, attention_mask = self.encode(chunk)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs  = torch.softmax(logits, dim=-1)
                if top_k == 1:
                    topk_probs, topk_indices = torch.max(probs, dim=-1)
                else:
                    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            # Keep only the real (non-padded) results
            topk_probs   = topk_probs[:real_n]
            topk_indices = topk_indices[:real_n]

            if top_k == 1:
                chunk_labels = [self.idx2label[topk_indices[i].item()] for i in range(real_n)]
            else:
                chunk_labels = [
                    [self.idx2label[idx.item()] for idx in sample_indices]
                    for sample_indices in topk_indices
                ]

            all_labels.extend(chunk_labels)
            all_probs.append(topk_probs.cpu())

        all_probs = torch.cat(all_probs, dim=0).numpy()
        return all_labels, all_probs
