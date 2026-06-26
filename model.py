import json
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# model registry
MODEL_REGISTRY = {
    "Geneformer-V1": {
        "config": "./assets/Geneformer-V1/config.json",
        "checkpoint": "./assets/Geneformer-V1/pretrained_GeneformerV1.pth",
    },
    "Geneformer-V2-104M": {
        "config": "./assets/Geneformer-V2-104M/config.json",
        "checkpoint": "./assets/Geneformer-V2-104M/pretrained_GeneformerV2-104M.pth",
    },
    "Geneformer-V2-104M_CLcancer": {
        "config": "./assets/Geneformer-V2-104M_CLcancer/config.json",
        "checkpoint": "./assets/Geneformer-V2-104M_CLcancer/pretrained_GeneformerV2-104M_CLcancer.pth",
    },
    "Geneformer-V2-316M": {
        "config": "./assets/Geneformer-V2-316M/config.json",
        "checkpoint": "./assets/Geneformer-V2-316M/pretrained_GeneformerV2-316M.pth",
    },
}

def _remap_qkv_state_dict(state, num_hidden_layers):
    """
    Convert checkpoints with separate query/key/value projections into
    fused qkv projection weights.

    Old keys:
        bert.encoder.layer.N.attention.self.query.weight
        bert.encoder.layer.N.attention.self.key.weight
        bert.encoder.layer.N.attention.self.value.weight

    New keys:
        bert.encoder.layer.N.attention.self.qkv.weight
    """
    state = dict(state)

    for i in range(num_hidden_layers):
        prefix = f"bert.encoder.layer.{i}.attention.self"

        q_w = f"{prefix}.query.weight"
        k_w = f"{prefix}.key.weight"
        v_w = f"{prefix}.value.weight"

        q_b = f"{prefix}.query.bias"
        k_b = f"{prefix}.key.bias"
        v_b = f"{prefix}.value.bias"

        fused_w = f"{prefix}.qkv.weight"
        fused_b = f"{prefix}.qkv.bias"

        if fused_w not in state and q_w in state and k_w in state and v_w in state:
            state[fused_w] = torch.cat(
                [state[q_w], state[k_w], state[v_w]],
                dim=0,
            )

        if fused_b not in state and q_b in state and k_b in state and v_b in state:
            state[fused_b] = torch.cat(
                [state[q_b], state[k_b], state[v_b]],
                dim=0,
            )

        state.pop(q_w, None)
        state.pop(k_w, None)
        state.pop(v_w, None)
        state.pop(q_b, None)
        state.pop(k_b, None)
        state.pop(v_b, None)

    return state


@dataclass
class GeneformerConfig:
    vocab_size: int = 25426
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 2048
    type_vocab_size: int = 2
    pad_token_id: int = 0
    hidden_dropout_prob: float = 0.02
    attention_probs_dropout_prob: float = 0.02
    layer_norm_eps: float = 1e-12
    hidden_act: str = "relu"
    initializer_range: float = 0.02

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        keep = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**keep)


class GeneformerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        position_ids = self.position_ids[:, :seq_len].expand(bsz, -1)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    #SDPA
    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
    
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(
            bsz,
            seq_len,
            3,
            self.num_attention_heads,
            self.head_dim,
        )
    
        # q, k, v: (B, heads, L, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
    
        dropout_p = self.dropout.p if self.training else 0.0
    
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
    
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.view(bsz, seq_len, self.all_head_size)


class GeneformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = nn.Module()
        self.output.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attn = self.self(hidden_states, attention_mask)
        attn = self.output.dense(attn)
        attn = self.output.dropout(attn)
        return self.output.LayerNorm(hidden_states + attn)


class GeneformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = F.relu if config.hidden_act == "relu" else F.gelu

    def forward(self, x):
        return self.act(self.dense(x))


class GeneformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, residual):
        x = self.dense(x)
        x = self.dropout(x)
        return self.LayerNorm(residual + x)


class GeneformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GeneformerAttention(config)
        self.intermediate = GeneformerIntermediate(config)
        self.output = GeneformerOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        x = self.attention(hidden_states, attention_mask)
        y = self.intermediate(x)
        return self.output(y, x)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [GeneformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_hidden_states=True,
        return_layer=None,
    ):
        all_hidden_states = [hidden_states] if output_hidden_states else None
        selected_hidden_state = None

        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states, attention_mask)
            layer_id = i + 1

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if return_layer is not None and layer_id == return_layer:
                selected_hidden_state = hidden_states

        return (
            hidden_states,
            tuple(all_hidden_states) if output_hidden_states else None,
            selected_hidden_state,
        )


class GeneformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = GeneformerEmbeddings(config)
        self.encoder = BertEncoder(config)

    def get_extended_attention_mask(self, attention_mask, dtype):
        mask = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * -10000.0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=True,
        return_layer=None,
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        extended_mask = self.get_extended_attention_mask(
            attention_mask,
            dtype=self.embeddings.word_embeddings.weight.dtype,
        )

        x = self.embeddings(input_ids, token_type_ids)

        last_hidden, hidden_states, selected_hidden_state = self.encoder(
            x,
            attention_mask=extended_mask,
            output_hidden_states=output_hidden_states,
            return_layer=return_layer,
        )

        return SimpleNamespace(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            selected_hidden_state=selected_hidden_state,
        )


class GeneformerPredictionHead(nn.Module):
    def __init__(self, config, embedding_weight):
        super().__init__()
        self.transform = nn.Module()
        self.transform.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform.act = F.relu if config.hidden_act == "relu" else F.gelu
        self.transform.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = embedding_weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform.dense(x)
        x = self.transform.act(x)
        x = self.transform.LayerNorm(x)
        return self.decoder(x)


class GeneformerForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = GeneformerModel(config)
        self.cls = nn.Module()
        self.cls.predictions = GeneformerPredictionHead(
            config,
            self.bert.embeddings.word_embeddings.weight,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=True,
        return_layer=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_layer=return_layer,
        )
    
        logits = self.cls.predictions(outputs.last_hidden_state)
    
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            selected_hidden_state=outputs.selected_hidden_state,
        )

    #compile
    def optimize_for_inference(
        self,
        compile_model=True,
        compile_mode="reduce-overhead",
    ):
        if compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile requires PyTorch >= 2.0")

            return torch.compile(
                self,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )

        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        map_location="cpu",
        strict=True,
        compile_model=True,
        compile_mode="reduce-overhead",
    ):
        """
        model = GeneformerForMaskedLM.from_pretrained("Geneformer-V1")
        model = GeneformerForMaskedLM.from_pretrained("Geneformer-V2-104M")
        model = GeneformerForMaskedLM.from_pretrained("Geneformer-V2-104M_CLcancer")
        model = GeneformerForMaskedLM.from_pretrained("Geneformer-V2-316M")
        """
    
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )
    
        repo_root = Path(__file__).resolve().parent
        cfg = MODEL_REGISTRY[model_name]
    
        config_path = repo_root / cfg["config"]
        checkpoint_path = repo_root / cfg["checkpoint"]
    
        config = GeneformerConfig.from_json(config_path)
        model = cls(config)
    
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        
        # HuggingFace BertForMaskedLM ties decoder weights with input embeddings.
        # Some checkpoints store only cls.predictions.bias and omit decoder.weight / decoder.bias.
        if "cls.predictions.decoder.weight" not in state:
            state["cls.predictions.decoder.weight"] = state[
                "bert.embeddings.word_embeddings.weight"
            ]
        
        if "cls.predictions.decoder.bias" not in state and "cls.predictions.bias" in state:
            state["cls.predictions.decoder.bias"] = state["cls.predictions.bias"]

        # position_ids is a non-persistent buffer in some HF BERT versions.
        # Some Geneformer checkpoints include it, some do not.
        # It is deterministic and recreated from config, so never load it from checkpoint.
        state.pop("bert.embeddings.position_ids", None)
        
        state = _remap_qkv_state_dict(
            state,
            num_hidden_layers=config.num_hidden_layers,
        )
        
        missing, unexpected = model.load_state_dict(state, strict=strict)
        
        if not strict:
            print(f"[nano-Geneformer] missing keys: {len(missing)}")
            print(f"[nano-Geneformer] unexpected keys: {len(unexpected)}")
        
        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        
        return model

    @torch.inference_mode()
    def encode(
        self,
        input_data,
        tokenizer=None,
        model_name=None,
        attention_mask=None,
        layer=-2,
        cell_emb_style="mean_pool",
        special_token=None,
        batch_size=64,
        device=None,
        use_amp=True,
    ):
        """
        Pure PyTorch embedding inference.
    
        If batch_size is None:
            encode the full tensor at once.
    
        If batch_size is int:
            encode in mini-batches and concatenate outputs.
    
        cell_emb_style:
        - "mean_pool": mean-pool token embeddings.
            If special_token=True, exclude first token (<cls>) and last real token (<eos>).
            If special_token=False, pool all non-padding tokens.
        - "cls": return first-token embedding.
        - "all": return full hidden states from selected layer.
    
        layer:
        - -1 = last layer
        - -2 = second-to-last layer
        Encode either:
        - input_data: path to .h5ad
        - input_data: input_ids tensor
    
        If input_data is a path, tokenizer is required unless model_name is given.
        """
    
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
    
        # Case 1: user passes path to h5ad
        if isinstance(input_data, (str, Path)):
            if tokenizer is None:
                if model_name is None:
                    raise ValueError(
                        "When input_data is a .h5ad path, pass tokenizer=... "
                        "or model_name=..."
                    )
    
                from Geneformer_tokenizer import GeneformerTokenizer
                tokenizer = GeneformerTokenizer.from_pretrained(model_name)
    
            encoded = tokenizer.encode_h5ad(input_data)
    
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
    
            if special_token is None:
                special_token = tokenizer.special_token
    
        # Case 2: user passes tensor directly
        else:
            input_ids = input_data
    
            if attention_mask is None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
    
            if special_token is None:
                special_token = False
    
        if batch_size is not None:
            embs = []
    
            for start in range(0, input_ids.shape[0], batch_size):
                end = min(start + batch_size, input_ids.shape[0])
    
                emb = self.encode(
                    input_data=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                    layer=layer,
                    cell_emb_style=cell_emb_style,
                    special_token=special_token,
                    batch_size=None,
                    device=device,
                    use_amp=use_amp,
                )
    
                embs.append(emb.cpu())
    
            return torch.cat(embs, dim=0)
    
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    
        autocast_enabled = use_amp and device.type == "cuda"
    
        # with torch.autocast(
        #     device_type=device.type,
        #     enabled=autocast_enabled,
        # ):
        #     outputs = self.forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         output_hidden_states=True,
        #     )
    
        # h = outputs.hidden_states[layer]
        num_layers = self.config.num_hidden_layers
        
        if layer < 0:
            return_layer = num_layers + 1 + layer
        else:
            return_layer = layer
        
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_layer=return_layer,
            )
        
        h = outputs.selected_hidden_state
        if h is None:
            h = outputs.last_hidden_state
    
        if cell_emb_style == "cls":
            return h[:, 0]
    
        if cell_emb_style == "all":
            return h
    
        if cell_emb_style != "mean_pool":
            raise ValueError(
                "cell_emb_style must be one of: 'mean_pool', 'cls', 'all'"
            )
    
        mask = attention_mask.clone()
    
        if special_token:
            mask[:, 0] = 0
    
            lengths = attention_mask.sum(dim=1)
            batch_idx = torch.arange(input_ids.size(0), device=device)
            eos_pos = lengths - 1
            valid = lengths > 1
            mask[batch_idx[valid], eos_pos[valid]] = 0
    
        mask = mask.unsqueeze(-1).to(h.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
    
        return (h * mask).sum(dim=1) / denom