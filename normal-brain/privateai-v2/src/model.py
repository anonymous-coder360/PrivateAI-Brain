"""
PrivateAI Model Architecture (FROM-SCRATCH - for reference only)

This is the ORIGINAL from-scratch transformer architecture.
It is NOT compatible with the fine-tuned SmolLM-360M model.

The fine-tuned model uses SmolLM-360M's native architecture (LlamaForCausalLM).
For inference, use llama.cpp directly via llama_cpp.Llama (see test_gguf.py, rag_integration.py).

This code is kept for reference/documentation of the from-scratch architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivateAIConfig:
    """Configuration for from-scratch PrivateAI model"""
    vocab_size = 3006
    context_length = 512
    embedding_dim = 512
    num_heads = 8
    num_layers = 12
    dropout = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""

    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.triu(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool
            ),
            diagonal=1
        )
        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False
        )

    def forward(self, x, key_padding_mask=None):
        sequence_length = x.size(1)
        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]
        output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return self.dropout(output)


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.embedding_dim * 4
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attention(self.norm1(x), key_padding_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class PrivateAIModel(nn.Module):
    """PrivateAI from-scratch transformer model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Weight tying
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens, padding_mask=None):
        batch_size, sequence_length = tokens.shape

        if sequence_length > self.config.context_length:
            tokens = tokens[:, -self.config.context_length:]
            sequence_length = tokens.shape[1]

        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.final_norm(x)
        logits = self.output(x)
        return logits


if __name__ == "__main__":
    # Test from-scratch model
    config = PrivateAIConfig()
    model = PrivateAIModel(config)
    test_input = torch.randint(0, config.vocab_size, (2, 32))
    output = model(test_input)
    parameters = sum(p.numel() for p in model.parameters())
    print("Model:", model.__class__.__name__)
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
    print("Parameters:", f"{parameters:,}")
    print("NOTE: This is the FROM-SCRATCH architecture (3006 vocab, 12 layers)")
    print("      NOT compatible with fine-tuned SmolLM-360M (49155 vocab, 30 layers)")
    print("      For inference, use llama.cpp with the fine-tuned model.")