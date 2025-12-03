"""
GPT-2 Model for Music Generation
12-layer transformer with causal attention
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Optional, Dict, Tuple
import math


class MusicGPT2Config:
    """Configuration for Music GPT-2 model"""

    def __init__(
        self,
        vocab_size: int = 336,
        n_positions: int = 2048,
        n_embd: int = 512,
        n_layer: int = 12,
        n_head: int = 8,
        n_inner: int = 2048,
        activation_function: str = 'gelu_new',
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def to_gpt2_config(self) -> GPT2Config:
        """Convert to HuggingFace GPT2Config"""
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.n_positions,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_inner=self.n_inner,
            activation_function=self.activation_function,
            resid_pdrop=self.resid_pdrop,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            layer_norm_epsilon=self.layer_norm_epsilon,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )


class MusicGPT2(nn.Module):
    """
    GPT-2 model for music generation using HuggingFace transformers

    Features:
    - 12-layer transformer
    - Causal (autoregressive) attention
    - REMI token vocabulary with chord events
    """

    def __init__(self, config: MusicGPT2Config):
        super().__init__()
        self.config = config

        # Create HuggingFace GPT2 model
        gpt2_config = config.to_gpt2_config()
        self.transformer = GPT2LMHeadModel(gpt2_config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]

        Returns:
            Dictionary with 'loss' and 'logits'
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_length: int = 4096,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        num_bars: int = 32,
        bar_token_id: int = None,
        eos_token_id: int = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate music sequence

        Args:
            input_ids: Optional seed sequence [1, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_bars: Target number of bars to generate
            bar_token_id: Token ID for bar marker
            eos_token_id: Token ID for end of sequence
            device: Device to generate on

        Returns:
            Generated token sequence [1, generated_len]
        """
        self.eval()

        if input_ids is None:
            input_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        else:
            input_ids = input_ids.to(device)

        generated = input_ids  # For model input (may be truncated)
        all_tokens = input_ids.clone()  # Keep full sequence for output
        bar_count = 0
        past_key_values = None  # KV cache
        last_truncation_bar_count = -1  # Track bar count at last truncation
        stall_count = 0  # Count consecutive truncations without new bars

        # Generate until we reach num_bars or max_length
        while all_tokens.shape[1] < max_length:
            # Check if we need to truncate (invalidates cache)
            if generated.shape[1] >= self.config.n_positions:
                # Check if we're stalling (no new bars since last truncation)
                if bar_count == last_truncation_bar_count:
                    stall_count += 1
                else:
                    stall_count = 0
                last_truncation_bar_count = bar_count

                # If stalling, inject a bar token to force continuation
                if stall_count >= 2 and bar_token_id is not None:
                    # Truncate to shorter length and inject bar token
                    keep_length = self.config.n_positions // 4
                    generated = generated[:, -keep_length:]

                    # Inject bar token to force model to continue from new bar
                    bar_token_tensor = torch.tensor([[bar_token_id]], device=device)
                    generated = torch.cat([generated, bar_token_tensor], dim=1)
                    all_tokens = torch.cat([all_tokens, bar_token_tensor], dim=1)
                    bar_count += 1
                    print(f"  [Stall Recovery] Injected Bar token, now at bar {bar_count}/{num_bars}", flush=True)
                    stall_count = 0

                    if bar_count >= num_bars:
                        break
                else:
                    # Normal truncation to half the max length
                    keep_length = self.config.n_positions // 2
                    generated = generated[:, -keep_length:]
                    print(f"  [Truncation] Model input truncated to {keep_length} tokens, bars so far: {bar_count}", flush=True)

                past_key_values = None  # Clear cache after truncation
                # Full forward pass after truncation
                input_for_model = generated
            else:
                # Use KV cache: only pass the last token
                if past_key_values is not None:
                    input_for_model = generated[:, -1:]
                else:
                    input_for_model = generated

            # Get model predictions with KV cache
            outputs = self.transformer(
                input_for_model,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values  # Save cache for next iteration

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Clamp token to valid vocabulary range
            next_token = torch.clamp(next_token, 0, self.config.vocab_size - 1)

            # Append to both sequences
            generated = torch.cat([generated, next_token], dim=1)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)

            # Count bars
            if bar_token_id is not None and next_token.item() == bar_token_id:
                bar_count += 1
                print(f"Bar {bar_count}/{num_bars} at token {all_tokens.shape[1]}", flush=True)
                if bar_count >= num_bars:
                    break

            # Check for EOS - but only stop if we've reached the target number of bars
            if eos_token_id is not None and next_token.item() == eos_token_id:
                if bar_count >= num_bars:
                    break
                # If we haven't reached num_bars yet, continue generating (ignore EOS)

        return all_tokens

    def save_pretrained(self, save_path: str):
        """Save model and config"""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save transformer
        self.transformer.save_pretrained(save_path)

        # Save custom config
        import json
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'n_positions': self.config.n_positions,
            'n_embd': self.config.n_embd,
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_inner': self.config.n_inner,
            'activation_function': self.config.activation_function,
            'resid_pdrop': self.config.resid_pdrop,
            'embd_pdrop': self.config.embd_pdrop,
            'attn_pdrop': self.config.attn_pdrop,
            'layer_norm_epsilon': self.config.layer_norm_epsilon,
            'pad_token_id': self.config.pad_token_id,
            'bos_token_id': self.config.bos_token_id,
            'eos_token_id': self.config.eos_token_id,
        }
        with open(os.path.join(save_path, 'music_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str) -> 'MusicGPT2':
        """Load model from checkpoint"""
        import json
        import os

        # Load custom config
        with open(os.path.join(load_path, 'music_config.json'), 'r') as f:
            config_dict = json.load(f)

        config = MusicGPT2Config(**config_dict)
        model = cls(config)

        # Load transformer weights
        model.transformer = GPT2LMHeadModel.from_pretrained(load_path)

        return model


def create_model(
    vocab_size: int,
    n_layer: int = 12,
    n_embd: int = 512,
    n_head: int = 8,
    max_length: int = 2048,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    dropout: float = 0.1
) -> MusicGPT2:
    """
    Create Music GPT-2 model

    Args:
        vocab_size: Size of vocabulary
        n_layer: Number of transformer layers (at least 12)
        n_embd: Embedding dimension
        n_head: Number of attention heads
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        dropout: Dropout rate

    Returns:
        MusicGPT2 model
    """
    config = MusicGPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    model = MusicGPT2(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"  - Layers: {n_layer}")
    print(f"  - Embedding dim: {n_embd}")
    print(f"  - Attention heads: {n_head}")
    print(f"  - Vocab size: {vocab_size}")

    return model


if __name__ == '__main__':
    # Test model creation
    model = create_model(
        vocab_size=336,
        n_layer=12,
        n_embd=512,
        n_head=8,
        max_length=2048
    )

    # Test forward pass
    batch_size = 4
    seq_len = 512
    input_ids = torch.randint(0, 336, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 336, (batch_size, seq_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # Test generation
    if torch.cuda.is_available():
        model = model.cuda()
        generated = model.generate(
            max_length=100,
            temperature=1.0,
            top_k=10,
            device='cuda'
        )
        print(f"Generated shape: {generated.shape}")
