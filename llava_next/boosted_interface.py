#!/usr/bin/env python3
"""
LLaVA-Next Boosted Forward Interface

Uses the strategies module for attention modification.
This file only handles the model-specific hooking logic.

Author: AdaVBoost Project
"""

import math
import warnings
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*past_key_value.*past_key_values.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import BaseBoostStrategy, AdaVBoostStrategy


class LLaVABoostedInterface:
    """
    LLaVA-Next interface with strategy-based attention boost.

    The boost logic is delegated to the strategies module.
    This class only handles LLaVA-specific model hooking.
    """

    def __init__(
        self,
        model_name_or_path: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        strategy: BaseBoostStrategy = None,
    ):
        print(f"Loading LLaVA-Next model: {model_name_or_path}")

        load_kwargs = {
            "dtype": torch_dtype,
            "device_map": device_map,
            "attn_implementation": "eager",
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            except ImportError:
                print("Warning: bitsandbytes not available, using float16 instead")

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_name_or_path)

        self.device = next(self.model.parameters()).device
        self._image_token_id = self.model.config.image_token_index

        # Strategy for boost (can be swapped at runtime)
        self.strategy = strategy

        # Visual token state
        self._visual_indices_tensor = None

        # Store original forwards
        self._original_forwards = {}

        # Patch attention layers
        self._patch_attention_layers()

        print(f"LLaVA Boosted Interface initialized!")
        print(f"  Device: {self.device}")
        print(f"  Strategy: {self.strategy}")

    def _patch_attention_layers(self):
        """Patch attention layers to use strategy."""
        if hasattr(self.model, 'language_model'):
            lm = self.model.language_model
        else:
            lm = self.model

        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            layers = lm.model.layers
        elif hasattr(lm, 'layers'):
            layers = lm.layers
        else:
            print("Warning: Could not find transformer layers")
            return

        self._num_layers = len(layers)
        print(f"Found {self._num_layers} layers, patching attention...")

        for i, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                self._original_forwards[i] = attn.forward
                attn.forward = self._create_boosted_forward(attn, i)

        print(f"Patched {self._num_layers} attention layers")

    def _create_boosted_forward(self, attn_module, layer_idx):
        """Create forward function that uses the strategy."""
        original_forward = self._original_forwards[layer_idx]
        interface = self

        def boosted_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ):
            # Check if strategy is enabled
            strategy = interface.strategy
            visual_indices = interface._visual_indices_tensor

            apply_boost = (
                strategy is not None and
                strategy.enabled and
                visual_indices is not None and
                visual_indices.numel() > 0
            )

            if not apply_boost:
                # Pass all arguments directly to original forward
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            # === Manual attention with strategy-based boost ===
            cache = past_key_value
            if cache is None and 'past_key_values' in kwargs:
                cache = kwargs['past_key_values']

            bsz, q_len, hidden_size = hidden_states.size()

            config = attn_module.config if hasattr(attn_module, 'config') else interface.model.config
            num_heads = config.num_attention_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            head_dim = config.hidden_size // num_heads

            # Q, K, V projections
            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Handle KV cache
            kv_seq_len = key_states.shape[-2]
            if cache is not None:
                if hasattr(cache, 'get_usable_length'):
                    kv_seq_len += cache.get_usable_length(kv_seq_len, layer_idx)
                elif isinstance(cache, tuple) and len(cache) > 0:
                    kv_seq_len += cache[0].shape[-2]

            # Apply rotary embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)

            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # Update cache
            if cache is not None:
                if hasattr(cache, 'update'):
                    cache_kwargs = {"sin": sin, "cos": cos}
                    if cache_position is not None:
                        cache_kwargs["cache_position"] = cache_position
                    key_states, value_states = cache.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )
                elif isinstance(cache, tuple):
                    key_states = torch.cat([cache[0], key_states], dim=2)
                    value_states = torch.cat([cache[1], value_states], dim=2)

            # GQA: repeat KV heads
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            # Compute attention weights
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            # Apply causal mask
            if attention_mask is not None:
                causal_mask = attention_mask
                if causal_mask.dim() == 4:
                    attn_weights = attn_weights + causal_mask

            # === APPLY STRATEGY (PRE-SOFTMAX) ===
            current_kv_len = attn_weights.shape[-1]
            attn_weights = strategy.apply(
                attn_weights,
                visual_indices,
                layer_idx=layer_idx,
                current_kv_len=current_kv_len,
            )
            
            # === END STRATEGY ===
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Apply to values
            attn_output = torch.matmul(attn_weights, value_states)

            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, hidden_size)
            attn_output = attn_module.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights

        return boosted_forward

    def set_visual_indices(self, visual_token_indices: List[int]):
        """
        Set visual token indices for attention modification.

        Args:
            visual_token_indices: Visual token positions
        """
        if visual_token_indices:
            self._visual_indices_tensor = torch.tensor(
                visual_token_indices, dtype=torch.long, device=self.device
            )
        else:
            self._visual_indices_tensor = None

    def clear_visual_indices(self):
        """Clear visual token indices."""
        self._visual_indices_tensor = None

    def set_strategy(self, strategy: BaseBoostStrategy):
        """Replace the current strategy."""
        self.strategy = strategy
        return self

    def eval(self):
        self.model.eval()
        return self

    def prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def detect_visual_token_indices(self, input_ids: torch.Tensor) -> List[int]:
        """Detect visual token positions."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        visual_indices = (input_ids == self._image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(visual_indices, int):
            visual_indices = [visual_indices]
        return visual_indices

    def setup_vge_grounding(self, inputs: Dict[str, torch.Tensor], strategy) -> bool:
        """Compute and set grounding scores for VGE scoring.

        Call this after prepare_inputs() but before generation.
        Only needed when using AdaVBoostStrategy with VGE.

        Args:
            inputs: Model inputs from prepare_inputs()
            strategy: AdaVBoostStrategy instance

        Returns:
            True if grounding scores were computed successfully
        """
        # Get image token range from detect_visual_token_indices
        visual_indices = self.detect_visual_token_indices(inputs['input_ids'])
        if not visual_indices:
            return False

        image_start = visual_indices[0]
        image_end = visual_indices[-1] + 1

        # Run prefill to get logits
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=False,
            )
            prefill_logits = outputs.logits[0]  # (seq_len, vocab_size)

        if image_end <= prefill_logits.shape[0]:
            strategy.set_grounding_scores(prefill_logits, image_start, image_end)
            return True

        return False


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Testing LLaVA Boosted Interface with AdaVBoostStrategy")
    print("=" * 60)

    strategy = AdaVBoostStrategy(model_name='llava')
    strategy.enable()

    model = LLaVABoostedInterface(load_in_4bit=True, strategy=strategy)
    model.eval()

    print(f"\nStrategy: {model.strategy}")
    print("Test completed!")
