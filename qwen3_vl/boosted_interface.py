#!/usr/bin/env python3
"""
Qwen3-VL Boosted Forward Interface

Uses the strategies module for attention modification.
This file only handles the model-specific hooking logic.

Author: SAVVA Project
"""

import math
import warnings
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*past_key_value.*past_key_values.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import BaseBoostStrategy, SAVVAStrategy

# Import Qwen3-VL specific functions
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_multimodal_rotary_pos_emb
    HAS_QWEN3_ROPE = True
except ImportError:
    HAS_QWEN3_ROPE = False

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_multimodal_rotary_pos_emb(q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

# Helper import is done lazily in prepare_inputs to avoid path issues
def _get_build_inputs():
    """Lazy import of build_inputs to handle path issues."""
    try:
        motivation_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Motivation')
        if motivation_path not in sys.path:
            sys.path.insert(0, motivation_path)
        from helper import build_inputs
        return build_inputs
    except ImportError:
        return None


class QwenBoostedInterface:
    """
    Qwen3-VL interface with strategy-based attention boost.

    The boost logic is delegated to the strategies module.
    This class only handles Qwen-specific model hooking.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        strategy: BaseBoostStrategy = None,
    ):
        print(f"Loading Qwen3-VL model: {model_name_or_path}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.device = next(self.model.parameters()).device

        # Get image token id for CFG
        if hasattr(self.model.config, 'image_token_id'):
            self._image_token_id = self.model.config.image_token_id
        else:
            self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')

        # Strategy for boost (can be swapped at runtime)
        self.strategy = strategy

        # Visual token state
        self._visual_indices_tensor = None

        # Store original forwards
        self._original_forwards = {}
        self._num_layers = 0

        # Patch attention layers
        self._patch_attention_layers()

        print(f"Qwen Boosted Interface initialized!")
        print(f"  Device: {self.device}")
        print(f"  Strategy: {self.strategy}")

    def _patch_attention_layers(self):
        """Patch attention layers to use strategy."""
        layers = None

        if (hasattr(self.model, 'model') and
            hasattr(self.model.model, 'language_model') and
            hasattr(self.model.model.language_model, 'layers')):
            layers = self.model.model.language_model.layers
            print("Found layers at: model.model.language_model.layers")
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            print("Found layers at: model.model.layers")

        if layers is None:
            print("Warning: Could not find model layers")
            return

        self._num_layers = len(layers)
        print(f"Found {self._num_layers} layers, patching attention...")

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
                self._original_forwards[layer_idx] = attn_module.forward
                attn_module.forward = self._create_boosted_forward(attn_module, layer_idx)

        print(f"Patched {self._num_layers} attention layers")

    def _create_boosted_forward(self, attn_module, layer_idx):
        """Create forward function that uses the strategy."""
        original_forward = self._original_forwards[layer_idx]
        interface = self

        def boosted_forward(
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            cache_position: Optional[torch.LongTensor] = None,
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
                return original_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            # === Manual attention with strategy-based boost ===
            cache = past_key_values

            bsz, q_len, hidden_size = hidden_states.size()

            # Get config
            if hasattr(attn_module, 'config'):
                config = attn_module.config
            elif hasattr(interface.model.config, 'text_config'):
                config = interface.model.config.text_config
            else:
                config = interface.model.config

            head_dim = getattr(attn_module, 'head_dim', getattr(config, 'head_dim', 128))

            if hasattr(attn_module, 'num_heads'):
                num_heads = attn_module.num_heads
            elif hasattr(config, 'num_attention_heads'):
                num_heads = config.num_attention_heads
            else:
                num_heads = attn_module.q_proj.out_features // head_dim

            if hasattr(attn_module, 'num_key_value_heads'):
                num_kv_heads = attn_module.num_key_value_heads
            elif hasattr(config, 'num_key_value_heads'):
                num_kv_heads = config.num_key_value_heads
            else:
                num_kv_heads = num_heads

            # Q, K, V projections
            hidden_shape = (bsz, q_len, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_states).view(hidden_shape)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape)

            # Apply Q/K normalization
            if hasattr(attn_module, 'q_norm') and attn_module.q_norm is not None:
                query_states = attn_module.q_norm(query_states)
            if hasattr(attn_module, 'k_norm') and attn_module.k_norm is not None:
                key_states = attn_module.k_norm(key_states)

            # Transpose
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            # Apply rotary embeddings
            cos, sin = None, None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            # Update cache
            if cache is not None:
                if hasattr(cache, 'update'):
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = cache.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )
                elif isinstance(cache, tuple):
                    key_states = torch.cat([cache[0], key_states], dim=2)
                    value_states = torch.cat([cache[1], value_states], dim=2)

            # GQA
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            # Compute attention
            scaling = getattr(attn_module, 'scaling', 1.0 / math.sqrt(head_dim))
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

            # Apply causal mask
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    attn_weights = attn_weights + attention_mask

            # === APPLY STRATEGY ===
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

            output_attentions = kwargs.get('output_attentions', False)
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
        build_inputs = _get_build_inputs()
        if build_inputs is not None:
            return build_inputs(self.processor, image, prompt, self.device, enable_thinking=False)
        # Fallback
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def detect_visual_token_indices(self, input_ids: torch.Tensor) -> List[int]:
        """Detect visual token indices in the input."""
        if hasattr(self.model.config, 'image_token_id'):
            image_token_id = self.model.config.image_token_id
        else:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')

        if image_token_id is None or image_token_id < 0:
            vision_start = getattr(self.model.config, 'vision_start_token_id', None)
            vision_end = getattr(self.model.config, 'vision_end_token_id', None)

            if vision_start is not None and vision_end is not None:
                ids = input_ids[0].tolist()
                indices = []
                in_vision = False
                for i, tok_id in enumerate(ids):
                    if tok_id == vision_start:
                        in_vision = True
                    elif tok_id == vision_end:
                        in_vision = False
                    elif in_vision:
                        indices.append(i)
                return indices
            return []

        indices = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
        return indices

    def setup_vge_grounding(self, inputs: Dict[str, torch.Tensor], strategy) -> bool:
        """Compute and set grounding scores for VGE scoring.

        Call this after prepare_inputs() but before generation.
        Only needed when using SAVVAStrategy with VGE.

        Args:
            inputs: Model inputs from prepare_inputs()
            strategy: SAVVAStrategy instance

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
    print("=" * 60)
    print("Testing Qwen Boosted Interface with SAVVAStrategy")
    print("=" * 60)

    strategy = SAVVAStrategy(model_name='qwen')
    strategy.enable()

    model = QwenBoostedInterface(strategy=strategy)
    model.eval()

    print(f"\nStrategy: {model.strategy}")
    print("Test completed!")
