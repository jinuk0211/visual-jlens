# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face adapters for Qwen3-VL and Gemma 4 Unified.

The adapters intentionally use duck typing rather than importing concrete
Transformers model classes.  This keeps :mod:`jlens` importable on a
Transformers release that does not yet contain Gemma 4 Unified, while still
failing early with a useful architecture check when an incompatible model is
passed.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from jlens.multimodal import (
    ActivationSite,
    MultimodalSample,
    PreparedMultimodalExample,
)


def _model_type(model: nn.Module) -> str:
    return str(getattr(getattr(model, "config", None), "model_type", ""))


def _text_config(model: nn.Module) -> Any:
    config = model.config
    getter = getattr(config, "get_text_config", None)
    return getter() if getter is not None else config.text_config


def _as_mapping(batch_feature: Any) -> dict[str, Any]:
    if hasattr(batch_feature, "items"):
        return dict(batch_feature.items())
    raise TypeError("processor did not return a mapping-like BatchFeature")


def _move_tensors(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _changed_span(
    actual: torch.Tensor, empty: torch.Tensor, *, name: str
) -> tuple[int, int]:
    """Return the span in ``actual`` inserted relative to ``empty``."""

    a = actual.tolist()
    b = empty.tolist()
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(a) - prefix
        and suffix < len(b) - prefix
        and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]
    ):
        suffix += 1
    end = len(a) - suffix
    if end <= prefix:
        raise ValueError(f"could not locate {name} content tokens in chat template")
    return prefix, end


def _content(image: Any, text: str) -> list[dict[str, Any]]:
    return [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]


class _HfMultimodalAdapterBase:
    def __init__(self, hf_model: nn.Module, processor: Any) -> None:
        self.hf_model = hf_model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.outer_model = hf_model.model
        self.language_model = self.outer_model.language_model
        self.layers = self.language_model.layers
        self.final_norm = self.language_model.norm
        self.lm_head = hf_model.lm_head
        config = _text_config(hf_model)
        self.n_layers = int(config.num_hidden_layers)
        self.d_model = int(config.hidden_size)
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config says {self.n_layers} decoder layers but found {len(self.layers)}"
            )
        hf_model.eval()
        for parameter in hf_model.parameters():
            parameter.requires_grad_(False)
        self.model_id = str(
            getattr(hf_model.config, "_name_or_path", None) or type(hf_model).__name__
        )
        self.revision = getattr(hf_model.config, "_commit_hash", None)
        self._softcap = getattr(config, "final_logit_softcapping", None)

    @property
    def input_device(self) -> torch.device:
        return self.language_model.embed_tokens.weight.device

    def _site_tag(self, block_index: int | None) -> str:
        return "decoder"

    def _make_sites(self) -> list[ActivationSite]:
        sites = [
            ActivationSite(
                "embed",
                self.layers[0],
                hook="pre",
                block_index=None,
                tag="fusion",
            )
        ]
        for block_index in range(self.n_layers - 1):
            sites.append(
                ActivationSite(
                    f"block_{block_index}",
                    self.layers[block_index + 1],
                    hook="pre",
                    block_index=block_index,
                    tag=self._site_tag(block_index),
                )
            )
        sites.append(
            ActivationSite(
                "target",
                self.layers[-1],
                hook="post",
                block_index=self.n_layers - 1,
                tag=self._site_tag(self.n_layers - 1),
                is_target=True,
            )
        )
        return sites

    def _truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(
            ids[:max_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        processor_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            output = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
                **processor_kwargs,
            )
        except TypeError as exc:
            raise RuntimeError(
                "the installed processor does not support the no-thinking multimodal "
                "chat-template API required by jlens; upgrade Transformers"
            ) from exc
        return _as_mapping(output)

    def _processor_kwargs(self, sample: MultimodalSample) -> dict[str, Any]:
        return {}

    def _render_and_mask(
        self,
        sample: MultimodalSample,
        *,
        include_answer: bool,
        max_seq_len: int,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, MultimodalSample]:
        user_text = self._truncate(sample.user_text, 128)
        assistant_text = (
            self._truncate(sample.assistant_text or "", 32) if include_answer else None
        )
        if include_answer and not assistant_text:
            raise ValueError(
                "teacher-forced fitting requires a non-empty assistant answer"
            )
        rendered_sample = replace(
            sample, user_text=user_text, assistant_text=assistant_text
        )
        kwargs = self._processor_kwargs(rendered_sample)
        prompt_messages = [
            {"role": "user", "content": _content(sample.image, user_text)}
        ]
        empty_prompt_messages = [
            {"role": "user", "content": _content(sample.image, "")}
        ]
        prompt = self._template(
            prompt_messages, add_generation_prompt=True, processor_kwargs=kwargs
        )
        empty_prompt = self._template(
            empty_prompt_messages,
            add_generation_prompt=True,
            processor_kwargs=kwargs,
        )
        prompt_ids = prompt["input_ids"][0].cpu()
        empty_prompt_ids = empty_prompt["input_ids"][0].cpu()
        prompt_start, prompt_end = _changed_span(
            prompt_ids, empty_prompt_ids, name="user"
        )

        if include_answer:
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": assistant_text},
            ]
            empty_answer_messages = [
                *prompt_messages,
                {"role": "assistant", "content": ""},
            ]
            full = self._template(
                full_messages, add_generation_prompt=False, processor_kwargs=kwargs
            )
            empty_answer = self._template(
                empty_answer_messages,
                add_generation_prompt=False,
                processor_kwargs=kwargs,
            )
            full_ids = full["input_ids"][0].cpu()
            if not torch.equal(prompt_ids, full_ids[: len(prompt_ids)]):
                raise ValueError(
                    "prompt-only chat template is not a prefix of the full conversation"
                )
            answer_start, answer_end = _changed_span(
                full_ids,
                empty_answer["input_ids"][0].cpu(),
                name="assistant",
            )
            if answer_start < len(prompt_ids):
                raise ValueError(
                    "assistant content begins before the generation prompt ends"
                )
        else:
            full = prompt
            full_ids = prompt_ids
            answer_start = answer_end = 0

        if len(full_ids) > max_seq_len:
            raise ValueError(
                f"processed sequence has {len(full_ids)} tokens, limit is {max_seq_len}"
            )
        prompt_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        prompt_mask[prompt_start:prompt_end] = True
        answer_token_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        if include_answer:
            if answer_start == 0:
                raise ValueError("assistant content has no preceding causal position")
            answer_token_mask[answer_start:answer_end] = True
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        if special_ids:
            is_special = torch.tensor(
                [int(token) in special_ids for token in full_ids], dtype=torch.bool
            )
            prompt_mask &= ~is_special
            answer_token_mask &= ~is_special
        answer_target_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        answer_positions = answer_token_mask.nonzero(as_tuple=True)[0]
        if include_answer and len(answer_positions) == 0:
            raise ValueError("assistant span contains no non-control content tokens")
        if len(answer_positions):
            answer_target_mask[answer_positions - 1] = True
        return full, prompt_mask, answer_target_mask, rendered_sample

    def _mm_token_type_ids(self, inputs: dict[str, Any]) -> torch.Tensor:
        value = inputs.get("mm_token_type_ids", inputs.get("token_type_ids"))
        if value is None:
            raise ValueError("processor did not return mm_token_type_ids")
        return value

    @staticmethod
    def _pooler_output(output: Any) -> Any:
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, dict):
            return output["pooler_output"]
        return output[1]

    @staticmethod
    def _deepstack_features(output: Any) -> list[torch.Tensor]:
        if hasattr(output, "deepstack_features"):
            return list(output.deepstack_features)
        if isinstance(output, dict):
            return list(output["deepstack_features"])
        raise ValueError("Qwen vision output has no DeepStack features")

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        device = self.lm_head.weight.device
        dtype = self.lm_head.weight.dtype
        logits = self.lm_head(self.final_norm(residual.to(device=device, dtype=dtype)))
        if self._softcap is not None:
            logits = self._softcap * torch.tanh(logits / self._softcap)
        return logits

    def _logits_from_normalized(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(
            hidden_states.to(
                device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype
            )
        )
        if self._softcap is not None:
            logits = self._softcap * torch.tanh(logits / self._softcap)
        return logits

    @torch.no_grad()
    def native_forward(self, prepared: PreparedMultimodalExample) -> Any:
        if prepared.native_inputs is None:
            raise ValueError("prepared example does not retain native model inputs")
        return self.hf_model(
            **prepared.native_inputs, use_cache=False, return_dict=True
        )


class Qwen3VLLensAdapter(_HfMultimodalAdapterBase):
    """Adapter for ``Qwen/Qwen3-VL-8B-Instruct`` and compatible checkpoints."""

    def __init__(self, hf_model: nn.Module, processor: Any) -> None:
        if _model_type(hf_model) != "qwen3_vl":
            raise ValueError(
                f"Qwen3VLLensAdapter requires model_type='qwen3_vl', got {_model_type(hf_model)!r}"
            )
        super().__init__(hf_model, processor)
        if not hasattr(self.outer_model, "visual"):
            raise ValueError("Qwen3-VL model has no visual tower")
        self.activation_sites = self._make_sites()

    def _site_tag(self, block_index: int | None) -> str:
        if block_index is not None and block_index < 3:
            return "deepstack"
        return "decoder"

    def _processor_kwargs(self, sample: MultimodalSample) -> dict[str, Any]:
        config_name = str(sample.metadata.get("config", "")).lower()
        target_tokens = 512 if config_name in {"docvqa", "chartqa"} else 256
        image_processor = self.processor.image_processor
        patch_size = int(getattr(image_processor, "patch_size", 16))
        merge_size = int(getattr(image_processor, "merge_size", 2))
        pixels_per_token = patch_size**2 * merge_size**2
        return {
            "min_pixels": 64 * pixels_per_token,
            "max_pixels": target_tokens * pixels_per_token,
        }

    def prepare(
        self,
        sample: MultimodalSample,
        *,
        include_answer: bool = True,
        max_seq_len: int = 768,
    ) -> PreparedMultimodalExample:
        raw, prompt_mask, answer_mask, rendered = self._render_and_mask(
            sample, include_answer=include_answer, max_seq_len=max_seq_len
        )
        inputs = _move_tensors(raw, self.input_device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        mm_types = self._mm_token_type_ids(inputs)
        image_mask = mm_types == 1
        if int(image_mask.sum()) == 0:
            raise ValueError("processor produced no image-token positions")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("Qwen processor omitted pixel_values/image_grid_thw")

        with torch.no_grad():
            inputs_embeds = self.language_model.embed_tokens(input_ids)
            vision_output = self.outer_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            pooled = self._pooler_output(vision_output)
            image_features = (
                torch.cat(pooled, dim=0)
                if isinstance(pooled, (list, tuple))
                else pooled
            )
            image_features = image_features.reshape(-1, self.d_model).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[expanded_mask].numel() != image_features.numel():
                raise ValueError(
                    "Qwen image features do not match image placeholder count"
                )
            inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_features)
            deepstack = [
                feature.reshape(-1, self.d_model).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                for feature in self._deepstack_features(vision_output)
            ]
            position_ids = self.outer_model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=mm_types,
            )
        if position_ids is None:
            raise ValueError("Qwen failed to construct multimodal RoPE positions")
        image_source_mask = image_mask[0].bool().cpu()
        prompt_mask &= ~image_source_mask
        prepared = PreparedMultimodalExample(
            sample_id=sample.sample_id,
            input_ids=input_ids.detach(),
            decoder_inputs={
                "inputs_embeds": inputs_embeds.detach(),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "visual_pos_masks": image_mask.bool(),
                "deepstack_visual_embeds": [feature.detach() for feature in deepstack],
            },
            image_source_mask=image_source_mask,
            prompt_source_mask=prompt_mask,
            answer_target_mask=answer_mask,
            metadata={
                **dict(rendered.metadata),
                "user_text": rendered.user_text,
                "assistant_text": rendered.assistant_text,
            },
            native_inputs=inputs,
        )
        if not prepared.prompt_source_mask.any():
            raise ValueError("could not locate user prompt content tokens")
        return prepared

    def expand_for_probes(
        self, prepared: PreparedMultimodalExample, batch_size: int
    ) -> dict[str, Any]:
        inputs = prepared.decoder_inputs
        position_ids = inputs["position_ids"]
        if position_ids.ndim == 3:
            position_ids = position_ids.expand(-1, batch_size, -1)
        else:
            position_ids = position_ids.expand(batch_size, -1)
        return {
            "inputs_embeds": inputs["inputs_embeds"].expand(batch_size, -1, -1),
            "attention_mask": inputs["attention_mask"].expand(batch_size, -1),
            "position_ids": position_ids,
            "visual_pos_masks": inputs["visual_pos_masks"].expand(batch_size, -1),
            "deepstack_visual_embeds": [
                feature.repeat(batch_size, 1)
                for feature in inputs["deepstack_visual_embeds"]
            ],
        }

    def forward_decoder(self, decoder_inputs: dict[str, Any]) -> Any:
        return self.language_model(
            input_ids=None,
            use_cache=False,
            return_dict=True,
            **decoder_inputs,
        )


class Gemma4UnifiedLensAdapter(_HfMultimodalAdapterBase):
    """Adapter specifically for encoder-free Gemma 4 12B Unified."""

    def __init__(self, hf_model: nn.Module, processor: Any) -> None:
        if _model_type(hf_model) != "gemma4_unified":
            raise ValueError(
                "Gemma4UnifiedLensAdapter only supports the encoder-free "
                f"model_type='gemma4_unified', got {_model_type(hf_model)!r}"
            )
        super().__init__(hf_model, processor)
        if hasattr(self.outer_model, "vision_tower"):
            raise ValueError("Gemma 4 Unified must not expose a separate vision tower")
        if not hasattr(self.outer_model, "embed_vision"):
            raise ValueError("Gemma 4 Unified model has no direct vision embedder")
        if getattr(_text_config(hf_model), "hidden_size_per_layer_input", 0):
            raise ValueError(
                "the selected Gemma 4 Unified adapter does not support PLE"
            )
        self._modeling_module = importlib.import_module(
            type(self.outer_model).__module__
        )
        self.activation_sites = self._make_sites()

    def _site_tag(self, block_index: int | None) -> str:
        if block_index is None:
            return "fusion"
        layer_types = list(getattr(_text_config(self.hf_model), "layer_types", []))
        if block_index < len(layer_types):
            return "global" if "full" in layer_types[block_index] else "sliding"
        return "global" if (block_index + 1) % 6 == 0 else "sliding"

    def _processor_kwargs(self, sample: MultimodalSample) -> dict[str, Any]:
        config_name = str(sample.metadata.get("config", "")).lower()
        return {"max_soft_tokens": 560 if config_name in {"docvqa", "chartqa"} else 280}

    def prepare(
        self,
        sample: MultimodalSample,
        *,
        include_answer: bool = True,
        max_seq_len: int = 768,
    ) -> PreparedMultimodalExample:
        raw, prompt_mask, answer_mask, rendered = self._render_and_mask(
            sample, include_answer=include_answer, max_seq_len=max_seq_len
        )
        inputs = _move_tensors(raw, self.input_device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        mm_types = self._mm_token_type_ids(inputs)
        image_mask = mm_types == 1
        if int(image_mask.sum()) == 0:
            raise ValueError("processor produced no image-token positions")
        pixel_values = inputs.get("pixel_values")
        image_position_ids = inputs.get("image_position_ids")
        if pixel_values is None or image_position_ids is None:
            raise ValueError("Gemma processor omitted pixel_values/image_position_ids")

        with torch.no_grad():
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask] = _text_config(self.hf_model).pad_token_id
            inputs_embeds = self.language_model.embed_tokens(llm_input_ids)
            vision_output = self.outer_model.get_image_features(
                pixel_values, image_position_ids, return_dict=True
            )
            image_features = (
                self._pooler_output(vision_output)
                .reshape(-1, self.d_model)
                .to(inputs_embeds.device, inputs_embeds.dtype)
            )
            expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[expanded_mask].numel() != image_features.numel():
                raise ValueError(
                    "Gemma image features do not match image placeholder count"
                )
            inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_features)
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        image_source_mask = image_mask[0].bool().cpu()
        prompt_mask &= ~image_source_mask
        prepared = PreparedMultimodalExample(
            sample_id=sample.sample_id,
            input_ids=input_ids.detach(),
            decoder_inputs={
                "inputs_embeds": inputs_embeds.detach(),
                "attention_mask_2d": attention_mask,
                "position_ids": position_ids,
                "mm_token_type_ids": mm_types,
            },
            image_source_mask=image_source_mask,
            prompt_source_mask=prompt_mask,
            answer_target_mask=answer_mask,
            metadata={
                **dict(rendered.metadata),
                "user_text": rendered.user_text,
                "assistant_text": rendered.assistant_text,
            },
            native_inputs=inputs,
        )
        if not prepared.prompt_source_mask.any():
            raise ValueError("could not locate user prompt content tokens")
        return prepared

    def _attention_masks(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> Any:
        kwargs = {
            "config": _text_config(self.hf_model),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        if (
            getattr(_text_config(self.hf_model), "use_bidirectional_attention", None)
            == "vision"
        ):
            block_fn = self._modeling_module.get_block_sequence_ids_for_mask
            try:
                block_ids = block_fn(mm_token_type_ids, device=inputs_embeds.device)
            except TypeError:  # older compatible signature
                block_ids = block_fn(mm_token_type_ids)
            kwargs["block_sequence_ids"] = block_ids
        create_masks = self._modeling_module.create_masks_for_generate
        return create_masks(**kwargs)

    def expand_for_probes(
        self, prepared: PreparedMultimodalExample, batch_size: int
    ) -> dict[str, Any]:
        inputs = prepared.decoder_inputs
        embeds = inputs["inputs_embeds"].expand(batch_size, -1, -1)
        attention = inputs["attention_mask_2d"]
        if attention is not None:
            attention = attention.expand(batch_size, -1)
        position_ids = inputs["position_ids"].expand(batch_size, -1)
        mm_types = inputs["mm_token_type_ids"].expand(batch_size, -1)
        return {
            "inputs_embeds": embeds,
            "attention_mask": self._attention_masks(
                embeds, attention, position_ids, mm_types
            ),
            "position_ids": position_ids,
        }

    def forward_decoder(self, decoder_inputs: dict[str, Any]) -> Any:
        return self.language_model(
            input_ids=None,
            use_cache=False,
            return_dict=True,
            **decoder_inputs,
        )


def from_hf_multimodal(hf_model: nn.Module, processor: Any):
    """Select the strict architecture-specific VLM adapter."""

    model_type = _model_type(hf_model)
    if model_type == "qwen3_vl":
        return Qwen3VLLensAdapter(hf_model, processor)
    if model_type == "gemma4_unified":
        return Gemma4UnifiedLensAdapter(hf_model, processor)
    raise ValueError(
        f"unsupported multimodal model_type {model_type!r}; supported: "
        "'qwen3_vl', 'gemma4_unified'"
    )


@torch.no_grad()
def validate_adapter_parity(
    adapter: _HfMultimodalAdapterBase,
    samples: list[MultimodalSample],
    *,
    max_seq_len: int = 768,
    rtol: float = 2e-2,
    atol: float = 2e-2,
    min_top1_match: float = 0.995,
) -> list[dict[str, float | str]]:
    """Require prepared-decoder logits to match the native VLM forward."""

    reports: list[dict[str, float | str]] = []
    for sample in samples:
        prepared = adapter.prepare(sample, include_answer=True, max_seq_len=max_seq_len)
        native = adapter.native_forward(prepared).logits.float()
        decoder_output = adapter.forward_decoder(adapter.expand_for_probes(prepared, 1))
        prepared_logits = adapter._logits_from_normalized(
            decoder_output.last_hidden_state
        ).float()
        if native.shape != prepared_logits.shape:
            raise RuntimeError(
                f"adapter parity shape mismatch for {sample.sample_id}: "
                f"native={native.shape}, prepared={prepared_logits.shape}"
            )
        top1_match = float(
            (native.argmax(dim=-1) == prepared_logits.argmax(dim=-1)).float().mean()
        )
        max_abs = float((native - prepared_logits).abs().max())
        if not torch.allclose(native, prepared_logits, rtol=rtol, atol=atol):
            raise RuntimeError(
                f"adapter parity failed for {sample.sample_id}: max_abs={max_abs:.4g}"
            )
        if top1_match < min_top1_match:
            raise RuntimeError(
                f"adapter top-1 parity failed for {sample.sample_id}: "
                f"{top1_match:.3%} < {min_top1_match:.3%}"
            )
        reports.append(
            {
                "sample_id": sample.sample_id,
                "max_abs_error": max_abs,
                "top1_match": top1_match,
            }
        )
    return reports
