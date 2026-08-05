# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Architecture adapter tests using HF-shaped mocks (no model downloads)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from jlens.multimodal import MultimodalActivationRecorder, MultimodalSample
from jlens.multimodal_hf import (
    Gemma4UnifiedLensAdapter,
    Qwen3VLLensAdapter,
    from_hf_multimodal,
    validate_adapter_parity,
)


def get_block_sequence_ids_for_mask(ids, device=None):
    return torch.where(ids.to(device) == 1, 0, -1)


def create_masks_for_generate(**kwargs):
    mask = kwargs["attention_mask"]
    return {"full_attention": mask, "sliding_attention": mask}


class _Tokenizer:
    all_special_ids = [90, 91, 92, 99]

    def encode(self, text, add_special_tokens=False):
        return list(range(min(4, max(1, len(text.split()))))) if text else []

    def decode(self, ids, **kwargs):
        return " ".join(str(item) for item in ids)


class _Processor:
    def __init__(self, kind):
        self.kind = kind
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace(patch_size=1, merge_size=1)

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        **kwargs,
    ):
        user_text = messages[0]["content"][1]["text"]
        has_user = bool(user_text)
        has_assistant = len(messages) == 2
        assistant = messages[1]["content"] if has_assistant else None
        ids = [90, 99, 99]
        if has_user:
            ids += [10, 11]
        ids += [91]
        if has_assistant and assistant:
            ids += [20, 21]
        if has_assistant:
            ids += [92]
        result = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, *([0] * (len(ids) - 3))]]),
            "pixel_values": torch.ones(2, 3),
        }
        if self.kind == "qwen":
            result["image_grid_thw"] = torch.tensor([[1, 1, 2]])
        else:
            result["image_position_ids"] = torch.tensor([[[0, 0], [0, 1]]])
        return result


class _Block(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states + 0.01


class _LanguageModel(nn.Module):
    def __init__(self, d_model=4, n_layers=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, d_model)
        self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if deepstack_visual_embeds is not None and index < len(
                deepstack_visual_embeds
            ):
                hidden = hidden.clone()
                hidden[visual_pos_masks] += deepstack_visual_embeds[index]
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _TextConfig:
    num_hidden_layers = 4
    hidden_size = 4
    pad_token_id = 0
    final_logit_softcapping = None
    hidden_size_per_layer_input = 0
    use_bidirectional_attention = "vision"
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type
        self._name_or_path = f"mock-{model_type}"
        self._commit_hash = "abc"
        self.text_config = _TextConfig()
        self.image_token_id = 99

    def get_text_config(self):
        return self.text_config


class _QwenOuter(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.visual = nn.Linear(1, 1)
        self.config = SimpleNamespace(image_token_id=99)

    def get_image_features(self, pixel_values, image_grid_thw, return_dict=True):
        base = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
        return SimpleNamespace(
            pooler_output=[base],
            deepstack_features=[base + index for index in range(3)],
        )

    def compute_3d_position_ids(self, input_ids, **kwargs):
        seq_len = input_ids.shape[1]
        return torch.arange(seq_len).view(1, 1, -1).expand(3, 1, -1)


class _GemmaOuter(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.embed_vision = nn.Linear(1, 1)

    def get_image_features(self, pixel_values, image_position_ids, return_dict=True):
        features = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
        return SimpleNamespace(pooler_output=features)


class _HFModel(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.config = _Config(model_type)
        self.model = _QwenOuter() if model_type == "qwen3_vl" else _GemmaOuter()
        self.lm_head = nn.Linear(4, 13, bias=False)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        mm_token_type_ids,
        image_grid_thw=None,
        image_position_ids=None,
        **kwargs,
    ):
        image_mask = mm_token_type_ids == 1
        llm_ids = input_ids.clone()
        llm_ids[image_mask] = 0 if self.config.model_type == "gemma4_unified" else 99
        embeds = self.model.language_model.embed_tokens(llm_ids)
        if self.config.model_type == "qwen3_vl":
            vision = self.model.get_image_features(pixel_values, image_grid_thw)
            features = torch.cat(vision.pooler_output)
            embeds = embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(embeds), features
            )
            output = self.model.language_model(
                inputs_embeds=embeds,
                visual_pos_masks=image_mask,
                deepstack_visual_embeds=vision.deepstack_features,
            )
        else:
            features = self.model.get_image_features(
                pixel_values, image_position_ids
            ).pooler_output
            embeds = embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(embeds), features
            )
            output = self.model.language_model(inputs_embeds=embeds)
        return SimpleNamespace(logits=self.lm_head(output.last_hidden_state))


def _sample():
    return MultimodalSample(
        "one", object(), "what is this", "an object", {"config": "vqav2"}
    )


def test_qwen_adapter_masks_expansion_and_post_deepstack_capture():
    adapter = Qwen3VLLensAdapter(_HFModel("qwen3_vl"), _Processor("qwen"))
    prepared = adapter.prepare(_sample())
    assert prepared.image_source_mask.nonzero().flatten().tolist() == [1, 2]
    assert prepared.prompt_source_mask.nonzero().flatten().tolist() == [3, 4]
    assert prepared.answer_target_mask.nonzero().flatten().tolist() == [5, 6]
    expanded = adapter.expand_for_probes(prepared, 2)
    assert expanded["inputs_embeds"].shape == (2, 9, 4)
    assert expanded["position_ids"].shape == (3, 2, 9)
    assert expanded["deepstack_visual_embeds"][0].shape == (4, 4)
    with MultimodalActivationRecorder(
        adapter.activation_sites, at=["embed", "block_0"]
    ) as recorder:
        adapter.forward_decoder(expanded)
    expected = adapter.layers[0](recorder.activations["embed"]).clone()
    expected[expanded["visual_pos_masks"]] += expanded["deepstack_visual_embeds"][0]
    torch.testing.assert_close(recorder.activations["block_0"], expected)


def test_gemma_unified_adapter_uses_direct_embedder_and_layer_tags():
    adapter = Gemma4UnifiedLensAdapter(_HFModel("gemma4_unified"), _Processor("gemma"))
    prepared = adapter.prepare(_sample())
    expanded = adapter.expand_for_probes(prepared, 2)
    assert set(expanded["attention_mask"]) == {
        "full_attention",
        "sliding_attention",
    }
    assert expanded["inputs_embeds"].shape == (2, 9, 4)
    tags = {site.block_index: site.tag for site in adapter.activation_sites}
    assert tags[0] == "sliding"
    assert tags[2] == "global"
    output = adapter.forward_decoder(expanded)
    assert output.last_hidden_state.shape == (2, 9, 4)


def test_from_hf_multimodal_is_strict():
    assert isinstance(
        from_hf_multimodal(_HFModel("qwen3_vl"), _Processor("qwen")),
        Qwen3VLLensAdapter,
    )
    bad = _HFModel("not-a-vlm")
    try:
        from_hf_multimodal(bad, _Processor("qwen"))
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported model should fail")


def test_adapter_parity_gate():
    adapters = [
        Qwen3VLLensAdapter(_HFModel("qwen3_vl"), _Processor("qwen")),
        Gemma4UnifiedLensAdapter(_HFModel("gemma4_unified"), _Processor("gemma")),
    ]
    for adapter in adapters:
        reports = validate_adapter_parity(adapter, [_sample()])
        assert reports[0]["top1_match"] == 1.0
