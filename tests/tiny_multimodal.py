# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Tiny architecture-independent multimodal adapter for CPU tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from jlens.multimodal import (
    ActivationSite,
    MultimodalSample,
    PreparedMultimodalExample,
)


class _MixingBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.local = nn.Linear(d_model, d_model, bias=False)
        self.context = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.local.weight.mul_(0.08)
            self.context.weight.mul_(0.08)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean = hidden_states.mean(dim=1, keepdim=True)
        return hidden_states + self.local(hidden_states) + self.context(mean)


class TinyMultimodalAdapter(nn.Module):
    def __init__(self, *, d_model: int = 4, n_layers: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.model_id = "tiny-vlm"
        self.revision = "test"
        self.d_model = d_model
        self.n_layers = n_layers
        self.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
        self.layers = nn.ModuleList([_MixingBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, 11, bias=False)
        self.register_buffer("deepstack", torch.linspace(-0.1, 0.1, d_model))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.activation_sites = [
            ActivationSite("embed", self.layers[0], hook="pre", tag="fusion"),
            *[
                ActivationSite(
                    f"block_{index}",
                    self.layers[index + 1],
                    hook="pre",
                    block_index=index,
                    tag="deepstack" if index == 0 else "decoder",
                )
                for index in range(n_layers - 1)
            ],
            ActivationSite(
                "target",
                self.layers[-1],
                hook="post",
                block_index=n_layers - 1,
                is_target=True,
            ),
        ]

    def prepare(
        self,
        sample: MultimodalSample,
        *,
        include_answer: bool = True,
        max_seq_len: int = 768,
    ) -> PreparedMultimodalExample:
        if include_answer and not sample.assistant_text:
            raise ValueError("missing answer")
        seq_len = 8
        if max_seq_len < seq_len:
            raise ValueError("too long")
        generator = torch.Generator().manual_seed(
            sum(sample.sample_id.encode()) + len(sample.user_text)
        )
        embeds = torch.randn(1, seq_len, self.d_model, generator=generator)
        image = torch.zeros(seq_len, dtype=torch.bool)
        image[1:3] = True
        prompt = torch.zeros(seq_len, dtype=torch.bool)
        prompt[3:5] = True
        answer = torch.zeros(seq_len, dtype=torch.bool)
        if include_answer:
            answer[5:7] = True
        return PreparedMultimodalExample(
            sample_id=sample.sample_id,
            input_ids=torch.arange(seq_len).unsqueeze(0),
            decoder_inputs={"inputs_embeds": embeds},
            image_source_mask=image,
            prompt_source_mask=prompt,
            answer_target_mask=answer,
        )

    def expand_for_probes(
        self, prepared: PreparedMultimodalExample, batch_size: int
    ) -> dict[str, torch.Tensor]:
        return {
            "inputs_embeds": prepared.decoder_inputs["inputs_embeds"].expand(
                batch_size, -1, -1
            )
        }

    def forward_decoder(self, decoder_inputs):
        hidden = decoder_inputs["inputs_embeds"]
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index == 0:
                # Deliberately outside layer 0, just like Qwen DeepStack.
                hidden = hidden + self.deepstack
        return SimpleNamespace(last_hidden_state=self.norm(hidden))

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual))
