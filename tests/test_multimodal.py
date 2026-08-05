# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import torch

from jlens.multimodal import (
    EstimatorConfig,
    MultimodalActivationRecorder,
    MultimodalJacobianLensBundle,
    MultimodalSample,
    _probe_vjp,
    evaluate_multimodal_sample,
    fit_multimodal,
)

from .tiny_multimodal import TinyMultimodalAdapter


def _sample(index: int = 0) -> MultimodalSample:
    return MultimodalSample(
        sample_id=f"sample-{index}",
        image=object(),
        user_text="What is in the image?",
        assistant_text="A test object.",
        metadata={"config": "vqav2"},
    )


def test_pre_hook_captures_external_deepstack_addition():
    adapter = TinyMultimodalAdapter()
    prepared = adapter.prepare(_sample())
    inputs = adapter.expand_for_probes(prepared, 1)
    with MultimodalActivationRecorder(
        adapter.activation_sites, at=["embed", "block_0", "target"]
    ) as recorder:
        adapter.forward_decoder(inputs)
    embed = recorder.activations["embed"]
    raw_block_output = adapter.layers[0](embed)
    expected = raw_block_output + adapter.deepstack
    torch.testing.assert_close(recorder.activations["block_0"], expected)
    assert not torch.equal(recorder.activations["block_0"], raw_block_output)


def test_randomized_fit_converges_to_exact_and_resumes(tmp_path):
    adapter = TinyMultimodalAdapter(d_model=4)
    sample = _sample()
    prepared = adapter.prepare(sample)
    exact_rows = _probe_vjp(
        adapter,
        prepared,
        torch.eye(adapter.d_model),
        modes=("image_to_answer", "prompt_to_answer"),
        target_reduction="mean",
    )
    output = tmp_path / "fit"
    fitted = fit_multimodal(
        adapter,
        [sample],
        output,
        config=EstimatorConfig(
            probes_per_example=2048,
            probe_microbatch=128,
            checkpoint_every=1,
            materialize_chunk_size=256,
        ),
        materialize_device="cpu",
    )
    for mode in ("image_to_answer", "prompt_to_answer"):
        for site in fitted.source_sites:
            relative_error = (
                fitted.jacobians[mode][site] - exact_rows[mode][site]
            ).norm() / exact_rows[mode][site].norm()
            assert relative_error < 0.12

    resumed = fit_multimodal(
        adapter,
        [sample],
        output,
        config=EstimatorConfig(
            probes_per_example=2048,
            probe_microbatch=128,
            checkpoint_every=1,
            materialize_chunk_size=256,
        ),
        materialize_device="cpu",
    )
    assert resumed.n_examples == fitted.n_examples == 1
    assert resumed.n_probes == fitted.n_probes == 2048
    for mode in fitted.modes:
        for site in fitted.source_sites:
            torch.testing.assert_close(
                resumed.jacobians[mode][site], fitted.jacobians[mode][site]
            )


def test_bundle_selective_load_readout_and_evaluation(tmp_path):
    adapter = TinyMultimodalAdapter()
    sites = [site.name for site in adapter.activation_sites if not site.is_target]
    matrices = {
        mode: {site: torch.eye(adapter.d_model) for site in sites}
        for mode in ("image_to_answer", "prompt_to_answer")
    }
    bundle = MultimodalJacobianLensBundle(
        matrices,
        d_model=adapter.d_model,
        n_examples=3,
        n_probes=12,
        site_metadata={site: {"tag": "test"} for site in sites},
        metadata={"model_id": adapter.model_id},
    )
    path = tmp_path / "lens"
    bundle.save(path, sites_per_shard=2)
    loaded = MultimodalJacobianLensBundle.load(
        path, modes=["image_to_answer"], sites=["embed", "block_1"]
    )
    assert loaded.modes == ["image_to_answer"]
    assert loaded.source_sites == ["embed", "block_1"]
    logits, input_ids = loaded.readout(
        adapter, _sample(), mode="image_to_answer", sites=["embed"]
    )
    assert logits["embed"].shape == (1, 11)
    assert input_ids.shape == (1, 8)

    rows = evaluate_multimodal_sample(
        adapter, bundle, _sample(), sites=["embed"], top_k=3
    )
    assert len(rows) == 2
    for row in rows:
        assert 0 <= row["top_k_overlap"] <= 1
        assert torch.isfinite(torch.tensor(row["kl_native_to_lens"]))


def test_unstable_initial_sketch_collects_adaptive_probes(tmp_path):
    adapter = TinyMultimodalAdapter()
    bundle = fit_multimodal(
        adapter,
        [_sample()],
        tmp_path / "adaptive",
        config=EstimatorConfig(
            probes_per_example=2,
            max_probes_per_example=4,
            probe_microbatch=2,
            checkpoint_every=1,
            stability_vectors=4,
            stability_threshold=1.0,
        ),
        materialize_device="cpu",
    )
    assert bundle.n_probes == 4
    assert bundle.metadata["adaptive_probe_count"] == 2


def test_report_cli_writes_csv_and_architecture_svg(tmp_path):
    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(
        json.dumps(
            {
                "source_sites": ["embed", "block_0"],
                "site_metadata": {
                    "embed": {"tag": "fusion"},
                    "block_0": {"tag": "deepstack"},
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text(
        "\n".join(
            json.dumps(
                {
                    "condition": "original",
                    "mode": "image_to_answer",
                    "site": site,
                    "logit_cosine": value,
                }
            )
            for site, value in (("embed", 0.2), ("block_0", 0.5))
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "scripts/report_multimodal.py",
            "--eval",
            str(evaluation),
            "--lens",
            str(lens),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    assert (output / "summary.csv").exists()
    svg = (output / "layer_plot.svg").read_text(encoding="utf-8")
    assert "deepstack" in svg and "image_to_answer" in svg
