# jlens — Jacobian lens

> **Reference implementation.** Not maintained and not accepting contributions.

Companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and
position into the final-layer basis, then decodes it with the model's own
unembedding into a ranked list of vocabulary tokens.

The transport is the average input–output Jacobian over a text corpus:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[∂h_final / ∂h_l]
```

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the precise estimator
(cotangents summed over target positions, then averaged over source positions)
is documented in the [`jlens.fitting`](jlens/fitting.py) module docstring.

This repo fits the lens on open-weights decoder transformers, applies it, and
renders the interactive layer × position view shown below. Examples use Qwen;
other HuggingFace decoders adapt cleanly.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

*The ASCII-face example: selecting the `^` (nose) position shows the lens
reading out "nose" at mid layers, although the word never appears in the
prompt.*

## Install

```bash
pip install -e .
```

## Usage

### Apply

To apply a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

### Multimodal fit (Gemma 4 / Qwen3-VL)

The multimodal path fits two separate decoder-residual lenses:

- `image_to_answer`: fused image-token positions to assistant-answer positions
- `prompt_to_answer`: user-text positions to assistant-answer positions

It records the input to each decoder block rather than the preceding block's
raw output. This matters for Qwen3-VL because its early DeepStack additions
happen between decoder blocks. Gemma 4 12B Unified is handled by a separate,
encoder-free adapter that preserves its visual bidirectional attention masks.

First create the pinned 1,000-fit/200-eval The Cauldron manifest. The manifest
contains dataset provenance and image hashes, not image bytes:

```bash
python scripts/prepare_cauldron.py out/cauldron.jsonl --seed 0
```

Run a 32-example pilot, then the full fit. Both commands are resumable when
re-run with the same output directory:

```bash
python scripts/fit_multimodal.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --manifest out/cauldron.jsonl \
  --output-dir out/qwen3-vl-pilot --limit 32

python scripts/fit_multimodal.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --manifest out/cauldron.jsonl \
  --output-dir out/qwen3-vl
```

Use `google/gemma-4-12B-it` for the Gemma run. Install the `vlm` extra with
`pip install -e '.[vlm]'`; this selects a Transformers release with
`AutoModelForMultimodalLM` and Gemma 4 Unified support.
Fitting defaults to BF16 weights, four Rademacher probes per example, and an
automatically selected probe microbatch of 4, 2, or 1. Unlike the exact
text-only fitter, this costs one forward/backward per probe microbatch rather
than one backward per hidden dimension.

The result is written to `OUTPUT_DIR/lens/` as sharded FP16 safetensors plus a
JSON manifest. Held-out evaluation, including optional gray-image and
image-swap controls, is available with:

```bash
python scripts/evaluate_multimodal.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --lens out/qwen3-vl/lens \
  --manifest out/cauldron.jsonl \
  --output out/qwen3-vl-eval.jsonl --controls

python scripts/report_multimodal.py \
  --eval out/qwen3-vl-eval.jsonl \
  --lens out/qwen3-vl/lens \
  --output-dir out/qwen3-vl-report
```

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model, load (or fit) a lens, apply it at a few layers, and render a slice page
like the one above.

For replaying saved tau2-bench tool-agent failures, see
[`TAU2_ANALYSIS.md`](TAU2_ANALYSIS.md). The included CLI reconstructs the exact
logged request and response, aligns full histories to the first verifiable
error, pairs successful trajectories from the same task, reads out semantic
observation/decision/tool/argument/update boundaries, and scores actual tool
and argument tokens layer by layer.

Reading a slice page:

- Each cell shows every saved top-K token at that (position, layer), labeled
  `#1` through `#K` by full-vocabulary rank.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

## License and data

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
