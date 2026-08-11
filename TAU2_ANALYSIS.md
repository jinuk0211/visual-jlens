# Analyze tau2 trajectories with Jacobian Lens

The `scripts/analyze_tau2.py` command reads both `results.json` and the verbose
agent-call logs saved below `artifacts/task_*/sim_*/llm_debug/`. The default
event-aligned analysis:

1. conservatively localizes the first verifiable error (`t_fail`);
2. retains every earlier agent call plus one post-error call;
3. includes successful runs of the same task for matched comparison;
4. reconstructs the exact pre-response request and teacher-forces the logged
   assistant response; and
5. reads out observation, decision, tool, argument, and post-result update
   boundaries.

Unknown tools, locally checkable schema violations, and recorded tool execution
errors are treated as verified first-error evidence. Tau2 reviewer turn labels
are marked `reviewed`, not silently promoted to verified evidence. End-state
failures that cannot be localized from the saved artifacts remain
`unlocalized`; all calls are retained, but they should be excluded from strict
event-aligned first-error statistics until manually audited.

## Vast.ai setup

Stop the vLLM server before the GPU analysis so Qwen can be loaded by PyTorch.
The existing 32 GB disk is too tight for another environment, the lens, and
reports; expand the instance disk to at least 50 GB first. Copy this modified
repository to `/workspace/jacobian-lens`, then run:

```bash
set -euo pipefail

export PATH="/root/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"

cd /workspace/jacobian-lens
uv sync --frozen

RUN_DIR="/workspace/tau2-bench/data/simulations/qwen3-8b-airline-10tasks-nonthinking-seed300"
OUT_DIR="/workspace/qwen3-8b-jlens-analysis"
MODEL_REVISION="$(tr -d '\r\n' < /workspace/qwen3-8b-revision.txt)"

uv run python scripts/analyze_tau2.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --model-revision "$MODEL_REVISION" \
  --inspect-only
```

`--inspect-only` does not load Qwen or the lens. Check that `selected_calls` is
non-zero and that the cases do not say `missing_agent_logs` in
`$OUT_DIR/manifest.json`.

First run one case without HTML as a GPU and tokenization check:

```bash
uv run python scripts/analyze_tau2.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR/pilot" \
  --model-revision "$MODEL_REVISION" \
  --limit 1 \
  --no-html \
  --layer-stride 4 \
  --max-seq-len 32768
```

Then run the event-aligned analysis. Task-matched successes are included by
default, so `--include-successes` is needed only when successes from unrelated
tasks should also be analyzed:

```bash
uv run python scripts/analyze_tau2.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR/full" \
  --model-revision "$MODEL_REVISION" \
  --call-selection event \
  --event-before all \
  --event-after 1 \
  --layer-stride 4 \
  --top-k 10 \
  --last-n-tokens 0 \
  --max-seq-len 32768
```

Use `--event-before N` for a bounded pilot. `--no-pair-successes` disables the
matched-success inclusion. `--call-selection last` reproduces the old pilot
behavior, while `--call-selection all` ignores event localization and analyzes
every selected call.

For failures that require conservative manual audit, pass a JSON object keyed
by simulation id:

```json
{
  "simulation-uuid": {
    "call_index": 3,
    "kind": "policy_violation",
    "reason": "First state-changing call violates the cancellation policy."
  }
}
```

Then add `--first-error-labels /path/to/first_error_labels.json`. A label may
identify a call with `call_index`, `call_id`, or `turn_index`.

Do not add `--enable-thinking` for the run above: tau2 generated it with
`enable_thinking=false`.

Serve the interactive pages over HTTP rather than opening `index.html`
directly, because the page loads binary sidecar files:

```bash
cd "$OUT_DIR/full"
python -m http.server 8080 --bind 0.0.0.0
```

Expose port 8080 in Vast.ai, then open the URL for a per-call `index.html` path
listed in `analysis_report.json`.

## Outputs and interpretation

- `manifest.json` inventories reward details, expected tools, emitted tools,
  candidate tools, selected verbose logs, first-error provenance/confidence,
  and failure/success call alignments.
- `semantic_boundary_readouts.csv` contains open-vocabulary J-lens top tokens
  at observation, decision, tool, argument, and update boundaries for every
  selected call and layer.
- `generated_span_scores.csv` contains layerwise likelihoods and ranks for the
  actual tool-name and scalar argument-value tokens replayed from the log.
- `tool_scores.csv` contains one row per candidate and layer. A higher
  `mean_logprob` and lower `mean_rank` indicate stronger support. Use
  `sum_logprob` when comparing complete tool-name sequence likelihoods and
  remember that it penalizes longer names.
- `analysis_report.json` records per-call success, errors, and visualization
  paths.
- Each per-call directory contains an interactive Jacobian-lens slice including
  the logged assistant response. `--last-n-tokens 0` renders every position,
  and `--top-k K` stores the top K lens tokens for every rendered
  position/layer cell. These options only change the HTML grid, not
  semantic-boundary CSV computation.

Useful diagnostic patterns:

- An expected tool that is strong in middle layers but loses near the final
  layer suggests a late transformation or decision problem.
- An expected tool that is weak at every layer suggests that the relevant
  task or policy information was not represented strongly in this context.
- An expected tool that scores above the sampled tool at the model-final row
  points toward stochastic decoding or serving-time processing as a candidate
  explanation. The original run used non-greedy sampling, so this distinction
  matters.
- If expected and actual tool names match but tau2 still fails, inspect
  `reward_info` and arguments. Tool-name lens scores cannot diagnose a wrong
  argument, wrong action order, policy violation, or communication-evaluator
  failure by themselves.

These measurements are diagnostic correlations, not a causal proof. Use the
emitted alignment rows for matched success/failure comparisons and an
intervention such as activation patching before claiming a causal mechanism.
Only `verified` or manually audited first-error labels belong in the strict
event-aligned analysis. More repetitions per task are much more reliable than
a single seed.
