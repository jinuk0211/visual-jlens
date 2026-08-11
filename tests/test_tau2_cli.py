import json
import sys

import torch

from jlens.tau2 import LoggedCall, Tau2Case
from scripts.analyze_tau2 import (
    SelectedCall,
    build_manifest,
    parse_args,
    score_generated_spans,
    score_semantic_boundaries,
)
from tests.test_tau2 import FakeTokenizer, sample_call


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_parse_args_accepts_all_token_top_k(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_tau2.py",
            "--run-dir",
            str(tmp_path),
            "--top-k",
            "10",
            "--last-n-tokens",
            "0",
        ],
    )

    args = parse_args()

    assert args.top_k == 10
    assert args.last_n_tokens == 0


def test_build_manifest_reports_missing_and_selected_logs(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "results.json",
        {
            "tasks": [
                {
                    "id": "1",
                    "evaluation_criteria": {
                        "actions": [{"name": "cancel_reservation"}]
                    },
                },
                {"id": "2", "evaluation_criteria": {"actions": []}},
            ],
            "simulations": [
                {
                    "id": "sim-1",
                    "task_id": "1",
                    "reward_info": {"reward": 0.0},
                    "messages": [],
                },
                {
                    "id": "sim-2",
                    "task_id": "2",
                    "reward_info": {"reward": 0.0},
                    "messages": [],
                },
            ],
        },
    )
    for call_id, tool_name in (("a", "lookup_booking"), ("b", "search_flights")):
        write_json(
            run_dir
            / "artifacts"
            / "task_1"
            / "sim_sim-1"
            / "llm_debug"
            / f"{call_id}.json",
            {
                "call_id": call_id,
                "call_name": "agent_response",
                "request": {"messages": [], "tools": []},
                "response": {"tool_calls": [{"name": tool_name}]},
            },
        )

    manifest, selected = build_manifest(run_dir, call_selection="last")

    assert {key: manifest["summary"][key] for key in (
        "saved_cases",
        "selected_cases",
        "selected_calls",
        "cases_without_agent_logs",
    )} == {
        "saved_cases": 2,
        "selected_cases": 2,
        "selected_calls": 1,
        "cases_without_agent_logs": 1,
    }
    assert manifest["cases"][0]["call_ids"] == ["b"]
    assert manifest["cases"][0]["candidate_tools"] == [
        "cancel_reservation",
        "search_flights",
    ]
    assert manifest["cases"][1]["status"] == "missing_agent_logs"
    assert len(selected) == 1


def test_event_manifest_localizes_failure_keeps_history_and_pairs_success(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "results.json",
        {
            "tasks": [{"id": "1", "evaluation_criteria": {"actions": []}}],
            "simulations": [
                {
                    "id": "failed",
                    "task_id": "1",
                    "reward_info": {"reward": 0.0},
                    "messages": [],
                },
                {
                    "id": "passed",
                    "task_id": "1",
                    "reward_info": {"reward": 1.0},
                    "messages": [],
                },
            ],
        },
    )

    def logged_call(call_id, tool_name):
        return {
            "call_id": call_id,
            "call_name": "agent_response",
            "request": {
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "allowed_tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            "response": {"tool_calls": [{"name": tool_name, "arguments": {}}]},
        }

    for simulation_id, tools in (
        ("failed", ("allowed_tool", "invented_tool", "allowed_tool")),
        ("passed", ("allowed_tool", "allowed_tool", "allowed_tool")),
    ):
        for index, tool_name in enumerate(tools):
            write_json(
                run_dir
                / "artifacts"
                / "task_1"
                / f"sim_{simulation_id}"
                / "llm_debug"
                / f"{index:03d}.json",
                logged_call(f"{simulation_id}-{index}", tool_name),
            )

    manifest, selected = build_manifest(
        run_dir,
        call_selection="event",
        pair_successes=True,
        event_before=None,
        event_after=1,
    )

    assert manifest["summary"]["selected_cases"] == 2
    assert manifest["summary"]["selected_calls"] == 6
    assert manifest["summary"]["localized_failures"] == 1
    assert manifest["summary"]["paired_success_cases"] == 1
    assert manifest["summary"]["alignment_rows"] == 3
    failure = next(case for case in manifest["cases"] if case["outcome"] == "failure")
    assert failure["first_error"]["call_index"] == 1
    assert failure["first_error"]["confidence"] == "verified"
    assert {item.case.simulation_id for item in selected} == {"failed", "passed"}
    assert sorted(
        item.relative_to_failure
        for item in selected
        if item.case.simulation_id == "failed"
    ) == [-1, 0, 1]


def test_boundary_and_generated_span_scoring_cover_all_semantic_points(tmp_path):
    class FakeModel:
        tokenizer = FakeTokenizer()
        n_layers = 3
        d_model = 4
        layers = []
        input_device = torch.device("cpu")

    class FakeLens:
        source_layers = [0, 1]

        @staticmethod
        def apply(model, prompt, *, layers, positions, max_seq_len):
            del model, prompt, max_seq_len
            logits = torch.zeros((len(positions), 256))
            return {layer: logits.clone() for layer in layers}, logits.clone(), None

    case = Tau2Case(
        task_id="5",
        simulation_id="failed",
        reward=0.0,
        expected_tools=("cancel_reservation",),
        actual_tools=("get_reservation_details",),
        task={},
        simulation={"reward_info": {"reward": 0.0}},
    )
    call = LoggedCall(tmp_path / "call.json", sample_call(), call_index=0)
    selected = SelectedCall(case=case, call=call, relative_to_failure=0)

    boundary_rows, replay = score_semantic_boundaries(
        selected,
        model=FakeModel(),
        lens=FakeLens(),
        enable_thinking=False,
        layer_stride=1,
        max_seq_len=4096,
    )
    span_rows = score_generated_spans(
        selected,
        replay,
        model=FakeModel(),
        lens=FakeLens(),
        layer_stride=1,
        max_seq_len=4096,
    )

    assert {row["boundary"] for row in boundary_rows} == {
        "observation",
        "decision",
        "tool",
        "argument",
    }
    assert {row["span_kind"] for row in span_rows} == {
        "tool_name",
        "argument_value",
    }
    assert {row["layer"] for row in boundary_rows} == {0, 1, "model_final"}
