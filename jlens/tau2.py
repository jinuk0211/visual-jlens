# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Adapters for replaying tau2 agent calls through a Jacobian lens.

The tau2 ``results.json`` file supplies task labels and rewards, while verbose
LLM logs contain the exact messages and tool schemas sent to the agent.  This
module deliberately has no dependency on tau2 so a saved run can be analysed
in the lightweight Jacobian-lens environment.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import torch

JSONDict = dict[str, Any]
AGENT_CALL_NAMES = frozenset(
    {"agent_response", "agent_gt_response", "agent_solo_response"}
)


@dataclass(frozen=True)
class Tau2Case:
    """One saved tau2 simulation and the labels needed for analysis."""

    task_id: str
    simulation_id: str
    reward: float | None
    expected_tools: tuple[str, ...]
    actual_tools: tuple[str, ...]
    task: JSONDict
    simulation: JSONDict

    @property
    def failed(self) -> bool:
        """Whether tau2 assigned a non-perfect reward."""
        return self.reward is not None and self.reward < 1.0

    @property
    def remaining_expected_tools(self) -> tuple[str, ...]:
        """Expected assistant actions not already present in the trajectory."""
        actual_counts = Counter(self.actual_tools)
        remaining: list[str] = []
        for name in self.expected_tools:
            if actual_counts[name]:
                actual_counts[name] -= 1
            else:
                remaining.append(name)
        return tuple(remaining)


@dataclass(frozen=True)
class LoggedCall:
    """A verbose tau2 LLM log for one agent response."""

    path: Path
    data: JSONDict
    call_index: int = -1
    turn_index: int | None = None

    @property
    def actual_tools(self) -> tuple[str, ...]:
        response = self.data.get("response") or {}
        return _tool_names(response.get("tool_calls") or [])


@dataclass(frozen=True)
class RenderedCall:
    """A logged request rendered with the model's Hugging Face chat template."""

    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class FailureEvent:
    """Conservatively localized first externally verifiable error."""

    call_index: int | None
    turn_index: int | None
    kind: str
    source: str
    confidence: Literal["verified", "reviewed", "terminal", "unlocalized"]
    reason: str


@dataclass(frozen=True)
class GeneratedSpan:
    """One generated tool-name or argument-value token span."""

    kind: Literal["tool_name", "argument_value"]
    label: str
    start: int
    end: int
    token_ids: tuple[int, ...]

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        """Positions whose next-token logits predict this span."""
        if self.start < 1:
            raise ValueError(f"{self.kind} span has no preceding token")
        return tuple(range(self.start - 1, self.end - 1))


@dataclass(frozen=True)
class SemanticBoundary:
    """A semantic source position in a reconstructed request or response."""

    name: str
    context: Literal["request", "response", "update"]
    position: int
    label: str | None = None


@dataclass(frozen=True)
class ReplayedResponse:
    """Exact pre-response request plus the logged response under teacher forcing."""

    request: RenderedCall
    response: RenderedCall
    boundaries: tuple[SemanticBoundary, ...]
    generated_spans: tuple[GeneratedSpan, ...]


@dataclass(frozen=True)
class ToolCandidate:
    """Teacher-forced tool name and positions whose logits predict its tokens."""

    name: str
    text: str
    token_ids: tuple[int, ...]
    name_token_ids: tuple[int, ...]
    name_start: int
    prediction_positions: tuple[int, ...]


@dataclass(frozen=True)
class LogitScore:
    """Aggregate and per-token scores for one teacher-forced candidate."""

    sum_logprob: float
    mean_logprob: float
    ranks: tuple[int, ...]
    mean_rank: float
    max_rank: int
    top_token_ids: tuple[int, ...]


def _read_json(path: Path) -> JSONDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _results_file(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return path
    return path / "results.json"


def _saved_simulations(results_file: Path, results: JSONDict) -> list[JSONDict]:
    simulations = results.get("simulations") or []
    if simulations:
        return [
            simulation for simulation in simulations if isinstance(simulation, dict)
        ]

    simulations_dir = results_file.parent / "simulations"
    if not simulations_dir.is_dir():
        return []
    return [_read_json(path) for path in sorted(simulations_dir.glob("*.json"))]


def _reward_of(simulation: Mapping[str, Any]) -> float | None:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward") if isinstance(reward_info, Mapping) else None
    if reward is None and isinstance(simulation.get("reward"), (int, float)):
        reward = simulation["reward"]
    return float(reward) if isinstance(reward, (int, float)) else None


def _tool_name(tool_call: Mapping[str, Any]) -> str | None:
    name = tool_call.get("name")
    if isinstance(name, str):
        return name
    function = tool_call.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _tool_names(tool_calls: Sequence[Any]) -> tuple[str, ...]:
    names = []
    for tool_call in tool_calls:
        if isinstance(tool_call, Mapping) and (name := _tool_name(tool_call)):
            names.append(name)
    return tuple(names)


def _tool_arguments(tool_call: Mapping[str, Any]) -> JSONDict | None:
    arguments: Any = tool_call.get("arguments")
    function = tool_call.get("function")
    if arguments is None and isinstance(function, Mapping):
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return dict(arguments) if isinstance(arguments, Mapping) else None


def _tool_call_id(tool_call: Mapping[str, Any]) -> str | None:
    value = tool_call.get("id")
    return str(value) if value is not None else None


def _assistant_messages(simulation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message
        for message in simulation.get("messages") or []
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]


def _normalized_content(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value).strip()
    return str(value or "").strip()


def _call_matches_message(call: LoggedCall, message: Mapping[str, Any]) -> bool:
    response = call.data.get("response") or {}
    if not isinstance(response, Mapping):
        return False
    response_tools = _tool_names(response.get("tool_calls") or [])
    message_tools = _tool_names(message.get("tool_calls") or [])
    if response_tools or message_tools:
        return response_tools == message_tools
    return _normalized_content(response.get("content")) == _normalized_content(
        message.get("content")
    )


def _actual_tools(simulation: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for message in simulation.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        names.extend(_tool_names(message.get("tool_calls") or []))
    return tuple(names)


def _expected_tools(task: Mapping[str, Any]) -> tuple[str, ...]:
    criteria = task.get("evaluation_criteria") or {}
    if not isinstance(criteria, Mapping):
        return ()
    actions = criteria.get("actions") or []
    names: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        if action.get("requestor", "assistant") != "assistant":
            continue
        name = action.get("name")
        if isinstance(name, str):
            names.append(name)
    return tuple(names)


def load_cases(path: str | Path) -> list[Tau2Case]:
    """Load text simulations from a tau2 result file or run directory."""
    results_file = _results_file(path)
    results = _read_json(results_file)
    tasks = {
        str(task["id"]): task
        for task in results.get("tasks") or []
        if isinstance(task, dict) and "id" in task
    }

    cases = []
    for simulation in _saved_simulations(results_file, results):
        task_id = str(simulation.get("task_id"))
        simulation_id = str(simulation.get("id"))
        task = tasks.get(task_id, {})
        cases.append(
            Tau2Case(
                task_id=task_id,
                simulation_id=simulation_id,
                reward=_reward_of(simulation),
                expected_tools=_expected_tools(task),
                actual_tools=_actual_tools(simulation),
                task=task,
                simulation=simulation,
            )
        )
    return cases


def _tool_schemas(call: LoggedCall) -> dict[str, Mapping[str, Any]]:
    request = call.data.get("request") or {}
    schemas: dict[str, Mapping[str, Any]] = {}
    for tool in request.get("tools") or []:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        definition = function if isinstance(function, Mapping) else tool
        name = definition.get("name")
        if isinstance(name, str):
            schemas[name] = definition
    return schemas


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    types: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "object": Mapping,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    python_type = types.get(str(expected))
    if python_type is None:
        return True
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def _schema_error(call: LoggedCall) -> str | None:
    """Return a conservative, locally verifiable tool-call schema error."""
    schemas = _tool_schemas(call)
    response = call.data.get("response") or {}
    if not isinstance(response, Mapping):
        return "response is not an object"
    for tool_call in response.get("tool_calls") or []:
        if not isinstance(tool_call, Mapping):
            return "tool call is not an object"
        name = _tool_name(tool_call)
        if name is None:
            return "tool call has no function name"
        if name not in schemas:
            return f"emitted unknown tool {name!r}"
        arguments = _tool_arguments(tool_call)
        if arguments is None:
            function = tool_call.get("function")
            raw_arguments = tool_call.get("arguments")
            if raw_arguments is None and isinstance(function, Mapping):
                raw_arguments = function.get("arguments")
            if raw_arguments is None:
                arguments = {}
            else:
                return f"arguments for {name!r} are not a JSON object"
        parameters = schemas[name].get("parameters") or {}
        if not isinstance(parameters, Mapping):
            continue
        required = parameters.get("required") or []
        missing = [key for key in required if key not in arguments]
        if missing:
            return f"{name!r} is missing required arguments {missing!r}"
        properties = parameters.get("properties") or {}
        if isinstance(properties, Mapping):
            for key, value in arguments.items():
                definition = properties.get(key)
                if isinstance(definition, Mapping) and not _matches_json_type(
                    value, definition.get("type")
                ):
                    return (
                        f"argument {key!r} for {name!r} violates type "
                        f"{definition.get('type')!r}"
                    )
        if parameters.get("additionalProperties") is False:
            extras = sorted(set(arguments) - set(properties))
            if extras:
                return f"{name!r} has unsupported arguments {extras!r}"
    return None


def _tool_error_ids(simulation: Mapping[str, Any]) -> set[str]:
    error_ids: set[str] = set()
    for message in simulation.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        if message.get("error") is True and message.get("id") is not None:
            error_ids.add(str(message["id"]))
    return error_ids


def _call_has_tool_error(call: LoggedCall, error_ids: set[str]) -> bool:
    response = call.data.get("response") or {}
    if not isinstance(response, Mapping):
        return False
    return any(
        isinstance(tool_call, Mapping)
        and (call_id := _tool_call_id(tool_call)) is not None
        and call_id in error_ids
        for tool_call in response.get("tool_calls") or []
    )


def _review_error(case: Tau2Case, calls: Sequence[LoggedCall]) -> FailureEvent | None:
    review = case.simulation.get("review") or {}
    if not isinstance(review, Mapping):
        return None
    errors = []
    for error in review.get("errors") or []:
        if not isinstance(error, Mapping) or error.get("source") != "agent":
            continue
        turn_index = error.get("turn_idx")
        if isinstance(turn_index, int):
            errors.append((turn_index, error))
    if not errors:
        return None
    turn_index, error = min(errors, key=lambda item: item[0])
    matched = next((call for call in calls if call.turn_index == turn_index), None)
    if matched is None:
        matched = next(
            (
                call
                for call in calls
                if call.turn_index is not None and call.turn_index >= turn_index
            ),
            None,
        )
    if matched is None:
        return None
    reason = error.get("reasoning") or error.get("error_type") or "reviewer error"
    return FailureEvent(
        call_index=matched.call_index,
        turn_index=matched.turn_index,
        kind=str(error.get("error_type") or "reviewed_agent_error"),
        source="tau2_review",
        confidence="reviewed",
        reason=str(reason),
    )


def _manual_error(
    calls: Sequence[LoggedCall], label: Mapping[str, Any]
) -> FailureEvent:
    matched: LoggedCall | None = None
    if isinstance(label.get("call_index"), int):
        matched = next(
            (call for call in calls if call.call_index == label["call_index"]), None
        )
    if matched is None and label.get("call_id") is not None:
        call_id = str(label["call_id"])
        matched = next(
            (
                call
                for call in calls
                if str(call.data.get("call_id", call.path.stem)) == call_id
            ),
            None,
        )
    if matched is None and isinstance(label.get("turn_index"), int):
        matched = next(
            (call for call in calls if call.turn_index == label["turn_index"]), None
        )
    if matched is None:
        raise ValueError("manual first-error label does not identify a saved agent call")
    return FailureEvent(
        call_index=matched.call_index,
        turn_index=matched.turn_index,
        kind=str(label.get("kind") or "manually_audited_error"),
        source="manual_label",
        confidence="verified",
        reason=str(label.get("reason") or "manually audited first error"),
    )


def infer_first_error(
    case: Tau2Case,
    calls: Sequence[LoggedCall],
    *,
    manual_label: Mapping[str, Any] | None = None,
    allow_review_labels: bool = True,
) -> FailureEvent | None:
    """Locate the earliest defensible failure without forcing ambiguous cases."""
    if not case.failed:
        return None
    if manual_label is not None:
        return _manual_error(calls, manual_label)

    verified: list[FailureEvent] = []
    error_ids = _tool_error_ids(case.simulation)
    for call in calls:
        if (reason := _schema_error(call)) is not None:
            verified.append(
                FailureEvent(
                    call_index=call.call_index,
                    turn_index=call.turn_index,
                    kind="tool_schema_error",
                    source="logged_request_schema",
                    confidence="verified",
                    reason=reason,
                )
            )
        if _call_has_tool_error(call, error_ids):
            verified.append(
                FailureEvent(
                    call_index=call.call_index,
                    turn_index=call.turn_index,
                    kind="tool_execution_error",
                    source="tau2_tool_result",
                    confidence="verified",
                    reason="tau2 recorded error=true for a tool result from this call",
                )
            )
    if verified:
        return min(
            verified,
            key=lambda event: event.call_index
            if event.call_index is not None
            else len(calls),
        )

    if allow_review_labels and (reviewed := _review_error(case, calls)) is not None:
        return reviewed

    reward_info = case.simulation.get("reward_info") or {}
    action_checks = (
        reward_info.get("action_checks") if isinstance(reward_info, Mapping) else None
    )
    db_check = reward_info.get("db_check") if isinstance(reward_info, Mapping) else None
    missing_action = any(
        isinstance(check, Mapping) and check.get("action_match") is False
        for check in action_checks or []
    )
    db_diverged = isinstance(db_check, Mapping) and db_check.get("db_match") is False
    termination_reason = str(case.simulation.get("termination_reason") or "")
    if calls and missing_action and not db_diverged and termination_reason in {
        "agent_stop",
        "max_steps",
        "timeout",
    }:
        last = calls[-1]
        return FailureEvent(
            call_index=last.call_index,
            turn_index=last.turn_index,
            kind="premature_termination",
            source="tau2_completion_checks",
            confidence="terminal",
            reason="agent terminated with required actions still unsatisfied",
        )

    return FailureEvent(
        call_index=None,
        turn_index=None,
        kind="unlocalized_failure",
        source="tau2_end_state",
        confidence="unlocalized",
        reason=(
            "the end-state reward proves failure, but saved artifacts do not "
            "conservatively identify the first erroneous call"
        ),
    )


def select_cases(
    cases: Sequence[Tau2Case],
    *,
    include_successes: bool = False,
    limit: int | None = None,
) -> list[Tau2Case]:
    """Select failed cases by default, preserving the saved run order."""
    selected = (
        list(cases) if include_successes else [case for case in cases if case.failed]
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]
    return selected


CallT = TypeVar("CallT")


def select_calls(
    calls: Sequence[CallT], selection: Literal["last", "all"]
) -> list[CallT]:
    """Select all logged calls or only the final call for a simulation."""
    if selection == "all":
        return list(calls)
    if selection == "last":
        return list(calls[-1:])
    raise ValueError(f"unknown call selection {selection!r}")


def select_event_calls(
    calls: Sequence[LoggedCall],
    anchor_index: int | None,
    *,
    before: int | None = None,
    after: int = 1,
) -> list[LoggedCall]:
    """Select a first-error window; ``before=None`` retains all prior calls."""
    if after < 0:
        raise ValueError("event window after must be non-negative")
    if before is not None and before < 0:
        raise ValueError("event window before must be non-negative or None")
    if anchor_index is None:
        return list(calls)
    start = 0 if before is None else max(0, anchor_index - before)
    stop = min(len(calls), anchor_index + after + 1)
    return [call for call in calls if start <= call.call_index < stop]


def candidate_tool_names(case: Tau2Case, call: LoggedCall) -> tuple[str, ...]:
    """Return unmet expected tools followed by tools emitted in this call."""
    expected = case.remaining_expected_tools or case.expected_tools
    return tuple(dict.fromkeys((*expected, *call.actual_tools)))


def discover_agent_calls(run_dir: str | Path, case: Tau2Case) -> list[LoggedCall]:
    """Find verbose agent-call logs associated with ``case``."""
    log_dir = (
        Path(run_dir)
        / "artifacts"
        / f"task_{case.task_id}"
        / f"sim_{case.simulation_id}"
        / "llm_debug"
    )
    raw_calls = []
    for path in sorted(log_dir.glob("*.json")):
        data = _read_json(path)
        if data.get("call_name") in AGENT_CALL_NAMES:
            raw_calls.append(LoggedCall(path=path, data=data))

    assistant_messages = _assistant_messages(case.simulation)
    calls: list[LoggedCall] = []
    message_cursor = 0
    for call_index, call in enumerate(raw_calls):
        matched: Mapping[str, Any] | None = None
        for message_index in range(message_cursor, len(assistant_messages)):
            message = assistant_messages[message_index]
            if _call_matches_message(call, message):
                matched = message
                message_cursor = message_index + 1
                break
        turn_index = None
        if matched is not None and isinstance(matched.get("turn_idx"), int):
            turn_index = int(matched["turn_idx"])
        calls.append(
            LoggedCall(
                path=call.path,
                data=call.data,
                call_index=call_index,
                turn_index=turn_index,
            )
        )
    return calls


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[JSONDict]:
    """Restore logged messages to the shape expected by HF chat templates.

    tau2 stores historical OpenAI-style tool arguments as JSON strings.  The
    Qwen chat templates expect those arguments to be mappings and call
    ``.items()`` on them, so replay must deserialize valid JSON objects first.
    """
    normalized: list[JSONDict] = []
    for message in messages:
        value = copy.deepcopy(dict(message))
        if isinstance(value.get("content"), list):
            value["content"] = "\n".join(str(line) for line in value["content"])
        tool_calls = value.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    continue
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed_arguments, Mapping):
                    function["arguments"] = dict(parsed_arguments)
        normalized.append(value)
    return normalized


def _chat_template_kwargs(call: Mapping[str, Any], enable_thinking: bool) -> JSONDict:
    request = call.get("request") or {}
    messages = request.get("messages") or []
    kwargs: JSONDict = {
        "conversation": normalize_messages(messages),
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if request.get("tools"):
        kwargs["tools"] = request["tools"]
    return kwargs


def _as_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError("chat template did not return a flat token-id list")
    return tuple(value)


def render_logged_call(
    tokenizer: Any, call: Mapping[str, Any], *, enable_thinking: bool
) -> RenderedCall:
    """Render the exact pre-response context recorded by tau2."""
    kwargs = _chat_template_kwargs(call, enable_thinking)
    text = tokenizer.apply_chat_template(**kwargs)
    token_ids = tokenizer.apply_chat_template(**{**kwargs, "tokenize": True})
    if not isinstance(text, str):
        raise TypeError("chat template did not return text")
    return RenderedCall(text=text, token_ids=_as_token_ids(token_ids))


def _render_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Any],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> RenderedCall:
    kwargs: JSONDict = {
        "conversation": normalize_messages(messages),
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    if tools:
        kwargs["tools"] = list(tools)
    text = tokenizer.apply_chat_template(**kwargs)
    token_ids = tokenizer.apply_chat_template(**{**kwargs, "tokenize": True})
    if not isinstance(text, str):
        raise TypeError("chat template did not return text")
    return RenderedCall(text=text, token_ids=_as_token_ids(token_ids))


def _response_message(call: Mapping[str, Any]) -> JSONDict:
    response = call.get("response") or {}
    if not isinstance(response, Mapping):
        raise ValueError("logged response is not a JSON object")
    message: JSONDict = {
        "role": "assistant",
        "content": copy.deepcopy(response.get("content")),
    }
    normalized_calls: list[JSONDict] = []
    for tool_call in response.get("tool_calls") or []:
        if not isinstance(tool_call, Mapping):
            continue
        name = _tool_name(tool_call)
        if name is None:
            continue
        arguments = _tool_arguments(tool_call)
        function = tool_call.get("function")
        raw_arguments = (
            function.get("arguments") if isinstance(function, Mapping) else None
        )
        if arguments is None and isinstance(raw_arguments, str):
            rendered_arguments: Any = raw_arguments
        else:
            rendered_arguments = arguments or {}
        value: JSONDict = {
            "type": "function",
            "function": {"name": name, "arguments": rendered_arguments},
        }
        if (call_id := _tool_call_id(tool_call)) is not None:
            value["id"] = call_id
        normalized_calls.append(value)
    if normalized_calls:
        message["tool_calls"] = normalized_calls
    return message


def _token_offsets_for_ids(
    tokenizer: Any, text: str, token_ids: Sequence[int]
) -> list[tuple[int, int]]:
    """Map exact rendered token IDs to character offsets without retokenizing.

    A decoded BPE string can have more than one valid tokenization.  Fast
    tokenizer offsets are safe only when their IDs equal the IDs returned by
    ``apply_chat_template``; otherwise offsets are recovered from those exact
    rendered IDs.
    """
    exact_ids = tuple(int(token_id) for token_id in token_ids)
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        canonical_ids = _as_token_ids(encoded)
        raw_offsets = encoded["offset_mapping"]
        if hasattr(raw_offsets, "tolist"):
            raw_offsets = raw_offsets.tolist()
        if (
            canonical_ids == exact_ids
            and isinstance(raw_offsets, list)
            and len(raw_offsets) == len(exact_ids)
        ):
            return [(int(pair[0]), int(pair[1])) for pair in raw_offsets]
    except (KeyError, TypeError, ValueError):
        pass

    decode_kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    decoded = tokenizer.decode(exact_ids, **decode_kwargs)
    if decoded != text:
        raise ValueError(
            "rendered token IDs do not decode to rendered text; refusing a "
            "non-exact boundary mapping"
        )

    pieces = [tokenizer.decode([token_id], **decode_kwargs) for token_id in exact_ids]
    if "".join(pieces) == text:
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for piece in pieces:
            end = cursor + len(piece)
            offsets.append((cursor, end))
            cursor = end
        return offsets

    # Byte-level tokenizers can decode partial UTF-8 tokens as replacement
    # characters.  Prefix decoding lets adjacent token bytes combine while
    # retaining an exact, monotonic mapping to the rendered text.
    boundaries = [0]
    stable = 0
    for end in range(1, len(exact_ids) + 1):
        prefix = tokenizer.decode(exact_ids[:end], **decode_kwargs)
        common = 0
        limit = min(len(prefix), len(text))
        while common < limit and prefix[common] == text[common]:
            common += 1
        if common < stable:
            raise ValueError(
                "rendered token decoder produced non-monotonic text prefixes; "
                "refusing to guess token boundaries"
            )
        stable = common
        boundaries.append(stable)
    if boundaries[-1] != len(text):
        raise ValueError(
            "rendered token IDs could not be aligned exactly to rendered text"
        )

    offsets = []
    for index in range(len(exact_ids)):
        start, end = boundaries[index], boundaries[index + 1]
        if start == end and start < len(text):
            next_index = index + 2
            while next_index < len(boundaries) and boundaries[next_index] <= start:
                next_index += 1
            if next_index < len(boundaries):
                end = boundaries[next_index]
        offsets.append((start, end))
    return offsets


def _token_span_for_chars(
    offsets: Sequence[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Return the exact tokens overlapping a non-empty character span."""
    if char_start < 0 or char_end <= char_start:
        raise ValueError(f"invalid character span [{char_start}, {char_end})")
    hits = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < char_end and end > char_start
    ]
    if not hits:
        raise ValueError(
            f"character span [{char_start}, {char_end}) maps to no tokens"
        )
    return min(hits), max(hits) + 1


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        length += 1
    return length


def _flatten_argument_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            values.extend(_flatten_argument_values(item, label))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            label = f"{prefix}[{index}]"
            values.extend(_flatten_argument_values(item, label))
    else:
        values.append((prefix or "value", value))
    return values


def _argument_text_candidates(value: Any) -> list[str]:
    return list(
        dict.fromkeys(
            (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                json.dumps(value, ensure_ascii=False),
                str(value),
            )
        )
    )


def render_actual_response(
    tokenizer: Any, call: Mapping[str, Any], *, enable_thinking: bool
) -> ReplayedResponse:
    """Teacher-force the logged response and locate semantic token boundaries."""
    request = call.get("request") or {}
    if not isinstance(request, Mapping):
        raise ValueError("logged request is not a JSON object")
    messages = normalize_messages(request.get("messages") or [])
    tools = request.get("tools") or []
    pre_response = render_logged_call(
        tokenizer, call, enable_thinking=enable_thinking
    )
    observation = _render_messages(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    response_message = _response_message(call)
    response = _render_messages(
        tokenizer,
        [*messages, response_message],
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    observation_prefix = _common_prefix_length(
        observation.token_ids, pre_response.token_ids
    )
    if observation_prefix == 0:
        raise ValueError("observation context is not a prefix of the decision request")

    boundaries = [
        SemanticBoundary(
            name="observation",
            context="request",
            position=observation_prefix - 1,
        ),
        SemanticBoundary(
            name="decision",
            context="request",
            position=len(pre_response.token_ids) - 1,
        ),
    ]
    spans: list[GeneratedSpan] = []
    response_char_prefix = _common_prefix_length(observation.text, response.text)
    search_start = response_char_prefix
    response_offsets = _token_offsets_for_ids(
        tokenizer, response.text, response.token_ids
    )
    response_calls = (call.get("response") or {}).get("tool_calls") or []
    for tool_index, tool_call in enumerate(response_calls):
        if not isinstance(tool_call, Mapping) or (name := _tool_name(tool_call)) is None:
            continue
        name_char_start = response.text.find(name, search_start)
        if name_char_start < 0:
            raise ValueError(f"tool name {name!r} was not found in logged response")
        name_char_end = name_char_start + len(name)
        name_start, name_end = _token_span_for_chars(
            response_offsets, name_char_start, name_char_end
        )
        name_ids = tuple(response.token_ids[name_start:name_end])
        label = f"tool[{tool_index}]={name}"
        spans.append(
            GeneratedSpan(
                kind="tool_name",
                label=label,
                start=name_start,
                end=name_end,
                token_ids=name_ids,
            )
        )
        boundaries.append(
            SemanticBoundary(
                name="tool",
                context="response",
                position=name_start - 1,
                label=label,
            )
        )

        argument_search = name_char_end
        arguments = _tool_arguments(tool_call)
        for argument_label, value in _flatten_argument_values(arguments or {}):
            match: tuple[int, int] | None = None
            for candidate in _argument_text_candidates(value):
                start = response.text.find(candidate, argument_search)
                if start >= 0 and (match is None or start < match[0]):
                    match = (start, start + len(candidate))
            if match is None:
                continue
            value_char_start, value_char_end = match
            value_start, value_end = _token_span_for_chars(
                response_offsets, value_char_start, value_char_end
            )
            value_ids = tuple(response.token_ids[value_start:value_end])
            span_label = f"tool[{tool_index}].{argument_label}"
            spans.append(
                GeneratedSpan(
                    kind="argument_value",
                    label=span_label,
                    start=value_start,
                    end=value_end,
                    token_ids=value_ids,
                )
            )
            boundaries.append(
                SemanticBoundary(
                    name="argument",
                    context="response",
                    position=value_start - 1,
                    label=span_label,
                )
            )
            argument_search = value_char_end
        search_start = max(name_char_end, argument_search)

    return ReplayedResponse(
        request=pre_response,
        response=response,
        boundaries=tuple(boundaries),
        generated_spans=tuple(spans),
    )


def build_tool_candidate(
    tokenizer: Any,
    call: Mapping[str, Any],
    tool_name: str,
    *,
    enable_thinking: bool,
) -> ToolCandidate:
    """Render a teacher-forced assistant tool call for layer-wise scoring."""
    request = call.get("request") or {}
    messages = normalize_messages(request.get("messages") or [])
    candidate_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": tool_name, "arguments": {}},
            }
        ],
    }
    kwargs: JSONDict = {
        "conversation": [*messages, candidate_message],
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": enable_thinking,
    }
    if request.get("tools"):
        kwargs["tools"] = request["tools"]

    text = tokenizer.apply_chat_template(**kwargs)
    token_ids = _as_token_ids(
        tokenizer.apply_chat_template(**{**kwargs, "tokenize": True})
    )
    context = _render_messages(
        tokenizer,
        messages,
        tools=request.get("tools") or [],
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    name_char_start = text.find(
        tool_name, _common_prefix_length(context.text, text)
    )
    if name_char_start < 0:
        raise ValueError("tool name was not found in the rendered assistant tool call")
    offsets = _token_offsets_for_ids(tokenizer, text, token_ids)
    name_start, name_end = _token_span_for_chars(
        offsets, name_char_start, name_char_start + len(tool_name)
    )
    name_token_ids = tuple(token_ids[name_start:name_end])
    prediction_positions = tuple(range(name_start - 1, name_end - 1))
    if prediction_positions[0] < 0:
        raise ValueError("tool name has no preceding token to score")
    if not isinstance(text, str):
        raise TypeError("chat template did not return text")
    return ToolCandidate(
        name=tool_name,
        text=text,
        token_ids=token_ids,
        name_token_ids=name_token_ids,
        name_start=name_start,
        prediction_positions=prediction_positions,
    )


def summarize_logits(logits: torch.Tensor, target_ids: Sequence[int]) -> LogitScore:
    """Score one target token at each row of a ``[tokens, vocab]`` tensor."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocab]")
    if logits.shape[0] != len(target_ids) or not target_ids:
        raise ValueError("target_ids must contain one token id per logits row")

    targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    row = torch.arange(len(target_ids), device=logits.device)
    target_logits = logits[row, targets]
    token_logprobs = target_logits - torch.logsumexp(logits, dim=-1)
    ranks = (logits > target_logits[:, None]).sum(dim=-1) + 1
    top_token_ids = logits.argmax(dim=-1)
    return LogitScore(
        sum_logprob=float(token_logprobs.sum().item()),
        mean_logprob=float(token_logprobs.mean().item()),
        ranks=tuple(int(rank) for rank in ranks.tolist()),
        mean_rank=float(ranks.float().mean().item()),
        max_rank=int(ranks.max().item()),
        top_token_ids=tuple(int(token_id) for token_id in top_token_ids.tolist()),
    )
