"""LLM access: the live Claude client and the bounded decide-validate-repair loop.

Nodes never talk to an SDK directly. They call `decide()` with any object
that has `structured(node, schema, prompt)` and `text(node, prompt)` -
`ClaudeLLM` for live runs, `ScriptedLLM` (fake_llm.py) for tests and the
offline demo.

Retry budget, so retries never multiply silently:
- Transient HTTP failures are retried by the Anthropic SDK, configured
  explicitly here as `max_retries=SDK_MAX_RETRIES`.
- `decide()` never retries API errors itself. It re-asks the model at most
  `max_repairs` times (RunConfig caps this at 1), and only for responses
  that failed schema parsing or semantic validation.
Worst case per decision: (1 + max_repairs) * (1 + SDK_MAX_RETRIES) requests.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel

from .errors import API_ERROR, SCHEMA_ERROR, SEMANTIC_ERROR, ConfigError, DecisionError

SDK_MAX_RETRIES = 2
REQUEST_TIMEOUT_S = 60.0


@dataclass
class StructuredResponse:
    parsed: Optional[BaseModel]
    error: Optional[str]  # set when the response could not be parsed into the schema
    usage: Optional[dict]


@dataclass
class TextResponse:
    text: str
    usage: Optional[dict]


def resolve_model_name(env=None) -> str:
    """Read CLAUDE_MODEL at call time (after .env is loaded), never at import time."""
    env = os.environ if env is None else env
    name = (env.get("CLAUDE_MODEL") or "").strip()
    if not name:
        raise ConfigError(
            "CLAUDE_MODEL is not set. Put a model identifier your Anthropic account "
            "can call in .env (see .env.example)."
        )
    return name


def _usage(message) -> Optional[dict]:
    meta = getattr(message, "usage_metadata", None)
    if not meta:
        return None
    return {"input_tokens": meta.get("input_tokens"), "output_tokens": meta.get("output_tokens")}


def _text_content(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class ClaudeLLM:
    mode = "live"

    def __init__(self, model_name: str, max_tokens: int = 2048):
        from langchain_anthropic import ChatAnthropic

        self.model_name = model_name
        self._chat = ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
            max_retries=SDK_MAX_RETRIES,
            timeout=REQUEST_TIMEOUT_S,
        )

    def structured(self, node: str, schema: type[BaseModel], prompt: str) -> StructuredResponse:
        out = self._chat.with_structured_output(schema, include_raw=True).invoke(prompt)
        usage = _usage(out.get("raw"))
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            error = out.get("parsing_error") or "model returned no structured output"
            return StructuredResponse(None, str(error)[:500], usage)
        return StructuredResponse(out["parsed"], None, usage)

    def text(self, node: str, prompt: str) -> TextResponse:
        message = self._chat.invoke(prompt)
        return TextResponse(_text_content(message), _usage(message))


def llm_event(node: str, candidate_id, attempt: int, status: str, elapsed: float, **extra) -> dict:
    event = {
        "node": node,
        "candidate_id": candidate_id,
        "type": "llm_call",
        "llm_attempt": attempt,
        "status": status,
        "elapsed_s": round(elapsed, 3),
    }
    event.update(extra)
    return event


def decide(
    llm,
    node: str,
    schema: type[BaseModel],
    prompt: str,
    validate: Callable[[BaseModel], list[str]],
    max_repairs: int,
    candidate_id: Optional[str] = None,
) -> tuple[BaseModel, list[dict]]:
    """Ask for a structured decision, validate it, and allow bounded repair.

    Returns (decision, events). Raises DecisionError (carrying the events)
    if no valid decision is obtained.
    """
    events: list[dict] = []
    current_prompt = prompt
    kind, problems = SCHEMA_ERROR, []

    for attempt in range(max_repairs + 1):
        started = time.perf_counter()
        try:
            response = llm.structured(node, schema, current_prompt)
        except Exception as exc:  # network, auth, quota, timeouts after SDK retries
            message = f"{type(exc).__name__}: {str(exc)[:300]}"
            events.append(llm_event(node, candidate_id, attempt, API_ERROR,
                                    time.perf_counter() - started, error=message, usage=None))
            raise DecisionError(API_ERROR, [message], events) from exc

        if response.parsed is None:
            kind, problems = SCHEMA_ERROR, [response.error or "unparseable response"]
        else:
            kind, problems = SEMANTIC_ERROR, validate(response.parsed)

        status = "ok" if not problems else kind
        events.append(llm_event(node, candidate_id, attempt, status, time.perf_counter() - started,
                                validation_errors=problems, usage=response.usage,
                                is_repair=attempt > 0))
        if not problems:
            return response.parsed, events

        current_prompt = (
            f"{prompt}\n\nYour previous response was rejected:\n"
            + "\n".join(f"- {p}" for p in problems[:10])
            + "\nReturn a corrected response that fixes every problem above."
        )

    raise DecisionError(kind, problems, events)
