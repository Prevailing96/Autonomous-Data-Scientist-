"""A scripted stand-in for Claude, used by tests and the offline demo.

Responses are queued per node. Each queued item is one of:
- a dict or pydantic object: parsed into the requested schema (a dict that
  does not fit the schema is reported as a schema error, like a malformed
  live response),
- an Exception instance: raised, simulating an external API failure.

Nothing here calls a model, so it reports no token usage.
"""

from pydantic import BaseModel, ValidationError

from .llm import StructuredResponse, TextResponse


class ScriptedLLM:
    mode = "offline"
    model_name = "offline decision script"

    def __init__(self, script: dict[str, list], narrative="Scripted narrative."):
        self.script = {node: list(items) for node, items in script.items()}
        self.narrative = narrative
        self.calls: list[tuple[str, str]] = []  # (node, prompt), for assertions

    def structured(self, node: str, schema: type[BaseModel], prompt: str) -> StructuredResponse:
        self.calls.append((node, prompt))
        queue = self.script.get(node)
        if not queue:
            raise RuntimeError(f"ScriptedLLM has no response left for node '{node}'")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, BaseModel):
            item = item.model_dump()
        try:
            return StructuredResponse(schema.model_validate(item), None, None)
        except ValidationError as exc:
            return StructuredResponse(None, str(exc)[:500], None)

    def text(self, node: str, prompt: str) -> TextResponse:
        self.calls.append((node, prompt))
        if isinstance(self.narrative, Exception):
            raise self.narrative
        return TextResponse(self.narrative, None)
