"""Failure categories used across the pipeline.

Keeping these distinct lets the trace and report say *why* something failed:

- schema_error:   the LLM response could not be parsed into the pydantic schema
- semantic_error: it parsed, but the decision is invalid for this data/task
- tool_error:     a deterministic data-science step raised while executing
- api_error:      the external model call itself failed (network, auth, quota)
"""

SCHEMA_ERROR = "schema_error"
SEMANTIC_ERROR = "semantic_error"
TOOL_ERROR = "tool_error"
API_ERROR = "api_error"


class ConfigError(ValueError):
    """Bad run configuration or environment (raised before the graph starts)."""


class DatasetError(ValueError):
    """The dataset cannot be evaluated as-is (missing values, bad dtypes, too small)."""


class DecisionError(Exception):
    """An LLM decision could not be obtained, even after the allowed repair attempts."""

    def __init__(self, kind: str, messages: list[str], events: list[dict]):
        super().__init__(f"{kind}: {'; '.join(messages)}")
        self.kind = kind
        self.messages = messages
        self.events = events
