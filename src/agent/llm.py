"""Central place to construct the Claude client used by every agent node."""

import os

from langchain_anthropic import ChatAnthropic

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")


def get_llm(temperature: float = 0) -> ChatAnthropic:
    return ChatAnthropic(model=DEFAULT_MODEL, temperature=temperature, max_tokens=4096)
