"""Agent implementations for experiment control.

Importing this package auto-registers all built-in agent backends.
"""

from safe_lab_agents.agents.base import (  # noqa: F401
    AGENT_REGISTRY,
    BaseAgent,
    ConversationEntry,
    get_agent,
    list_agents,
    register_agent,
)

# Auto-import agent modules so they self-register via @register_agent.
from safe_lab_agents.agents import claude_code as _claude_code  # noqa: F401
from safe_lab_agents.agents import openclaw as _openclaw  # noqa: F401
