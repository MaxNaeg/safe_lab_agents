"""Sentinel MCP tool file for the end-to-end pipeline tests.

Exposes a single deterministic ``ping`` tool. The agent under test is instructed
to call it exactly once; the test then asserts that the unique ``SENTINEL`` string
flowed all the way through the pipeline into the session's ``history.json``.
"""

# Unique, greppable marker string. Assertions look for this in the tool output
# recorded in history.json, proving the MCP tool round-trip actually happened
# inside the real container.
SENTINEL = "e2e-ping-ok-7f3a"


def ping(note: str = "") -> dict:
    """Health-check tool. Call this exactly once, then stop.

    Args:
        note: Optional free-text note echoed back in the result.
    """
    return {"status": SENTINEL, "echo": note}


MCP_TOOLS = [ping]
