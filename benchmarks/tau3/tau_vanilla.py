"""Baseline tau3-bench config: vanilla ClaudeAgentOptions."""
from claude_agent_sdk import ClaudeAgentOptions
from meta_agent.run_context import RunContext


def build_options(ctx: RunContext) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code"},
        tools={"type": "preset", "preset": "claude_code"},
        cwd=ctx.cwd,
        model=ctx.model,
        permission_mode="bypassPermissions",
        max_turns=50,
        max_budget_usd=5.0,
        thinking={"type": "adaptive"},
    )
