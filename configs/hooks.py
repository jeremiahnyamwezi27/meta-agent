from __future__ import annotations

"""Baseline C: Vanilla + hook functions.

Adds two hooks:
- PreToolUse on Bash: detects repeated failing commands and injects guidance
- Stop: rejects premature stops and forces verification before completing
"""

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from typing import Any

from meta_agent.run_context import RunContext

_recent_bash_commands: list[tuple[str, bool]] = []


async def detect_bash_loops(
    input_data: dict[str, Any], tool_use_id: str | None, context: Any
) -> dict[str, Any]:
    """Detect repeated failing Bash commands and inject recovery guidance."""
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        return {}

    failing_recent = [
        cmd for cmd, succeeded in _recent_bash_commands[-5:]
        if not succeeded
    ]

    if len(failing_recent) >= 3:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": (
                    "WARNING: The last 3+ Bash commands failed. "
                    "Stop and reconsider your approach. Read error messages carefully. "
                    "Try a fundamentally different strategy instead of retrying similar commands."
                ),
            }
        }

    repeated = [
        cmd for cmd, _ in _recent_bash_commands[-4:]
        if cmd.strip() == command.strip()
    ]
    if len(repeated) >= 2:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": (
                    f"WARNING: You are repeating the same command: `{command[:100]}`. "
                    "This suggests you're stuck in a loop. Try a different approach."
                ),
            }
        }

    return {}


async def track_bash_result(
    input_data: dict[str, Any], tool_use_id: str | None, context: Any
) -> dict[str, Any]:
    """Track Bash command outcomes for loop detection."""
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")
    tool_response = input_data.get("tool_response")

    is_error = False
    if isinstance(tool_response, dict):
        is_error = tool_response.get("exitCode", 0) != 0
    elif hasattr(tool_response, "is_error"):
        is_error = bool(tool_response.is_error)

    _recent_bash_commands.append((command, not is_error))
    if len(_recent_bash_commands) > 20:
        _recent_bash_commands.pop(0)

    return {}


async def force_verification_on_stop(
    input_data: dict[str, Any], tool_use_id: str | None, context: Any
) -> dict[str, Any]:
    """Reject the first stop attempt and require verification."""
    stop_active = input_data.get("stop_hook_active", False)

    if not stop_active:
        return {
            "reason": (
                "Before completing, verify your solution actually works: "
                "1. Run any tests if they exist. "
                "2. Check that your code compiles/runs without errors. "
                "3. Verify the output matches what was requested. "
                "Only stop after confirming the solution is correct."
            ),
            "continue_": True,
        }

    return {}


def build_options(ctx: RunContext) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code"},
        tools={"type": "preset", "preset": "claude_code"},
        cwd=ctx.cwd,
        model=ctx.model,
        permission_mode="bypassPermissions",
        max_turns=200,
        max_budget_usd=10.0,
        thinking={"type": "adaptive"},
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[detect_bash_loops]),
            ],
            "PostToolUse": [
                HookMatcher(matcher="Bash", hooks=[track_bash_result]),
            ],
            "Stop": [
                HookMatcher(hooks=[force_verification_on_stop]),
            ],
        },
    )
