"""Baseline B: Vanilla + environment bootstrap prompt.

Injects instructions to inspect the environment before starting work.
This is the key discovery from the Meta-Harness paper's TB2 run —
agents waste 3-5 turns probing what's installed. Telling the agent
to inspect first saves those turns on dependency-heavy tasks.
"""

from claude_agent_sdk import ClaudeAgentOptions

from meta_agent.run_context import RunContext

BOOTSTRAP_PROMPT = """\
Before writing any code or making changes, spend your first turn understanding the environment:
1. Run `ls` to see what files and directories exist
2. Check available languages and tools: `which python3 node gcc g++ rustc go java 2>/dev/null`
3. Check package managers: `which pip apt-get npm cargo 2>/dev/null`
4. Check available memory: `free -h 2>/dev/null || cat /proc/meminfo 2>/dev/null | head -3`
5. Read any README or instruction files in the working directory

Use this information to plan your approach before making any changes.\
"""


def build_options(ctx: RunContext) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": BOOTSTRAP_PROMPT,
        },
        tools={"type": "preset", "preset": "claude_code"},
        cwd=ctx.cwd,
        model=ctx.model,
        permission_mode="bypassPermissions",
        max_turns=200,
        max_budget_usd=10.0,
        thinking={"type": "adaptive"},
    )
