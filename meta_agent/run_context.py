"""RunContext — runtime context passed to config modules' build_options()."""

from dataclasses import dataclass


@dataclass
class RunContext:
    """Runtime values available when building ClaudeAgentOptions inside a container.

    Config modules receive this and use it to set fields like cwd and model
    that aren't known until execution time.
    """

    cwd: str
    model: str
    task_instruction: str
