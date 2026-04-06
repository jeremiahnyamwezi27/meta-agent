"""Run tau-bench tasks through the Claude Agent SDK.

Wraps tau-bench tools + user simulator as MCP tools so the full SDK
surface (hooks, MCP tools, subagents, stop hooks) is available for
harness optimization.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    ResultMessage,
    ToolUseBlock,
)

from meta_agent.run_context import RunContext


@dataclass
class ConversationState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    tau2_trajectory: list = field(default_factory=list)
    tau2_trajectory_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    next_tool_call_idx: int = 0


def build_mcp_tools(env: Any, user: Any, state: ConversationState) -> list:
    """Create MCP tool wrappers for tau-bench env tools + user simulator."""
    mcp_tools = []

    @tool(
        "talk_to_customer",
        "Send a message to the customer and receive their response. Use this to communicate.",
        {"message": str},
    )
    async def talk_to_customer(args: dict[str, Any]) -> dict[str, Any]:
        try:
            from tau2.data_model.message import AssistantMessage as TauAssistantMsg

            from tau2.data_model.message import UserMessage as TauUserMsg

            agent_msg = TauAssistantMsg(role="assistant", content=args["message"])
            state.messages.append({"role": "assistant", "content": args["message"]})
            state.tau2_trajectory.append(agent_msg)

            if not hasattr(state, "_user_state"):
                state._user_state = user.get_init_state()

            user_msg, state._user_state = await asyncio.to_thread(
                user.generate_next_message, agent_msg, state._user_state
            )
            user_text = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
            state.messages.append({"role": "user", "content": user_text})
            state.tau2_trajectory.append(TauUserMsg.text(content=user_text))

            return {"content": [{"type": "text", "text": f"Customer: {user_text}"}]}
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            return {"content": [{"type": "text", "text": f"Error in talk_to_customer: {err}"}]}

    mcp_tools.append(talk_to_customer)

    def _make_tool_handler(name: str, desc: str, schema: dict) -> Any:
        @tool(name, desc, schema)
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            try:
                from tau2.data_model.message import AssistantMessage as _TauAM, ToolCall as _TauTC, ToolMessage as _TauTM

                # Match official tau2 ordering constraints: each tool-call message
                # must be immediately followed by its corresponding tool message.
                async with state.tau2_trajectory_lock:
                    error = False
                    try:
                        result = await asyncio.to_thread(env.make_tool_call, name, **args)
                    except Exception as tool_err:
                        result = f"Error: {tool_err}"
                        error = True

                    result_str = env.to_json_str(result)
                    tc_id = f"tc_{state.next_tool_call_idx}"
                    state.next_tool_call_idx += 1

                    state.tool_call_log.append(
                        {"tool": name, "args": args, "result": str(result)[:500], "error": error}
                    )
                    state.tau2_trajectory.append(
                        _TauAM(
                            role="assistant",
                            tool_calls=[_TauTC(id=tc_id, name=name, arguments=args, requestor="assistant")],
                        )
                    )
                    state.tau2_trajectory.append(
                        _TauTM(
                            id=tc_id,
                            role="tool",
                            content=result_str,
                            requestor="assistant",
                            error=error,
                        )
                    )

                return {"content": [{"type": "text", "text": result_str}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}]}
        return handler

    for tau_tool in env.get_tools():
        t_name = tau_tool.name
        t_desc = tau_tool.short_desc if hasattr(tau_tool, "short_desc") else t_name
        t_schema = tau_tool.openai_schema["function"]["parameters"]
        mcp_tools.append(_make_tool_handler(t_name, t_desc, t_schema))

    return mcp_tools


@dataclass
class SDKTaskResult:
    task_id: str
    domain: str
    reward: float
    passed: bool
    gold_reward: float
    num_turns: int
    cost_usd: float | None
    duration_s: float
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    session_id: str | None


TAU_JUDGE_SYSTEM_BINARY = (
    "You are evaluating a customer service agent. Given the company policy, "
    "the conversation with the customer, and the database operations performed, "
    "determine if the agent correctly resolved the customer's issue according "
    'to policy. Respond with ONLY "correct" or "incorrect".'
)

TAU_JUDGE_SYSTEM_CRITIQUE = (
    "You are evaluating a customer service agent. Given the company policy, "
    "the conversation with the customer, and the database operations performed, "
    "determine if the agent correctly resolved the customer's issue according "
    "to policy. First, identify the customer's request. Then check each policy "
    "rule that applies. Then verify the database operations match what was needed. "
    'Write your analysis, then end with your final verdict on a new line: "CORRECT" or "INCORRECT".'
)


def _parse_verdict(text: str) -> bool:
    lowered = text.strip().lower()
    return "correct" in lowered and "incorrect" not in lowered


@dataclass
class TauJudgeResult:
    correct: bool
    explanation: str | None = None


def _judge_tau_task(
    domain: str,
    policy: str,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    judge_model: str,
    strategy: str = "binary",
) -> TauJudgeResult:
    """Ask an LLM judge whether a customer service agent correctly resolved the issue."""
    import os

    conversation = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in messages
    )

    tool_summary = "\n".join(
        f"  {tc['tool']}({json.dumps(tc['args'])[:200]}) -> {tc['result'][:200]}"
        for tc in tool_calls
    ) or "(no database operations)"

    user_content = (
        f"DOMAIN: {domain}\n\n"
        f"POLICY:\n{policy[:8000]}\n\n"
        f"CONVERSATION:\n{conversation}\n\n"
        f"DATABASE OPERATIONS:\n{tool_summary}\n\n"
        f"Did the agent correctly resolve the customer's issue according to policy?"
    )

    if strategy == "self":
        return _judge_tau_self(user_content, judge_model)

    system_prompt = TAU_JUDGE_SYSTEM_CRITIQUE if strategy == "critique" else TAU_JUDGE_SYSTEM_BINARY
    max_tokens = 1000 if strategy == "critique" else 10

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_completion_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    explanation = raw.strip() if strategy == "critique" else None
    return TauJudgeResult(correct=_parse_verdict(raw), explanation=explanation)


def _judge_tau_self(user_content: str, model: str) -> TauJudgeResult:
    """Same-family judge: Claude judging Claude via Bedrock."""
    import os
    import anthropic

    client = anthropic.AnthropicBedrock(
        api_key=os.environ.get("BEDROCK_BEARER_TOKEN") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
        aws_region=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1"),
    )
    response = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0,
        system=TAU_JUDGE_SYSTEM_BINARY,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text if response.content else ""
    return TauJudgeResult(correct=_parse_verdict(raw))


async def run_tau_task_sdk(
    domain: str,
    task_id: str,
    config_path: str,
    model: str,
    user_model: str | None = None,
    judge_model: str | None = None,
    judge_strategy: str = "binary",
) -> SDKTaskResult:
    """Run a single tau-bench task using the Claude Agent SDK."""
    from tau2.runner import get_tasks, build_environment, build_user

    tasks = get_tasks(domain, task_ids=[str(task_id)])
    task = tasks[0]
    env = build_environment(domain)
    if task.initial_state and hasattr(env, "set_state"):
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=task.initial_state.message_history or [],
        )
    user = build_user("user_simulator", env, task, llm=user_model or model)

    conv_state = ConversationState()
    mcp_tools = build_mcp_tools(env, user, conv_state)
    server = create_sdk_mcp_server(name="tau", tools=mcp_tools)

    tool_names = ["mcp__tau__talk_to_customer"] + [f"mcp__tau__{t.name}" for t in env.get_tools()]

    policy = env.get_policy()
    task_desc = str(task.user_scenario) if hasattr(task, "user_scenario") else str(task.description)

    tau_prompt = (
        f"You are a customer service agent for the {domain} domain.\n\n"
        f"TOOLS AVAILABLE:\n"
        f"- mcp__tau__talk_to_customer: Send a message to the customer, get their response\n"
        f"- mcp__tau__* database tools: Look up and modify reservations, bookings, etc.\n\n"
        f"WORKFLOW:\n"
        f"1. Greet the customer using mcp__tau__talk_to_customer\n"
        f"2. Listen to their request\n"
        f"3. Use database tools to look up relevant records\n"
        f"4. Follow the policy to determine the correct action\n"
        f"5. Execute any changes using the database tools\n"
        f"6. Confirm with the customer via mcp__tau__talk_to_customer\n"
        f"7. When the issue is resolved, stop.\n\n"
        f"POLICY (follow this exactly):\n{policy}"
    )

    import importlib.util
    spec = importlib.util.spec_from_file_location("harness_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import os
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
        _bedrock_map = {
            "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
            "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
        }
        model = _bedrock_map.get(model, model)

    ctx = RunContext(cwd="/tmp", model=model, task_instruction=task_desc)
    options = module.build_options(ctx)
    perm_override = os.environ.get("CLAUDE_PERMISSION_MODE")
    if perm_override and hasattr(options, "permission_mode") and options.permission_mode != perm_override:
        options.permission_mode = perm_override

    options.tools = None
    options.cwd = "/tmp"

    options.mcp_servers = {**(options.mcp_servers or {}), "tau": server}

    # Extend allowed_tools: tau tools + tools from config's custom MCP servers
    config_allowed = options.allowed_tools or []
    extra_tools = [t for t in config_allowed if not t.startswith("mcp__tau__")]
    options.allowed_tools = tool_names + extra_tools

    # Preserve config's append text, combine with tau context
    config_append = ""
    if isinstance(options.system_prompt, str):
        config_append = options.system_prompt
    elif isinstance(options.system_prompt, dict):
        config_append = options.system_prompt.get("append", "")
    options.system_prompt = f"{config_append}\n\n{tau_prompt}" if config_append else tau_prompt

    prompt = "A customer is calling. Greet them using mcp__tau__talk_to_customer and help resolve their issue."

    start = time.time()
    num_turns = 0
    cost_usd = None
    session_id = None
    sdk_error = False
    trace: list[dict[str, Any]] = []

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            num_turns += 1
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    trace.append({"type": "tool_call", "name": block.name, "input": str(block.input)[:200]})
        elif isinstance(message, ResultMessage):
            cost_usd = message.total_cost_usd
            session_id = message.session_id
            sdk_error = getattr(message, "is_error", False)

    duration = time.time() - start

    gold_reward = 0.0
    try:
        from tau2.data_model.simulation import SimulationRun as TauSimRun, TerminationReason as TauTermReason
        from tau2.evaluator.evaluator import evaluate_simulation as tau_evaluate, EvaluationType
        from datetime import datetime

        term_reason = TauTermReason.AGENT_ERROR if sdk_error else TauTermReason.AGENT_STOP

        init_msgs = (task.initial_state.message_history or []) if task.initial_state else []
        full_trajectory = list(init_msgs) + conv_state.tau2_trajectory

        sim_run = TauSimRun(
            id=f"sdk-{task_id}-{int(start)}",
            task_id=str(task_id),
            start_time=datetime.fromtimestamp(start).isoformat(),
            end_time=datetime.fromtimestamp(start + duration).isoformat(),
            duration=duration,
            termination_reason=term_reason,
            messages=full_trajectory,
        )

        reward_info = tau_evaluate(
            simulation=sim_run,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=domain,
        )
        gold_reward = reward_info.reward
    except Exception as e:
        import traceback
        print(f"[GOLD-EVAL ERROR] task={task_id}: {e}")
        traceback.print_exc()
        gold_reward = 0.0

    if judge_model:
        jr = await asyncio.to_thread(
            _judge_tau_task,
            domain, policy, conv_state.messages, conv_state.tool_call_log,
            judge_model, strategy=judge_strategy,
        )
        reward = 1.0 if jr.correct else 0.0
    else:
        reward = gold_reward

    return SDKTaskResult(
        task_id=str(task_id),
        domain=domain,
        reward=reward,
        passed=reward > 0,
        gold_reward=gold_reward,
        num_turns=num_turns,
        cost_usd=cost_usd,
        duration_s=duration,
        messages=conv_state.messages,
        tool_calls=conv_state.tool_call_log,
        session_id=session_id,
    )
