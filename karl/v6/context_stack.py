"""Assembles the 7-block context stack for the twin prompt.

Total budget: 1848 tokens input + 200 tokens generation = 2048.
"""

from .session_state import SessionState
from .phase_prompts import get_system_prompt


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_context_stack(
    state: SessionState,
    terminal_content: str,
    max_total_tokens: int = 1848,
) -> list[dict]:
    """Build the 7-block context stack as chat messages.

    Returns messages list for the OpenAI-compatible API.
    """
    # Block 1: System prompt (phase-aware) — 300t budget
    system = get_system_prompt(state.phase)
    system = truncate_to_tokens(system, 300)

    # Block 2: Session brief — 200t budget
    brief = state.to_brief()
    brief = truncate_to_tokens(brief, 200)

    # Block 3: Task graph snapshot — 150t budget
    task_section = ""
    if state.plan_steps:
        done = state.goals_completed
        remaining = [s for s in state.plan_steps if s not in done]
        done_count = len(done)
        lines = [f"Goals completed: {done_count}"]
        for i, step in enumerate(remaining[:3]):
            marker = "ACTIVE" if i == 0 else "PENDING"
            lines.append(f"  [{marker}] {step}")
        if len(remaining) > 3:
            lines.append(f"  ... +{len(remaining)-3} more")
        task_section = "\n".join(lines)
    task_section = truncate_to_tokens(task_section, 150)

    # Block 4: Session digest — 150t budget
    digest = state.to_digest()
    digest = truncate_to_tokens(digest, 150)

    # Block 5: Anti-repeat fence — 100t budget
    fence = ""
    if state.last_prompts:
        forbidden = state.last_prompts[-5:]
        fence = "DO NOT repeat any of these recent prompts:\n"
        for p in forbidden:
            fence += f"  - {p[:60]}\n"
    fence = truncate_to_tokens(fence, 100)

    # Block 6: Terminal context — remaining budget
    used = sum(estimate_tokens(b) for b in [system, brief, task_section, digest, fence])
    terminal_budget = max_total_tokens - used - 48  # 48 for user turn
    terminal = truncate_to_tokens(terminal_content, max(terminal_budget, 200))

    # Block 7: User turn — 48t
    user_turn = "What does Mohamed type next? One sentence. Be specific."

    # Assemble as messages
    context_parts = []
    if brief:
        context_parts.append(f"SESSION:\n{brief}")
    if task_section:
        context_parts.append(f"TASKS:\n{task_section}")
    if digest:
        context_parts.append(f"HISTORY:\n{digest}")
    if fence:
        context_parts.append(fence)
    context_parts.append(f"CURRENT TERMINAL OUTPUT:\n{terminal}")

    user_content = "\n\n".join(context_parts) + f"\n\n{user_turn}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    return messages
