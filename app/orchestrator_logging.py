"""Colored terminal logging for orchestrator agent state and orchestration contract."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

try:
    from colorama import Fore, Back, Style, init
except ImportError:
    # Fallback if colorama not installed
    class ForeStub:
        CYAN = YELLOW = GREEN = BLUE = MAGENTA = WHITE = RED = ""
    class BackStub:
        pass
    class StyleStub:
        BRIGHT = RESET_ALL = ""
    Fore = ForeStub()
    Back = BackStub()
    Style = StyleStub()
    def init(*args, **kwargs):  # pragma: no cover
        pass


# Initialize colorama (autoreset=True means colors reset after each print)
try:
    init(autoreset=True)
except TypeError:
    # Older colorama versions might not support all parameters
    init(autoreset=True)

# Agent color mappings
AGENT_COLORS = {
    "planner": Fore.CYAN,
    "retriever": Fore.GREEN,
    "reasoner": Fore.YELLOW,
    "generator": Fore.BLUE,
    "critic": Fore.MAGENTA,
    "orchestrator": Fore.WHITE,
}


def _format_agent_badge(agent: str) -> str:
    """Format agent name with color and styling."""
    color = AGENT_COLORS.get(agent, Fore.WHITE)
    return f"{color}[{agent.upper()}]{Style.RESET_ALL}"


def _format_orchestration_header() -> str:
    """Format the orchestration contract header."""
    return f"{Fore.WHITE}{Style.BRIGHT}╔═══ ORCHESTRATION CONTRACT ═══╗{Style.RESET_ALL}"


def _format_orchestration_footer() -> str:
    """Format the orchestration contract footer."""
    return f"{Fore.WHITE}{Style.BRIGHT}╚════════════════════════════════╝{Style.RESET_ALL}"


def _format_contract_line(key: str, value: str, indent: int = 2) -> str:
    """Format a single line of the orchestration contract."""
    spaces = " " * indent
    return f"{Fore.WHITE}{spaces}├─ {key}:{Style.RESET_ALL} {value}"


def display_orchestration_contract(orchestration: Dict[str, Any]) -> None:
    """Display the orchestration contract in colored format to stderr.
    
    Args:
        orchestration: Dict containing route_family, agent roles, tool ownership, etc.
    """
    if not orchestration:
        return
    
    print(_format_orchestration_header(), file=sys.stderr)
    
    # Route family
    route_family = orchestration.get("route_family", "general")
    print(_format_contract_line("Route Family", f"{Fore.CYAN}{route_family}{Style.RESET_ALL}"), file=sys.stderr)
    
    # Agent requirements
    agents_needed = []
    if orchestration.get("planner_required"):
        agents_needed.append(f"{Fore.CYAN}planner{Style.RESET_ALL}")
    if orchestration.get("retriever_required"):
        agents_needed.append(f"{Fore.GREEN}retriever{Style.RESET_ALL}")
    if orchestration.get("reasoner_required"):
        agents_needed.append(f"{Fore.YELLOW}reasoner{Style.RESET_ALL}")
    if orchestration.get("generator_required"):
        agents_needed.append(f"{Fore.BLUE}generator{Style.RESET_ALL}")
    if orchestration.get("critic_required"):
        agents_needed.append(f"{Fore.MAGENTA}critic{Style.RESET_ALL}")
    
    if agents_needed:
        agents_str = ", ".join(agents_needed)
        print(_format_contract_line("Agents", agents_str), file=sys.stderr)
    
    # Tool sequence
    tool_sequence = orchestration.get("tool_sequence", [])
    if tool_sequence:
        tools_str = " → ".join(tool_sequence)
        print(_format_contract_line("Tool Sequence", f"{Fore.YELLOW}{tools_str}{Style.RESET_ALL}"), file=sys.stderr)
    
    # Tool ownership
    tool_owner_map = orchestration.get("tool_owner_map", {})
    if tool_owner_map:
        owner_lines = [f"{tool}:{owner}" for tool, owner in tool_owner_map.items()]
        owner_str = ", ".join(owner_lines)
        print(_format_contract_line("Tool Ownership", f"{Fore.CYAN}{owner_str}{Style.RESET_ALL}"), file=sys.stderr)
    
    # Validation owner
    validation_owner = orchestration.get("validation_owner")
    if validation_owner:
        print(_format_contract_line("Validation", f"{Fore.YELLOW}{validation_owner}{Style.RESET_ALL}"), file=sys.stderr)
    
    # Memory sources
    memory_sources = orchestration.get("memory_sources", [])
    if memory_sources:
        sources_str = ", ".join(memory_sources)
        print(_format_contract_line("Memory Sources", f"{Fore.GREEN}{sources_str}{Style.RESET_ALL}"), file=sys.stderr)
    
    # Short-circuit policy
    can_short_circuit = orchestration.get("can_short_circuit", False)
    short_circuit_str = f"{Fore.GREEN}yes{Style.RESET_ALL}" if can_short_circuit else f"{Fore.RED}no{Style.RESET_ALL}"
    print(_format_contract_line("Can Short-Circuit", short_circuit_str), file=sys.stderr)
    
    print(_format_orchestration_footer(), file=sys.stderr)
    print("", file=sys.stderr)  # Blank line


def colored_event_sink(event: Dict[str, Any]) -> None:
    """Colored terminal event sink for displaying agent activity.
    
    This is the default event sink if none is provided. It prints colored
    agent state and progress to stderr.
    
    Args:
        event: Event dict from _emit_event() with keys:
               event_id, run_id, timestamp, event_type, agent, stage, status,
               message, tool, attempt, duration_ms, result_count, error
    """
    agent = event.get("agent", "unknown")
    event_type = event.get("event_type", "")
    status = event.get("status", "")
    message = event.get("message", "")
    tool = event.get("tool")
    attempt = event.get("attempt")
    duration_ms = event.get("duration_ms")
    result_count = event.get("result_count")
    error = event.get("error")
    
    agent_badge = _format_agent_badge(agent)
    
    # Build message prefix with event type indicator
    if event_type == "run_started":
        prefix = f"{agent_badge} {Fore.GREEN}▶ START{Style.RESET_ALL}"
    elif event_type == "run_finished":
        status_color = Fore.GREEN if status == "completed" else Fore.YELLOW
        prefix = f"{agent_badge} {status_color}◼ FINISH{Style.RESET_ALL}"
    elif event_type == "agent_started":
        prefix = f"{agent_badge} {Fore.CYAN}→ START{Style.RESET_ALL}"
    elif event_type == "agent_finished":
        status_color = Fore.GREEN if status == "completed" else Fore.YELLOW
        prefix = f"{agent_badge} {status_color}◼ DONE{Style.RESET_ALL}"
    elif event_type == "agent_progress":
        prefix = f"{agent_badge} {Fore.WHITE}◆ PROGRESS{Style.RESET_ALL}"
    elif event_type == "tool_started":
        prefix = f"{agent_badge} {Fore.YELLOW}⚙ TOOL-START{Style.RESET_ALL}"
    elif event_type == "tool_finished":
        prefix = f"{agent_badge} {Fore.GREEN}⚙ TOOL-DONE{Style.RESET_ALL}"
    elif event_type == "retry_started":
        prefix = f"{agent_badge} {Fore.YELLOW}⟳ RETRY{Style.RESET_ALL}"
    elif event_type == "run_failed":
        prefix = f"{agent_badge} {Fore.RED}✗ FAILED{Style.RESET_ALL}"
    else:
        prefix = f"{agent_badge} {Fore.WHITE}• {event_type.upper()}{Style.RESET_ALL}"
    
    # Build suffix with metadata
    suffix_parts = []
    if tool:
        suffix_parts.append(f"tool={Fore.YELLOW}{tool}{Style.RESET_ALL}")
    if attempt is not None:
        suffix_parts.append(f"attempt={attempt}")
    if duration_ms is not None:
        suffix_parts.append(f"took={duration_ms}ms")
    if result_count is not None:
        suffix_parts.append(f"results={result_count}")
    
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    
    # Build and print the log line
    log_line = f"{prefix} | {Fore.WHITE}{message}{Style.RESET_ALL}{suffix}"
    print(log_line, file=sys.stderr)
    
    # Print error details if present
    if error:
        error_line = f"  {Fore.RED}ERROR: {error}{Style.RESET_ALL}"
        print(error_line, file=sys.stderr)


def create_default_event_sink() -> Any:
    """Create and return the default colored event sink function.
    
    Returns:
        The colored_event_sink function suitable for use as an event_sink callback.
    """
    return colored_event_sink
