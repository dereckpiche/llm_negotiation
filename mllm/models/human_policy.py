import asyncio
import os
import re
import shutil
import sys
from typing import Callable, Dict, List, Optional

try:
    import rstr  # For generating example strings from regex
except Exception:  # pragma: no cover
    rstr = None


def _clear_terminal() -> None:
    """
    Clear the terminal screen in a cross-platform manner.
    """
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def _terminal_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def _horizontal_rule(char: str = "─") -> str:
    width = max(20, _terminal_width() - 2)
    return char * width


class _Style:
    # ANSI colors (bright, readable)
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Foreground colors
    FG_BLUE = "\033[94m"     # user/system headers
    FG_GREEN = "\033[92m"    # human response header
    FG_YELLOW = "\033[93m"   # notices
    FG_RED = "\033[91m"      # errors
    FG_MAGENTA = "\033[95m"  # regex
    FG_CYAN = "\033[96m"     # tips


def _render_chat(prompt: List[Dict]) -> str:
    """
    Render prior messages in a compact, readable terminal format.

    Expected message dict keys: {"role": str, "content": str, ...}
    """
    lines: List[str] = []
    lines.append(_horizontal_rule())
    lines.append(f"{_Style.FG_BLUE}{_Style.BOLD} Conversation so far {_Style.RESET}")
    lines.append(_horizontal_rule())
    for msg in prompt:
        role = msg.get("role", "user").strip()
        content = str(msg.get("content", "")).strip()
        # Map roles to display names and colors/emojis
        if role == "assistant":
            header = f"{_Style.FG_GREEN}{_Style.BOLD}HUMAN--🧑‍💻{_Style.RESET}"
        elif role == "user":
            header = f"{_Style.FG_BLUE}{_Style.BOLD}USER--⚙️{_Style.RESET}"
        else:
            header = f"[{_Style.DIM}{role.upper()}{_Style.RESET}]"
        lines.append(header)
        # Indent content for readability
        for line in content.splitlines() or [""]:
            lines.append(f"  {line}")
        lines.append("")
    lines.append(_horizontal_rule())
    return "\n".join(lines)


async def _async_input(prompt_text: str) -> str:
    """Non-blocking input using a background thread."""
    return await asyncio.to_thread(input, prompt_text)


def _short_regex_example(regex: str, max_len: int = 30) -> Optional[str]:
    """
    Try to produce a short example string that matches the regex.
    We attempt multiple times and pick the first <= max_len.
    """
    if rstr is None:
        return None
    try:
        for _ in range(20):
            candidate = rstr.xeger(regex)
            if len(candidate) <= max_len:
                return candidate
        # Fallback to truncation (may break match, so don't return)
        return None
    except Exception:
        return None


async def human_policy(prompt: List[Dict], regex: str | None = None) -> str:
    """
    Async human-in-the-loop policy.

    - Displays prior conversation context in the terminal.
    - Prompts the user for a response.
    - If a regex is provided, validates and re-prompts until it matches.

    Args:
        prompt: Chat history as a list of {role, content} dicts.
        regex: Optional fullmatch validation pattern.

    Returns:
        The user's validated response string.
    """
    while True:
        _clear_terminal()
        print(_render_chat(prompt))

        if regex:
            example = _short_regex_example(regex, max_len=30)
            print(f"{_Style.FG_MAGENTA}{_Style.BOLD}Expected format (regex fullmatch):{_Style.RESET}")
            print(f"  {_Style.FG_MAGENTA}{regex}{_Style.RESET}")
            if example:
                print(f"{_Style.FG_CYAN}Example (random, <=30 chars):{_Style.RESET} {example}")
            print(_horizontal_rule("."))
            print(f"{_Style.FG_YELLOW}Type your response and press Enter.{_Style.RESET}")
            print(f"{_Style.DIM}Commands: /help to view commands, /refresh to re-render, /quit to abort{_Style.RESET}")
        else:
            print(f"{_Style.FG_YELLOW}Type your response and press Enter.{_Style.RESET} {_Style.DIM}(/help for commands){_Style.RESET}")

        user_in = (await _async_input("> ")).rstrip("\n")

        # Commands
        if user_in.strip().lower() in {"/help", "/h"}:
            print(f"\n{_Style.FG_CYAN}{_Style.BOLD}Available commands:{_Style.RESET}")
            print(f"  {_Style.FG_CYAN}/help{_Style.RESET} or {_Style.FG_CYAN}/h{_Style.RESET}     Show this help")
            print(f"  {_Style.FG_CYAN}/refresh{_Style.RESET} or {_Style.FG_CYAN}/r{_Style.RESET}  Re-render the conversation and prompt")
            print(f"  {_Style.FG_CYAN}/quit{_Style.RESET} or {_Style.FG_CYAN}/q{_Style.RESET}     Abort the run (raises KeyboardInterrupt)")/quit
            await asyncio.sleep(1.0)
            continue
        if user_in.strip().lower() in {"/refresh", "/r"}:
            continue
        if user_in.strip().lower() in {"/quit", "/q"}:
            raise KeyboardInterrupt("Human aborted run from human_policy")

        if regex is None:
            return user_in

        # Validate against regex (fullmatch)
        try:
            pattern = re.compile(regex)
        except re.error as e:
            # If regex is invalid, fall back to accepting any input
            print(f"{_Style.FG_RED}Warning:{_Style.RESET} Provided regex is invalid: {e}. Accepting input without validation.")
            await asyncio.sleep(0.5)
            return user_in

        if pattern.fullmatch(user_in):
            return user_in

        # Show validation error and re-prompt
        print("")
        print(f"{_Style.FG_RED}{_Style.BOLD}Input did not match the required format.{_Style.RESET} Please try again.")
        print(f"Expected (regex):")
        print(f"  {_Style.FG_MAGENTA}{regex}{_Style.RESET}")
        print(_horizontal_rule("."))
        print(f"{_Style.FG_YELLOW}Press Enter to retry...{_Style.RESET}")
        await _async_input("")


def get_human_policies() -> Dict[str, Callable[[List[Dict]], str]]:
    """
    Expose the human policy in the same map shape used elsewhere.
    """
    # Type hint says Callable[[List[Dict]], str] but we intentionally return the async callable.
    return {"human_policy": human_policy}  # type: ignore[return-value]


